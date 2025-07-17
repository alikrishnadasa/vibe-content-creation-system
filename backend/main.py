#!/usr/bin/env python3
"""
FastAPI Backend for Video Content Creation System
Integrates with Quantum Pipeline and BatchVideoGenerator
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import asyncio
import uuid
import json
import os
import sys
import shutil
from pathlib import Path
from datetime import datetime
import logging

# Add the unified-video-system-main to Python path
sys.path.append(str(Path(__file__).parent.parent / "unified-video-system-main"))

try:
    from core.quantum_pipeline import UnifiedQuantumPipeline
    from generate_batch_videos import BatchVideoGenerator, generate_videos
    from video_config import VideoConfig, DEFAULT_CONFIG, create_custom_config
    from core.real_content_generator import RealVideoRequest
except ImportError as e:
    print(f"Error importing video system: {e}")
    sys.exit(1)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Video Content Creation API", 
    version="1.0.0",
    description="AI-powered video generation with quantum pipeline"
)

# CORS middleware for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", 
        "http://localhost:3001",
        "https://your-frontend-domain.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for video downloads
output_dir = Path("../unified-video-system-main/output")
if output_dir.exists():
    app.mount("/videos", StaticFiles(directory=str(output_dir)), name="videos")

# Job storage and WebSocket connections
jobs: Dict[str, Dict[str, Any]] = {}
websocket_connections: Dict[str, WebSocket] = {}

# Global quantum pipeline instance
quantum_pipeline: Optional[UnifiedQuantumPipeline] = None

# Pydantic models
class VideoGenerationRequest(BaseModel):
    script_name: str
    caption_style: str = "default"
    variation_number: int = 1
    music_sync: bool = True
    burn_in_captions: bool = True

class BatchVideoRequest(BaseModel):
    num_videos: int = 5
    caption_style: str = "default"
    output_directory: Optional[str] = None

class JobStatus(BaseModel):
    job_id: str
    status: str  # pending, processing, completed, failed
    progress: float = 0.0
    message: str = ""
    created_at: datetime
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    request_type: str = "single"  # single, batch
    processing_time: Optional[float] = None

@app.on_event("startup")
async def startup_event():
    """Initialize the quantum pipeline on startup"""
    global quantum_pipeline
    try:
        logger.info("🚀 Initializing Quantum Pipeline...")
        quantum_pipeline = UnifiedQuantumPipeline()
        
        # Initialize real content mode
        config = DEFAULT_CONFIG
        success = await quantum_pipeline.initialize_real_content_mode(
            clips_directory=config.clips_directory,
            metadata_file=config.metadata_file,
            scripts_directory=config.scripts_directory,
            music_file=config.music_file
        )
        
        if success:
            logger.info("✅ Quantum Pipeline initialized successfully")
        else:
            logger.error("❌ Failed to initialize Quantum Pipeline")
            
    except Exception as e:
        logger.error(f"💥 Startup failed: {e}")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Video Content Creation API with Quantum Pipeline", 
        "status": "running",
        "quantum_pipeline": quantum_pipeline is not None
    }

@app.get("/api/health")
async def health_check():
    """Detailed health check"""
    config = DEFAULT_CONFIG
    
    # Check available scripts with cached captions
    cached_scripts = config.get_available_cached_scripts("default")
    
    return {
        "status": "healthy",
        "quantum_pipeline_ready": quantum_pipeline is not None,
        "clips_directory": config.clips_directory,
        "scripts_directory": config.scripts_directory,
        "output_directory": config.output_directory,
        "available_scripts": len(config.get_available_scripts()),
        "cached_caption_scripts": len(cached_scripts),
        "target_resolution": config.target_resolution,
        "caption_styles": ["default", "tiktok", "cinematic", "minimal", "youtube", "karaoke"]
    }

@app.get("/api/scripts")
async def get_scripts():
    """Get available script names with caption cache info"""
    try:
        config = DEFAULT_CONFIG
        script_files = config.get_available_scripts()
        
        scripts = []
        for script_file in script_files:
            script_name = script_file.stem
            has_cache = config.has_cached_captions(script_name, "default")
            
            scripts.append({
                "name": script_name,
                "path": str(script_file),
                "has_cached_captions": has_cache,
                "cache_path": str(config.get_caption_cache_path(script_name)) if has_cache else None
            })
        
        return {"scripts": scripts}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching scripts: {str(e)}")

@app.get("/api/config")
async def get_config():
    """Get current configuration and system status"""
    config = DEFAULT_CONFIG
    
    # Get quantum pipeline performance stats if available
    pipeline_stats = {}
    if quantum_pipeline:
        try:
            pipeline_stats = quantum_pipeline.get_performance_report()
        except Exception as e:
            logger.error(f"Failed to get pipeline stats: {e}")
            pipeline_stats = {"error": "Failed to get performance report"}
    
    return {
        "video_config": {
            "target_resolution": config.target_resolution,
            "target_fps": config.target_fps,
            "caption_style": config.caption_style,
            "burn_in_captions": config.burn_in_captions,
            "music_sync": config.music_sync
        },
        "available_styles": ["default", "tiktok", "cinematic", "minimal", "youtube", "karaoke"],
        "quantum_pipeline": {
            "initialized": quantum_pipeline is not None,
            "target_time": "0.7s",
            "stats": pipeline_stats
        }
    }

@app.post("/api/generate-video")
async def generate_video(request: VideoGenerationRequest, background_tasks: BackgroundTasks):
    """Generate a single video using quantum pipeline"""
    if not quantum_pipeline:
        raise HTTPException(status_code=500, detail="Quantum pipeline not initialized")
    
    job_id = str(uuid.uuid4())
    
    # Create job entry
    jobs[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "progress": 0.0,
        "message": "Quantum video generation queued",
        "created_at": datetime.now(),
        "completed_at": None,
        "result": None,
        "request": request.dict(),
        "request_type": "single"
    }
    
    # Start background task
    background_tasks.add_task(process_quantum_video, job_id, request)
    
    return {"job_id": job_id, "status": "pending", "message": "Video generation started with quantum pipeline"}

@app.post("/api/generate-batch")
async def generate_batch(request: BatchVideoRequest, background_tasks: BackgroundTasks):
    """Generate multiple videos using batch generator"""
    job_id = str(uuid.uuid4())
    
    # Create job entry
    jobs[job_id] = {
        "job_id": job_id,
        "status": "pending", 
        "progress": 0.0,
        "message": f"Batch generation queued: {request.num_videos} videos",
        "created_at": datetime.now(),
        "completed_at": None,
        "result": None,
        "request": request.dict(),
        "request_type": "batch"
    }
    
    # Start background task
    background_tasks.add_task(process_batch_videos, job_id, request)
    
    return {"job_id": job_id, "status": "pending", "message": f"Batch generation started: {request.num_videos} videos"}

@app.websocket("/api/ws/{job_id}")
async def websocket_endpoint(websocket: WebSocket, job_id: str):
    """WebSocket endpoint for real-time job updates"""
    await websocket.accept()
    websocket_connections[job_id] = websocket
    
    try:
        while True:
            # Send current job status if available
            if job_id in jobs:
                job_data = jobs[job_id]
                await websocket.send_json({
                    "job_id": job_id,
                    "status": job_data["status"],
                    "progress": job_data["progress"],
                    "message": job_data["message"],
                    "timestamp": datetime.now().isoformat()
                })
            
            await asyncio.sleep(1)  # Send updates every second
            
    except WebSocketDisconnect:
        if job_id in websocket_connections:
            del websocket_connections[job_id]

@app.get("/api/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Get job status"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    return JobStatus(**job)

@app.get("/api/jobs")
async def get_all_jobs():
    """Get all jobs"""
    job_list = []
    for job in jobs.values():
        job_list.append(JobStatus(**job))
    
    # Sort by creation time, newest first
    job_list.sort(key=lambda x: x.created_at, reverse=True)
    return {"jobs": job_list}

@app.get("/api/download/{job_id}")
async def download_video(job_id: str):
    """Download generated video"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job not completed")
    
    if not job["result"] or "output_path" not in job["result"]:
        raise HTTPException(status_code=400, detail="No output file available")
    
    output_path = job["result"]["output_path"]
    if not os.path.exists(output_path):
        raise HTTPException(status_code=404, detail="Output file not found")
    
    filename = Path(output_path).name
    return FileResponse(
        path=output_path,
        media_type="video/mp4",
        filename=filename
    )

@app.get("/api/outputs")
async def list_outputs():
    """List all generated videos"""
    try:
        config = DEFAULT_CONFIG
        output_dir = Path(config.output_directory)
        
        if not output_dir.exists():
            return {"videos": []}
        
        videos = []
        for video_file in output_dir.glob("*.mp4"):
            if video_file.is_file():
                stat = video_file.stat()
                videos.append({
                    "name": video_file.name,
                    "size": stat.st_size,
                    "size_mb": round(stat.st_size / (1024 * 1024), 1),
                    "created": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "path": str(video_file)
                })
        
        # Sort by creation time, newest first
        videos.sort(key=lambda x: x["created"], reverse=True)
        return {"videos": videos}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing outputs: {str(e)}")

@app.delete("/api/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete a job"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    del jobs[job_id]
    return {"message": "Job deleted"}

async def process_quantum_video(job_id: str, request: VideoGenerationRequest):
    """Background task to process single video with quantum pipeline"""
    try:
        await notify_progress(job_id, 10.0, "Initializing quantum pipeline...")
        
        # Generate video using quantum pipeline
        result = await quantum_pipeline.generate_real_content_video(
            script_name=request.script_name,
            variation_number=request.variation_number,
            caption_style=request.caption_style
        )
        
        await notify_progress(job_id, 90.0, "Finalizing video...")
        
        if result['success']:
            jobs[job_id]["status"] = "completed"
            jobs[job_id]["progress"] = 100.0
            jobs[job_id]["message"] = "Quantum video generation completed"
            jobs[job_id]["result"] = {
                "output_path": result['output_path'],
                "processing_time": result['processing_time'],
                "target_achieved": result['target_achieved'],
                "quantum_stats": result.get('statistics', {}),
                "file_size": os.path.getsize(result['output_path']) if os.path.exists(result['output_path']) else 0
            }
        else:
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["message"] = f"Generation failed: {result.get('error', 'Unknown error')}"
        
        jobs[job_id]["completed_at"] = datetime.now()
        jobs[job_id]["processing_time"] = result.get('processing_time', 0)
        
    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["message"] = f"Error: {str(e)}"
        jobs[job_id]["completed_at"] = datetime.now()
        logger.error(f"Quantum video generation failed for job {job_id}: {e}")

async def process_batch_videos(job_id: str, request: BatchVideoRequest):
    """Background task to process batch video generation"""
    try:
        await notify_progress(job_id, 5.0, "Setting up batch generation...")
        
        # Create custom config if needed
        config_kwargs = {}
        if request.output_directory:
            config_kwargs["output_directory"] = request.output_directory
        if request.caption_style:
            config_kwargs["caption_style"] = request.caption_style
        
        config = create_custom_config(**config_kwargs) if config_kwargs else DEFAULT_CONFIG
        
        await notify_progress(job_id, 10.0, f"Starting batch: {request.num_videos} videos...")
        
        # Generate videos using the working batch system
        success = await generate_videos(request.num_videos, config)
        
        if success:
            jobs[job_id]["status"] = "completed"
            jobs[job_id]["progress"] = 100.0
            jobs[job_id]["message"] = f"Batch completed: {request.num_videos} videos generated"
            jobs[job_id]["result"] = {
                "total_videos": request.num_videos,
                "output_directory": config.output_directory,
                "caption_style": request.caption_style
            }
        else:
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["message"] = "Batch generation encountered issues"
        
        jobs[job_id]["completed_at"] = datetime.now()
        
    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["message"] = f"Error: {str(e)}"
        jobs[job_id]["completed_at"] = datetime.now()
        logger.error(f"Batch generation failed for job {job_id}: {e}")

async def notify_progress(job_id: str, progress: float, message: str):
    """Update job progress and notify WebSocket clients"""
    if job_id in jobs:
        jobs[job_id]["progress"] = progress
        jobs[job_id]["message"] = message
        
        # Notify WebSocket client if connected
        if job_id in websocket_connections:
            try:
                await websocket_connections[job_id].send_json({
                    "job_id": job_id,
                    "progress": progress,
                    "message": message,
                    "timestamp": datetime.now().isoformat()
                })
            except:
                # Connection might be closed
                if job_id in websocket_connections:
                    del websocket_connections[job_id]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)