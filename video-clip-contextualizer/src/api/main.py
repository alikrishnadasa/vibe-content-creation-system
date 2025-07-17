from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import tempfile
import os
import uuid
import logging
from datetime import datetime
import asyncio

from ..processing.video_processor import VideoProcessor
from ..processing.text_processor import TextProcessor
from ..matching.semantic_matcher import SemanticMatcher
from ..config import get_config


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Video Clip Contextualizer API",
    description="AI-powered semantic video-to-script matching system",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load configuration
config = get_config()

# Initialize processors
video_processor = VideoProcessor()
text_processor = TextProcessor()
semantic_matcher = SemanticMatcher()

# Job storage (in production, use Redis or database)
active_jobs: Dict[str, Dict[str, Any]] = {}


# Request/Response models
class AnalyzeRequest(BaseModel):
    video_url: Optional[str] = None
    script: str
    clip_duration: Optional[float] = None
    overlap: Optional[float] = None
    language: str = "en"
    matching_strategy: str = "optimal"


class VideoSegmentResponse(BaseModel):
    start: float
    end: float


class ScriptSegmentResponse(BaseModel):
    start: int
    end: int


class ExplanationResponse(BaseModel):
    match_type: str
    video_duration: float
    text_length: int
    text_preview: str
    keywords: List[str]
    temporal_alignment: Dict[str, float]


class MatchResponse(BaseModel):
    video_segment: VideoSegmentResponse
    script_segment: ScriptSegmentResponse
    confidence: float
    explanation: ExplanationResponse


class AnalyzeResponse(BaseModel):
    matches: List[MatchResponse]
    processing_time: float
    job_id: str
    metadata: Dict[str, Any]


class JobStatus(BaseModel):
    job_id: str
    status: str
    progress: float
    result: Optional[AnalyzeResponse] = None
    error: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class BatchRequest(BaseModel):
    requests: List[AnalyzeRequest]


# API Endpoints
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Video Clip Contextualizer API",
        "version": "1.0.0",
        "description": "AI-powered semantic video-to-script matching system",
        "endpoints": {
            "analyze": "/api/v1/analyze",
            "batch": "/api/v1/batch",
            "job_status": "/api/v1/job/{job_id}",
            "models": "/api/v1/models"
        }
    }


@app.post("/api/v1/analyze", response_model=AnalyzeResponse)
async def analyze_video(
    request: AnalyzeRequest,
    background_tasks: BackgroundTasks,
    video_file: UploadFile = File(None)
):
    """
    Analyze video and match with script.
    
    Args:
        request: Analysis request parameters
        video_file: Optional video file upload
        
    Returns:
        Analysis results with matches
    """
    start_time = datetime.now()
    job_id = str(uuid.uuid4())
    
    try:
        # Handle video input
        video_path = None
        if video_file:
            # Save uploaded file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                content = await video_file.read()
                tmp_file.write(content)
                video_path = tmp_file.name
        elif request.video_url:
            # For production, implement video download from URL
            raise HTTPException(
                status_code=501, 
                detail="Video URL download not implemented in MVP"
            )
        else:
            raise HTTPException(
                status_code=400, 
                detail="Either video_file or video_url must be provided"
            )
        
        # Process video
        if request.clip_duration:
            video_processor.clip_duration = request.clip_duration
        if request.overlap:
            video_processor.overlap = request.overlap
            
        video_segments = video_processor.process_video_file(video_path)
        
        # Process text
        text_segments = text_processor.segment_text(request.script)
        text_segments = text_processor.generate_embeddings(text_segments)
        
        # Perform matching
        matches = semantic_matcher.match_video_to_text(
            video_segments, 
            text_segments, 
            request.matching_strategy
        )
        
        # Convert to response format
        match_responses = []
        for match in matches:
            match_response = MatchResponse(
                video_segment=VideoSegmentResponse(
                    start=match.video_start,
                    end=match.video_end
                ),
                script_segment=ScriptSegmentResponse(
                    start=match.text_start,
                    end=match.text_end
                ),
                confidence=match.confidence,
                explanation=ExplanationResponse(**match.explanation)
            )
            match_responses.append(match_response)
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Generate metadata
        metadata = {
            "video_segments_count": len(video_segments),
            "text_segments_count": len(text_segments),
            "total_matches": len(matches),
            "avg_confidence": sum(m.confidence for m in matches) / len(matches) if matches else 0,
            "processing_config": {
                "clip_duration": video_processor.clip_duration,
                "overlap": video_processor.overlap,
                "matching_strategy": request.matching_strategy
            }
        }
        
        result = AnalyzeResponse(
            matches=match_responses,
            processing_time=processing_time,
            job_id=job_id,
            metadata=metadata
        )
        
        # Clean up temporary file
        if video_path and os.path.exists(video_path):
            background_tasks.add_task(os.unlink, video_path)
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing video analysis: {str(e)}")
        
        # Clean up temporary file
        if video_path and os.path.exists(video_path):
            background_tasks.add_task(os.unlink, video_path)
        
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


@app.get("/api/v1/job/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    """Get status of a processing job."""
    if job_id not in active_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return JobStatus(**active_jobs[job_id])


@app.post("/api/v1/batch")
async def batch_analyze(request: BatchRequest):
    """
    Process multiple video analysis requests in batch.
    
    Args:
        request: Batch request with multiple analysis requests
        
    Returns:
        Batch job ID for tracking
    """
    job_id = str(uuid.uuid4())
    
    # Store job info
    active_jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "progress": 0.0,
        "result": None,
        "error": None,
        "created_at": datetime.now(),
        "updated_at": datetime.now()
    }
    
    # In production, add to queue for background processing
    # For now, return job ID
    return {"job_id": job_id, "status": "queued", "batch_size": len(request.requests)}


@app.get("/api/v1/models")
async def get_model_info():
    """Get information about loaded models."""
    return {
        "clip_encoder": semantic_matcher.clip_encoder.get_model_info(),
        "text_processor": {
            "model": config.models.text_encoder,
            "device": config.models.device
        },
        "video_processor": video_processor.get_processing_stats(),
        "system_config": {
            "device": config.processing.device,
            "precision": config.processing.precision,
            "batch_size": config.processing.batch_size
        }
    }


@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }


@app.get("/api/v1/metrics")
async def get_metrics():
    """Get system metrics."""
    import psutil
    
    return {
        "system": {
            "cpu_usage": psutil.cpu_percent(),
            "memory_usage": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage('/').percent
        },
        "api": {
            "active_jobs": len(active_jobs),
            "total_processed": 0  # Would track in production
        }
    }


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "timestamp": datetime.now().isoformat()}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unexpected error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "timestamp": datetime.now().isoformat()}
    )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=config.api.host,
        port=config.api.port,
        workers=config.api.workers,
        timeout_keep_alive=config.api.timeout
    )