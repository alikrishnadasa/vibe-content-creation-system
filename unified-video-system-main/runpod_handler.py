#!/usr/bin/env python3
"""
RunPod Serverless Handler for Unified Video System
Handles API requests for video generation on RunPod
"""

import runpod
import asyncio
import json
import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import tempfile
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import video generation components
try:
    from core.real_content_generator import RealContentGenerator, RealVideoRequest
    from core.quantum_pipeline import UnifiedQuantumPipeline
    logger.info("Successfully imported video generation components")
except ImportError as e:
    logger.error(f"Failed to import components: {e}")
    raise

# Global generator instance for reuse
GENERATOR = None

async def initialize_generator():
    """Initialize the video generator once"""
    global GENERATOR
    if GENERATOR is None:
        logger.info("Initializing Real Content Generator...")
        
        # Use environment variables for configuration
        clips_dir = os.environ.get('CLIPS_DIRECTORY', '/app/MJAnime')
        metadata_file = os.environ.get('METADATA_FILE', '/app/MJAnime/metadata_final_clean_shots.json')
        scripts_dir = os.environ.get('SCRIPTS_DIRECTORY', '/app/scripts')
        music_file = os.environ.get('MUSIC_FILE', '/app/music/Beanie (Slowed).mp3')
        output_dir = os.environ.get('OUTPUT_DIRECTORY', '/app/output')
        
        GENERATOR = RealContentGenerator(
            clips_directory=clips_dir,
            metadata_file=metadata_file,
            scripts_directory=scripts_dir,
            music_file=music_file,
            output_directory=output_dir
        )
        
        # Initialize generator
        init_success = await GENERATOR.initialize()
        if not init_success:
            logger.error("Failed to initialize generator")
            raise RuntimeError("Generator initialization failed")
        
        logger.info("Generator initialized successfully")
    
    return GENERATOR

async def generate_video_handler(job_input: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle video generation requests
    
    Expected input format:
    {
        "script_text": "Your script content here",
        "script_name": "script_name",
        "variation_number": 1,
        "caption_style": "tiktok",
        "music_sync": true,
        "min_clip_duration": 2.5,
        "target_duration": null  # Uses actual audio duration
    }
    """
    try:
        start_time = time.time()
        logger.info(f"Received job input: {job_input}")
        
        # Validate input
        if not job_input.get('script_text'):
            return {
                "error": "Missing required field: script_text",
                "success": False
            }
        
        # Get generator
        generator = await initialize_generator()
        
        # Extract parameters with defaults
        script_text = job_input['script_text']
        script_name = job_input.get('script_name', 'generated_script')
        variation_number = job_input.get('variation_number', 1)
        caption_style = job_input.get('caption_style', 'tiktok')
        music_sync = job_input.get('music_sync', True)
        min_clip_duration = job_input.get('min_clip_duration', 2.5)
        target_duration = job_input.get('target_duration', None)
        
        # Create temporary script file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(script_text)
            script_path = f.name
        
        try:
            # Create video request
            request = RealVideoRequest(
                script_path=script_path,
                script_name=script_name,
                variation_number=variation_number,
                caption_style=caption_style,
                music_sync=music_sync,
                target_duration=target_duration,
                min_clip_duration=min_clip_duration
            )
            
            # Generate video
            logger.info(f"Starting video generation for {script_name}")
            result = await generator.generate_video(request)
            
            # Process result
            if result.success:
                # Read output file
                output_path = Path(result.output_path)
                if output_path.exists():
                    # For RunPod, we need to return the file data or upload to storage
                    # For now, we'll return the path and metadata
                    file_size = output_path.stat().st_size
                    
                    # Convert to base64 for small files (< 50MB) or provide download URL
                    if file_size < 50 * 1024 * 1024:  # 50MB limit
                        import base64
                        with open(output_path, 'rb') as f:
                            video_data = base64.b64encode(f.read()).decode('utf-8')
                        
                        return {
                            "success": True,
                            "video_data": video_data,
                            "video_filename": output_path.name,
                            "file_size_bytes": file_size,
                            "generation_time": result.generation_time,
                            "processing_time": time.time() - start_time,
                            "metadata": {
                                "script_name": script_name,
                                "variation_number": variation_number,
                                "caption_style": caption_style,
                                "clips_used": result.clips_used,
                                "total_clips": result.total_clips,
                                "audio_duration": result.audio_duration
                            }
                        }
                    else:
                        # For large files, return metadata and indicate file needs to be downloaded
                        return {
                            "success": True,
                            "video_path": str(output_path),
                            "video_filename": output_path.name,
                            "file_size_bytes": file_size,
                            "generation_time": result.generation_time,
                            "processing_time": time.time() - start_time,
                            "note": "File too large for direct return. Use file download endpoint.",
                            "metadata": {
                                "script_name": script_name,
                                "variation_number": variation_number,
                                "caption_style": caption_style,
                                "clips_used": result.clips_used,
                                "total_clips": result.total_clips,
                                "audio_duration": result.audio_duration
                            }
                        }
                else:
                    return {
                        "success": False,
                        "error": "Video file not found after generation",
                        "processing_time": time.time() - start_time
                    }
            else:
                return {
                    "success": False,
                    "error": result.error_message,
                    "processing_time": time.time() - start_time
                }
        
        finally:
            # Clean up temporary script file
            try:
                os.unlink(script_path)
            except:
                pass
    
    except Exception as e:
        logger.error(f"Error in generate_video_handler: {e}")
        return {
            "success": False,
            "error": str(e),
            "processing_time": time.time() - start_time if 'start_time' in locals() else 0
        }

async def health_check_handler(job_input: Dict[str, Any]) -> Dict[str, Any]:
    """Health check endpoint"""
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if gpu_available else "No GPU"
        
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "gpu_available": gpu_available,
            "gpu_name": gpu_name,
            "generator_initialized": GENERATOR is not None
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "timestamp": time.time()
        }

def handler(job):
    """Main RunPod handler function"""
    logger.info(f"Handler called with job: {job}")
    
    try:
        # Get job input
        job_input = job.get('input', {})
        
        # Determine endpoint based on input
        endpoint = job_input.get('endpoint', 'generate')
        
        if endpoint == 'generate':
            # Run video generation
            result = asyncio.run(generate_video_handler(job_input))
        elif endpoint == 'health':
            # Run health check
            result = asyncio.run(health_check_handler(job_input))
        else:
            result = {
                "error": f"Unknown endpoint: {endpoint}",
                "available_endpoints": ["generate", "health"]
            }
        
        return result
    
    except Exception as e:
        logger.error(f"Handler error: {e}")
        return {
            "error": str(e),
            "success": False
        }

# Start the RunPod serverless handler
if __name__ == "__main__":
    logger.info("Starting RunPod serverless handler...")
    runpod.serverless.start({"handler": handler})