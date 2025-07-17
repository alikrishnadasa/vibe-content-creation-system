import asyncio
import logging
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
import uuid
import json
from enum import Enum

from ..processing.video_processor import VideoProcessor
from ..processing.text_processor import TextProcessor
from ..matching.semantic_matcher import SemanticMatcher
from ..config import get_config


class JobStatus(Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class BatchJob:
    job_id: str
    status: JobStatus = JobStatus.QUEUED
    progress: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    video_path: Optional[str] = None
    script: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    processing_time: Optional[float] = None


@dataclass
class BatchRequest:
    videos: List[str]
    scripts: List[str]
    configs: List[Dict[str, Any]]
    batch_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    priority: int = 1
    callback_url: Optional[str] = None


class BatchProcessor:
    """Handles batch processing of video-to-text matching requests."""
    
    def __init__(self, max_concurrent_jobs: int = None):
        self.config = get_config()
        self.logger = logging.getLogger(__name__)
        
        # Initialize processors
        self.video_processor = VideoProcessor()
        self.text_processor = TextProcessor()
        self.semantic_matcher = SemanticMatcher()
        
        # Job management
        self.max_concurrent_jobs = max_concurrent_jobs or self.config.processing.max_parallel
        self.active_jobs: Dict[str, BatchJob] = {}
        self.job_queue: asyncio.Queue = asyncio.Queue()
        self.processing_semaphore = asyncio.Semaphore(self.max_concurrent_jobs)
        
        # Statistics
        self.stats = {
            "total_jobs": 0,
            "completed_jobs": 0,
            "failed_jobs": 0,
            "avg_processing_time": 0.0,
            "started_at": datetime.now()
        }
        
        # Start background processor
        self._processor_task = None
        self._start_processor()
    
    def _start_processor(self):
        """Start the background job processor."""
        if self._processor_task is None or self._processor_task.done():
            self._processor_task = asyncio.create_task(self._process_queue())
    
    async def _process_queue(self):
        """Background task to process queued jobs."""
        while True:
            try:
                job = await self.job_queue.get()
                
                if job.status == JobStatus.CANCELLED:
                    self.job_queue.task_done()
                    continue
                
                # Process job with semaphore for concurrency control
                async with self.processing_semaphore:
                    await self._process_single_job(job)
                
                self.job_queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in batch processor: {str(e)}")
                await asyncio.sleep(1)
    
    async def submit_job(self, video_path: str, script: str, config: Dict[str, Any] = None) -> str:
        """
        Submit a single job for processing.
        
        Args:
            video_path: Path to video file
            script: Text script to match
            config: Processing configuration
            
        Returns:
            Job ID
        """
        job_id = str(uuid.uuid4())
        
        job = BatchJob(
            job_id=job_id,
            video_path=video_path,
            script=script,
            config=config or {}
        )
        
        self.active_jobs[job_id] = job
        await self.job_queue.put(job)
        
        self.stats["total_jobs"] += 1
        self.logger.info(f"Submitted job {job_id} for processing")
        
        return job_id
    
    async def submit_batch(self, batch_request: BatchRequest) -> List[str]:
        """
        Submit multiple jobs as a batch.
        
        Args:
            batch_request: Batch request with multiple videos and scripts
            
        Returns:
            List of job IDs
        """
        job_ids = []
        
        for video_path, script, config in zip(
            batch_request.videos, 
            batch_request.scripts, 
            batch_request.configs
        ):
            job_id = await self.submit_job(video_path, script, config)
            job_ids.append(job_id)
        
        self.logger.info(f"Submitted batch {batch_request.batch_id} with {len(job_ids)} jobs")
        return job_ids
    
    async def _process_single_job(self, job: BatchJob):
        """Process a single job."""
        start_time = datetime.now()
        
        try:
            # Update job status
            job.status = JobStatus.PROCESSING
            job.updated_at = datetime.now()
            job.progress = 0.1
            
            self.logger.info(f"Processing job {job.job_id}")
            
            # Configure processors
            config = job.config or {}
            if "clip_duration" in config:
                self.video_processor.clip_duration = config["clip_duration"]
            if "overlap" in config:
                self.video_processor.overlap = config["overlap"]
            
            # Process video
            job.progress = 0.2
            video_segments = self.video_processor.process_video_file(job.video_path)
            
            # Process text
            job.progress = 0.4
            text_segments = self.text_processor.segment_text(job.script)
            text_segments = self.text_processor.generate_embeddings(text_segments)
            
            # Perform matching
            job.progress = 0.7
            matching_strategy = config.get("matching_strategy", "optimal")
            matches = self.semantic_matcher.match_video_to_text(
                video_segments, 
                text_segments, 
                matching_strategy
            )
            
            # Generate results
            job.progress = 0.9
            result = self._format_results(matches, video_segments, text_segments)
            
            # Complete job
            job.status = JobStatus.COMPLETED
            job.progress = 1.0
            job.result = result
            job.processing_time = (datetime.now() - start_time).total_seconds()
            job.updated_at = datetime.now()
            
            self.stats["completed_jobs"] += 1
            self._update_avg_processing_time(job.processing_time)
            
            self.logger.info(f"Completed job {job.job_id} in {job.processing_time:.2f}s")
            
        except Exception as e:
            # Handle job failure
            job.status = JobStatus.FAILED
            job.error = str(e)
            job.updated_at = datetime.now()
            
            self.stats["failed_jobs"] += 1
            
            self.logger.error(f"Failed to process job {job.job_id}: {str(e)}")
    
    def _format_results(self, matches, video_segments, text_segments) -> Dict[str, Any]:
        """Format processing results."""
        match_results = []
        
        for match in matches:
            match_result = {
                "video_segment": {
                    "start": match.video_start,
                    "end": match.video_end
                },
                "text_segment": {
                    "start": match.text_start,
                    "end": match.text_end
                },
                "confidence": match.confidence,
                "explanation": match.explanation
            }
            match_results.append(match_result)
        
        return {
            "matches": match_results,
            "metadata": {
                "video_segments_count": len(video_segments),
                "text_segments_count": len(text_segments),
                "total_matches": len(matches),
                "avg_confidence": sum(m.confidence for m in matches) / len(matches) if matches else 0
            }
        }
    
    def _update_avg_processing_time(self, processing_time: float):
        """Update average processing time statistics."""
        total_completed = self.stats["completed_jobs"]
        if total_completed == 1:
            self.stats["avg_processing_time"] = processing_time
        else:
            current_avg = self.stats["avg_processing_time"]
            self.stats["avg_processing_time"] = (
                (current_avg * (total_completed - 1) + processing_time) / total_completed
            )
    
    def get_job_status(self, job_id: str) -> Optional[BatchJob]:
        """Get status of a specific job."""
        return self.active_jobs.get(job_id)
    
    def get_batch_status(self, job_ids: List[str]) -> Dict[str, Any]:
        """Get status of multiple jobs."""
        jobs = [self.active_jobs.get(job_id) for job_id in job_ids]
        jobs = [job for job in jobs if job is not None]
        
        if not jobs:
            return {"error": "No jobs found"}
        
        status_counts = {}
        for job in jobs:
            status = job.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
        
        total_progress = sum(job.progress for job in jobs) / len(jobs)
        
        return {
            "total_jobs": len(jobs),
            "status_counts": status_counts,
            "overall_progress": total_progress,
            "jobs": [
                {
                    "job_id": job.job_id,
                    "status": job.status.value,
                    "progress": job.progress,
                    "created_at": job.created_at.isoformat(),
                    "updated_at": job.updated_at.isoformat(),
                    "processing_time": job.processing_time,
                    "error": job.error
                }
                for job in jobs
            ]
        }
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a queued or processing job."""
        if job_id not in self.active_jobs:
            return False
        
        job = self.active_jobs[job_id]
        
        if job.status in [JobStatus.QUEUED, JobStatus.PROCESSING]:
            job.status = JobStatus.CANCELLED
            job.updated_at = datetime.now()
            return True
        
        return False
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        return {
            "stats": self.stats,
            "active_jobs": len([j for j in self.active_jobs.values() if j.status == JobStatus.PROCESSING]),
            "queued_jobs": self.job_queue.qsize(),
            "max_concurrent": self.max_concurrent_jobs,
            "uptime": (datetime.now() - self.stats["started_at"]).total_seconds()
        }
    
    def cleanup_completed_jobs(self, max_age_hours: int = 24):
        """Clean up old completed jobs."""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        to_remove = []
        for job_id, job in self.active_jobs.items():
            if job.status in [JobStatus.COMPLETED, JobStatus.FAILED] and job.updated_at < cutoff_time:
                to_remove.append(job_id)
        
        for job_id in to_remove:
            del self.active_jobs[job_id]
        
        self.logger.info(f"Cleaned up {len(to_remove)} old jobs")
    
    async def shutdown(self):
        """Gracefully shutdown the batch processor."""
        self.logger.info("Shutting down batch processor...")
        
        # Cancel background task
        if self._processor_task:
            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass
        
        # Cancel all queued jobs
        for job in self.active_jobs.values():
            if job.status == JobStatus.QUEUED:
                job.status = JobStatus.CANCELLED
                job.updated_at = datetime.now()
        
        self.logger.info("Batch processor shutdown complete")