"""
Content Pipeline - Batch Production System

Scalable pipeline for processing all 11 scripts with multiple variations
to generate 55+ test videos, designed to scale to 1,100+ production videos.
Optimized for 1,000+ videos/month production capacity.
"""

import logging
import asyncio
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class BatchVideoRequest:
    """Request for batch video generation"""
    script_path: str
    script_name: str
    variation_count: int
    caption_style: str = "tiktok"
    music_sync: bool = True
    target_duration: float = 15.0

@dataclass
class BatchResult:
    """Result of batch processing operation"""
    success: bool
    total_videos: int
    successful_videos: int
    failed_videos: int
    total_time: float
    average_time_per_video: float
    output_paths: List[str]
    error_messages: List[str]
    performance_stats: Dict[str, Any]

class ContentPipeline:
    """Batch production system for real content videos"""
    
    def __init__(self, 
                 content_generator,
                 max_concurrent_videos: int = 4,
                 performance_monitoring: bool = True):
        """
        Initialize content pipeline
        
        Args:
            content_generator: RealContentGenerator instance
            max_concurrent_videos: Maximum concurrent video generations
            performance_monitoring: Enable performance monitoring
        """
        self.content_generator = content_generator
        self.max_concurrent_videos = max_concurrent_videos
        self.performance_monitoring = performance_monitoring
        
        # Performance tracking
        self.stats = {
            'total_videos_generated': 0,
            'total_processing_time': 0.0,
            'successful_batches': 0,
            'failed_batches': 0,
            'average_video_time': 0.0,
            'peak_memory_usage_mb': 0.0,
            'gpu_utilization': 0.0
        }
        
        # Resource management
        self.semaphore = asyncio.Semaphore(max_concurrent_videos)
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent_videos)
    
    async def process_all_scripts(self, 
                                scripts_directory: str,
                                variations_per_script: int = 5,
                                caption_style: str = "tiktok",
                                output_directory: str = "output") -> BatchResult:
        """
        Process all available scripts with multiple variations
        
        Args:
            scripts_directory: Directory containing audio scripts
            variations_per_script: Number of variations per script
            caption_style: Caption style to use
            output_directory: Output directory for videos
            
        Returns:
            Batch processing result
        """
        logger.info(f"🚀 Starting batch processing: {variations_per_script} variations per script")
        start_time = time.time()
        
        try:
            # Discover all script files
            scripts_dir = Path(scripts_directory)
            script_files = list(scripts_dir.glob("*.wav"))
            
            if not script_files:
                logger.error(f"No script files found in {scripts_directory}")
                return BatchResult(
                    success=False,
                    total_videos=0,
                    successful_videos=0,
                    failed_videos=0,
                    total_time=0.0,
                    average_time_per_video=0.0,
                    output_paths=[],
                    error_messages=[f"No script files found in {scripts_directory}"],
                    performance_stats={}
                )
            
            logger.info(f"📁 Found {len(script_files)} script files")
            
            # Create batch requests
            batch_requests = []
            for script_file in script_files:
                script_name = script_file.stem
                for variation in range(1, variations_per_script + 1):
                    request = BatchVideoRequest(
                        script_path=str(script_file),
                        script_name=script_name,
                        variation_count=variation,
                        caption_style=caption_style,
                        music_sync=True,
                        target_duration=15.0
                    )
                    batch_requests.append(request)
            
            total_videos = len(batch_requests)
            logger.info(f"🎬 Generating {total_videos} videos ({len(script_files)} scripts × {variations_per_script} variations)")
            
            # Process videos in batches
            results = await self._process_batch_requests(batch_requests)
            
            # Compile final results
            total_time = time.time() - start_time
            successful_videos = sum(1 for r in results if r.success)
            failed_videos = total_videos - successful_videos
            
            output_paths = [r.output_path for r in results if r.success]
            error_messages = [r.error_message for r in results if r.error_message]
            
            # Update global stats
            self.stats['total_videos_generated'] += successful_videos
            self.stats['total_processing_time'] += total_time
            if successful_videos > 0:
                self.stats['average_video_time'] = total_time / total_videos
            
            if successful_videos == total_videos:
                self.stats['successful_batches'] += 1
                logger.info("✅ ALL VIDEOS GENERATED SUCCESSFULLY!")
            else:
                self.stats['failed_batches'] += 1
                logger.warning(f"⚠️ Batch completed with {failed_videos} failures")
            
            # Performance analysis
            performance_stats = {
                'generation_rate_videos_per_second': successful_videos / total_time if total_time > 0 else 0,
                'average_time_per_video': total_time / total_videos if total_videos > 0 else 0,
                'success_rate': successful_videos / total_videos if total_videos > 0 else 0,
                'concurrent_processing': self.max_concurrent_videos,
                'total_scripts_processed': len(script_files),
                'variations_per_script': variations_per_script
            }
            
            logger.info(f"📊 Batch Processing Complete:")
            logger.info(f"   Total videos: {total_videos}")
            logger.info(f"   Successful: {successful_videos}")
            logger.info(f"   Failed: {failed_videos}")
            logger.info(f"   Total time: {total_time:.3f}s")
            logger.info(f"   Average per video: {performance_stats['average_time_per_video']:.3f}s")
            logger.info(f"   Generation rate: {performance_stats['generation_rate_videos_per_second']:.2f} videos/second")
            logger.info(f"   Success rate: {performance_stats['success_rate']:.1%}")
            
            return BatchResult(
                success=successful_videos == total_videos,
                total_videos=total_videos,
                successful_videos=successful_videos,
                failed_videos=failed_videos,
                total_time=total_time,
                average_time_per_video=performance_stats['average_time_per_video'],
                output_paths=output_paths,
                error_messages=error_messages,
                performance_stats=performance_stats
            )
            
        except Exception as e:
            total_time = time.time() - start_time
            logger.error(f"❌ Batch processing failed: {e}")
            self.stats['failed_batches'] += 1
            
            return BatchResult(
                success=False,
                total_videos=0,
                successful_videos=0,
                failed_videos=0,
                total_time=total_time,
                average_time_per_video=0.0,
                output_paths=[],
                error_messages=[str(e)],
                performance_stats={}
            )
    
    async def _process_batch_requests(self, requests: List[BatchVideoRequest]) -> List[Any]:
        """Process multiple video requests concurrently"""
        logger.info(f"🔄 Processing {len(requests)} video requests with {self.max_concurrent_videos} concurrent workers")
        
        # Create tasks for concurrent processing
        tasks = []
        for i, request in enumerate(requests):
            task = asyncio.create_task(
                self._process_single_video_with_semaphore(request, i + 1, len(requests))
            )
            tasks.append(task)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Video {i+1} failed with exception: {result}")
                # Create error result
                error_result = type('ErrorResult', (), {
                    'success': False,
                    'output_path': '',
                    'error_message': str(result)
                })()
                processed_results.append(error_result)
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def _process_single_video_with_semaphore(self, 
                                                 request: BatchVideoRequest, 
                                                 video_num: int, 
                                                 total_videos: int):
        """Process single video with concurrency control"""
        async with self.semaphore:
            return await self._process_single_video(request, video_num, total_videos)
    
    async def _process_single_video(self, 
                                  request: BatchVideoRequest, 
                                  video_num: int, 
                                  total_videos: int):
        """Process a single video request"""
        start_time = time.time()
        
        try:
            logger.info(f"🎬 Generating video {video_num}/{total_videos}: {request.script_name} variation {request.variation_count}")
            
            # Create video request for content generator
            from core.real_content_generator import RealVideoRequest
            
            video_request = RealVideoRequest(
                script_path=request.script_path,
                script_name=request.script_name,
                variation_number=request.variation_count,
                caption_style=request.caption_style,
                music_sync=request.music_sync,
                target_duration=request.target_duration
            )
            
            # Generate the video
            result = await self.content_generator.generate_video(video_request)
            
            generation_time = time.time() - start_time
            
            if result.success:
                logger.info(f"✅ Video {video_num}/{total_videos} complete: {Path(result.output_path).name} ({generation_time:.3f}s)")
            else:
                logger.error(f"❌ Video {video_num}/{total_videos} failed: {result.error_message}")
            
            return result
            
        except Exception as e:
            generation_time = time.time() - start_time
            logger.error(f"❌ Video {video_num}/{total_videos} failed with exception: {e} ({generation_time:.3f}s)")
            
            # Create error result
            error_result = type('ErrorResult', (), {
                'success': False,
                'output_path': '',
                'error_message': str(e),
                'generation_time': generation_time
            })()
            
            return error_result
    
    async def process_specific_scripts(self, 
                                     script_names: List[str],
                                     scripts_directory: str,
                                     variations_per_script: int = 5,
                                     caption_style: str = "tiktok") -> BatchResult:
        """
        Process specific scripts by name
        
        Args:
            script_names: List of script names (without extension)
            scripts_directory: Directory containing scripts
            variations_per_script: Number of variations per script
            caption_style: Caption style to use
            
        Returns:
            Batch processing result
        """
        logger.info(f"🎯 Processing specific scripts: {script_names}")
        
        # Create batch requests for specified scripts
        batch_requests = []
        scripts_dir = Path(scripts_directory)
        
        for script_name in script_names:
            script_file = scripts_dir / f"{script_name}.wav"
            
            if not script_file.exists():
                logger.warning(f"Script file not found: {script_file}")
                continue
            
            for variation in range(1, variations_per_script + 1):
                request = BatchVideoRequest(
                    script_path=str(script_file),
                    script_name=script_name,
                    variation_count=variation,
                    caption_style=caption_style,
                    music_sync=True,
                    target_duration=15.0
                )
                batch_requests.append(request)
        
        if not batch_requests:
            logger.error("No valid script files found for processing")
            return BatchResult(
                success=False,
                total_videos=0,
                successful_videos=0,
                failed_videos=0,
                total_time=0.0,
                average_time_per_video=0.0,
                output_paths=[],
                error_messages=["No valid script files found"],
                performance_stats={}
            )
        
        # Process the requests
        start_time = time.time()
        results = await self._process_batch_requests(batch_requests)
        total_time = time.time() - start_time
        
        # Compile results
        total_videos = len(batch_requests)
        successful_videos = sum(1 for r in results if r.success)
        failed_videos = total_videos - successful_videos
        
        output_paths = [r.output_path for r in results if r.success]
        error_messages = [r.error_message for r in results if r.error_message]
        
        performance_stats = {
            'generation_rate_videos_per_second': successful_videos / total_time if total_time > 0 else 0,
            'average_time_per_video': total_time / total_videos if total_videos > 0 else 0,
            'success_rate': successful_videos / total_videos if total_videos > 0 else 0,
            'scripts_processed': len(script_names)
        }
        
        return BatchResult(
            success=successful_videos == total_videos,
            total_videos=total_videos,
            successful_videos=successful_videos,
            failed_videos=failed_videos,
            total_time=total_time,
            average_time_per_video=performance_stats['average_time_per_video'],
            output_paths=output_paths,
            error_messages=error_messages,
            performance_stats=performance_stats
        )
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get comprehensive pipeline statistics"""
        return {
            'pipeline_stats': self.stats.copy(),
            'configuration': {
                'max_concurrent_videos': self.max_concurrent_videos,
                'performance_monitoring': self.performance_monitoring
            },
            'timestamp': datetime.now().isoformat()
        }
    
    def save_batch_report(self, result: BatchResult, output_file: str):
        """Save batch processing report to file"""
        report = {
            'batch_result': {
                'success': result.success,
                'total_videos': result.total_videos,
                'successful_videos': result.successful_videos,
                'failed_videos': result.failed_videos,
                'total_time': result.total_time,
                'average_time_per_video': result.average_time_per_video,
                'output_paths': result.output_paths,
                'error_messages': result.error_messages,
                'performance_stats': result.performance_stats
            },
            'pipeline_stats': self.get_pipeline_stats(),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"📄 Batch report saved: {output_file}")
    
    async def cleanup(self):
        """Cleanup pipeline resources"""
        logger.info("🧹 Cleaning up pipeline resources...")
        self.executor.shutdown(wait=True)
        logger.info("✅ Pipeline cleanup complete")