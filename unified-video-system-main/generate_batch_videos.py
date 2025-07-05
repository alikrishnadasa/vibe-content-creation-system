#!/usr/bin/env python3
"""
Batch Video Generator
Generate multiple videos using modular configuration
Preserves 2:3 aspect ratio and caption settings from generate_single_video.py
"""

import logging
import asyncio
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from core.real_content_generator import RealContentGenerator, RealVideoRequest
from video_config import DEFAULT_CONFIG, VideoConfig
import json

class BatchVideoGenerator:
    """
    Modular batch video generator that preserves all settings from single video generator
    """
    
    def __init__(self, config: VideoConfig = DEFAULT_CONFIG):
        """Initialize with video configuration"""
        self.config = config
        self.generator: Optional[RealContentGenerator] = None
        self.successful_videos = 0
        self.failed_videos = 0
        self.start_time = None
        
    async def initialize(self) -> bool:
        """Initialize the content generator"""
        logger.info("🔧 Initializing batch video generator...")
        
        # Validate configuration
        if not self.config.validate_paths():
            logger.error("❌ Configuration validation failed")
            return False
        
        # Initialize content generator with config
        self.generator = RealContentGenerator(**self.config.get_generator_kwargs())
        
        # Initialize all components
        init_success = await self.generator.initialize()
        if not init_success:
            logger.error("❌ Failed to initialize content generator")
            return False
        
        logger.info("✅ Batch video generator initialized successfully")
        logger.info(f"📐 Resolution: {self.config.target_resolution} (2:3 aspect ratio)")
        logger.info(f"📝 Caption style: {self.config.caption_style}")
        logger.info(f"🔥 Burn captions: {self.config.burn_in_captions}")
        
        # Check caption cache availability
        cached_scripts = self.config.get_available_cached_scripts()
        if cached_scripts:
            logger.info(f"🚀 Found {len(cached_scripts)} scripts with cached captions")
            logger.info(f"📁 Caption cache: {self.config.caption_cache_directory}")
        else:
            logger.warning(f"⚠️  No cached captions found in {self.config.caption_cache_directory}")
            logger.warning("   Caption generation will be slower without cache")
        
        return True
    
    def load_cached_captions(self, script_name: str) -> Optional[dict]:
        """Load cached captions for a script"""
        try:
            cache_path = self.config.get_caption_cache_path(script_name)
            if not cache_path.exists():
                return None
            
            with open(cache_path, 'r') as f:
                caption_data = json.load(f)
            
            logger.info(f"📋 Loaded {len(caption_data.get('captions', []))} cached captions for {script_name}")
            return caption_data
            
        except Exception as e:
            logger.warning(f"⚠️  Failed to load cached captions for {script_name}: {e}")
            return None
    
    async def generate_single_video(self, script_file: Path, variation_number: int, video_number: int, total_videos: int) -> bool:
        """Generate a single video"""
        script_name = script_file.stem
        video_start = time.time()
        
        logger.info(f"🎬 Generating video {video_number}/{total_videos}: {script_name} variation {variation_number}")
        
        # Check for cached captions
        cached_captions = self.load_cached_captions(script_name)
        if cached_captions:
            logger.info(f"   🚀 Using cached captions ({len(cached_captions.get('captions', []))} captions)")
        else:
            logger.info(f"   🔄 Generating captions on-the-fly (slower)")
        
        try:
            # Create video request using config
            request = RealVideoRequest(**self.config.get_video_request_kwargs(
                script_path=str(script_file),
                script_name=script_name,
                variation_number=variation_number
            ))
            
            # Generate video (RealContentGenerator automatically loads cached captions)
            result = await self.generator.generate_video(request)
            video_time = time.time() - video_start
            
            if result.success:
                self.successful_videos += 1
                file_size_mb = Path(result.output_path).stat().st_size / (1024 * 1024)
                
                logger.info(f"✅ Video {video_number}/{total_videos} SUCCESS: {Path(result.output_path).name}")
                logger.info(f"   ⚡ Time: {video_time:.1f}s | Size: {file_size_mb:.1f}MB | Duration: {result.total_duration:.1f}s")
                return True
            else:
                self.failed_videos += 1
                logger.error(f"❌ Video {video_number}/{total_videos} FAILED: {result.error_message}")
                return False
                
        except Exception as e:
            self.failed_videos += 1
            logger.error(f"💥 Video {video_number}/{total_videos} EXCEPTION: {e}")
            return False
    
    def log_progress(self, current_video: int, total_videos: int):
        """Log progress update"""
        if current_video % 10 == 0:
            elapsed_time = time.time() - self.start_time
            avg_time_per_video = elapsed_time / current_video
            estimated_remaining = (total_videos - current_video) * avg_time_per_video
            
            logger.info("=" * 60)
            logger.info(f"📊 PROGRESS: {current_video}/{total_videos} videos completed")
            logger.info(f"✅ Successful: {self.successful_videos} | ❌ Failed: {self.failed_videos}")
            logger.info(f"📈 Success rate: {(self.successful_videos/current_video)*100:.1f}%")
            logger.info(f"⏱️  Average: {avg_time_per_video:.1f}s per video")
            logger.info(f"🕒 Estimated remaining: {estimated_remaining/60:.1f} minutes")
            logger.info("=" * 60)
    
    def should_clear_cache(self, video_number: int) -> bool:
        """Determine if uniqueness cache should be cleared"""
        return video_number % self.config.clear_uniqueness_cache_frequency == 1
    
    def log_final_summary(self, total_videos: int):
        """Log final generation summary"""
        total_time = time.time() - self.start_time
        
        logger.info("=" * 80)
        logger.info("🎉 BATCH VIDEO GENERATION COMPLETE!")
        logger.info("=" * 80)
        logger.info(f"✅ Successful videos: {self.successful_videos}")
        logger.info(f"❌ Failed videos: {self.failed_videos}")
        logger.info(f"📊 Success rate: {(self.successful_videos/total_videos)*100:.1f}%")
        logger.info(f"⏱️  Total time: {total_time/60:.1f} minutes")
        logger.info(f"⚡ Average time per video: {total_time/total_videos:.1f}s")
        logger.info(f"📁 Videos saved to: {self.config.output_directory}/")
        logger.info(f"📐 All videos: {self.config.target_resolution} (2:3 aspect ratio)")
        logger.info(f"📝 Caption style: {self.config.caption_style} (burned in)")
        logger.info("=" * 80)
    
    async def generate_batch(self, num_videos: int = 100) -> bool:
        """Generate a batch of videos"""
        logger.info(f"🎬 GENERATING {num_videos} VIDEOS")
        logger.info("📝 Using Whisper word-by-word captions with 2:3 aspect ratio")
        
        self.start_time = time.time()
        
        # Get available scripts
        script_files = self.config.get_available_scripts()
        if not script_files:
            logger.error("❌ No script files found")
            return False
        
        logger.info(f"📁 Found {len(script_files)} script files")
        
        # Generate videos
        for i in range(1, num_videos + 1):
            # Cycle through scripts and create variations
            script_file = script_files[(i - 1) % len(script_files)]
            variation_number = ((i - 1) // len(script_files)) + 1
            
            # Clear uniqueness cache for variety
            if self.should_clear_cache(i):
                self.generator.uniqueness_engine.clear_cache()
                logger.info("🔄 Cleared uniqueness cache for more variety")
            
            # Generate video
            await self.generate_single_video(script_file, variation_number, i, num_videos)
            
            # Log progress
            self.log_progress(i, num_videos)
        
        # Final summary
        self.log_final_summary(num_videos)
        
        # Consider success if 90%+ videos generated
        success_rate = self.successful_videos / num_videos
        return success_rate >= 0.9

async def generate_videos(num_videos: int = 100, config: VideoConfig = DEFAULT_CONFIG) -> bool:
    """
    Main function to generate batch videos
    
    Args:
        num_videos: Number of videos to generate
        config: Video configuration (uses DEFAULT_CONFIG if not specified)
    
    Returns:
        bool: True if batch generation was successful
    """
    
    generator = BatchVideoGenerator(config)
    
    # Initialize
    if not await generator.initialize():
        return False
    
    # Generate batch
    return await generator.generate_batch(num_videos)

async def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate batch videos with modular configuration")
    parser.add_argument("--count", "-c", type=int, default=100, help="Number of videos to generate")
    parser.add_argument("--output", "-o", type=str, help="Output directory override")
    parser.add_argument("--style", "-s", type=str, help="Caption style override")
    parser.add_argument("--caption-cache", type=str, help="Caption cache directory override")
    
    args = parser.parse_args()
    
    # Create config with overrides
    config_kwargs = {}
    if args.output:
        config_kwargs["output_directory"] = args.output
    if args.style:
        config_kwargs["caption_style"] = args.style
    if args.caption_cache:
        config_kwargs["caption_cache_directory"] = args.caption_cache
    
    from video_config import create_custom_config
    config = create_custom_config(**config_kwargs) if config_kwargs else DEFAULT_CONFIG
    
    # Generate videos
    success = await generate_videos(args.count, config)
    
    if success:
        print(f"\n🎉 SUCCESS! {args.count} video batch generation completed!")
        return True
    else:
        print(f"\n💥 BATCH GENERATION ENCOUNTERED ISSUES - Check logs for details")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    if success:
        sys.exit(0)
    else:
        sys.exit(1)