#!/usr/bin/env python3
"""
Generate Unified Videos
Easy script to generate videos using your integrated quantum pipeline
"""

import asyncio
import logging
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project paths
sys.path.append('unified-video-system-main')

from core.real_content_generator import RealContentGenerator, RealVideoRequest

# Available scripts (you can modify this list)
AVAILABLE_SCRIPTS = [
    "anxiety1", "anxiety1_mixed",
    "safe1", "safe1_mixed", 
    "deadinside", "deadinside_mixed",
    "phone1.", "phone1._mixed",
    "before", "before_mixed",
    "adhd", "adhd_mixed",
    "miserable1", "miserable1_mixed"
]

async def generate_videos(script_names=None, num_variations=1):
    """Generate videos using unified metadata"""
    
    if script_names is None:
        script_names = AVAILABLE_SCRIPTS[:3]  # Generate 3 by default
    
    logger.info(f"🚀 Generating {len(script_names)} videos with {num_variations} variation(s) each")
    
    # Initialize generator with unified metadata
    generator = RealContentGenerator(
        clips_directory="",
        metadata_file="unified_clips_metadata.json", 
        scripts_directory="11-scripts-for-tiktok",
        music_file="unified-video-system-main/music/Beanie (Slowed).mp3"
    )
    
    # Enable unified metadata
    generator.content_database.clips_loader.use_unified_metadata = True
    
    # Initialize
    success = await generator.initialize()
    if not success:
        logger.error("❌ Failed to initialize generator")
        return False
    
    stats = generator.content_database.clips_loader.get_clip_stats()
    source_stats = generator.content_database.clips_loader.get_source_stats()
    logger.info(f"📊 Using {stats['total_clips']} clips: {source_stats}")
    
    results = []
    
    for script_name in script_names:
        for variation in range(1, num_variations + 1):
            logger.info(f"\n🎬 Generating {script_name} variation {variation}...")
            
            try:
                request = RealVideoRequest(
                    script_path=f"11-scripts-for-tiktok/{script_name}.wav",
                    script_name=script_name,
                    variation_number=variation,
                    caption_style="default",
                    music_sync=True,
                    burn_in_captions=True
                )
                
                result = await generator.generate_video(request)
                
                if result.success:
                    logger.info(f"✅ Success: {result.output_path}")
                    logger.info(f"   Duration: {result.total_duration:.1f}s, Clips: {len(result.clips_used)}")
                    results.append({"script": script_name, "variation": variation, "success": True, "path": result.output_path})
                else:
                    logger.error(f"❌ Failed: {result.error_message}")
                    results.append({"script": script_name, "variation": variation, "success": False, "error": result.error_message})
                    
            except Exception as e:
                logger.error(f"❌ Error: {e}")
                results.append({"script": script_name, "variation": variation, "success": False, "error": str(e)})
    
    # Summary
    successful = sum(1 for r in results if r['success'])
    logger.info(f"\n📊 Generation Summary:")
    logger.info(f"✅ Successful: {successful}/{len(results)}")
    
    for result in results:
        status = "✅" if result['success'] else "❌"
        script_var = f"{result['script']} v{result['variation']}"
        if result['success']:
            logger.info(f"{status} {script_var}: {Path(result['path']).name}")
        else:
            logger.info(f"{status} {script_var}: {result.get('error', 'Unknown error')}")
    
    return successful > 0

async def main():
    """Main function"""
    
    print("🎬 Unified Video Generator")
    print("Using clips from MJAnime + Midjourney Composite")
    print("=" * 50)
    
    # You can customize this:
    scripts_to_generate = ["anxiety1", "safe1", "phone1."]  # Generate 3 videos
    variations_per_script = 1  # 1 variation each
    
    success = await generate_videos(scripts_to_generate, variations_per_script)
    
    if success:
        print(f"\n🎉 Video generation completed!")
        print(f"📁 Check the 'output' directory for your videos")
        print(f"💡 Each video uses clips from both MJAnime and Midjourney sources")
    else:
        print(f"\n❌ Video generation failed")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())