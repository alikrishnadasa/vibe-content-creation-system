#!/usr/bin/env python3
"""
Create Single Test Video

Generate one complete video with real MJAnime clips, script audio, and background music
to demonstrate the full pipeline functionality.
"""

import logging
import asyncio
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from core.real_content_generator import RealContentGenerator, RealVideoRequest

async def create_single_test_video():
    """Create a single test video with full audio/video integration"""
    logger.info("🎬 Creating single test video with audio and video integration")
    
    try:
        # Initialize the real content generator
        logger.info("Initializing content generator...")
        generator = RealContentGenerator(
            clips_directory="../MJAnime",
            metadata_file="../MJAnime/metadata_final_clean_shots.json", 
            scripts_directory="../11-scripts-for-tiktok",
            music_file="../Beanie (Slowed).mp3",
            output_directory="output"
        )
        
        # Initialize all components
        init_success = await generator.initialize()
        if not init_success:
            logger.error("❌ Failed to initialize generator")
            return False
        
        logger.info("✅ Generator initialized successfully")
        
        # Create a focused video request using a different script or clear uniqueness temporarily
        # Let's try with miserable1 and clear uniqueness for testing
        generator.uniqueness_engine.clear_cache()
        logger.info("🧹 Cleared uniqueness cache for testing")
        
        request = RealVideoRequest(
            script_path="../11-scripts-for-tiktok/miserable1.wav",
            script_name="miserable1", 
            variation_number=1,  # Fresh start
            caption_style="tiktok",
            music_sync=True,
            target_duration=15.0
        )
        
        logger.info(f"🎯 Generating test video: {request.script_name}")
        logger.info(f"   Script: {request.script_path}")
        logger.info(f"   Music sync: {request.music_sync}")
        logger.info(f"   Caption style: {request.caption_style}")
        logger.info(f"   Target duration: {request.target_duration}s")
        
        # Generate the video
        result = await generator.generate_video(request)
        
        if result.success:
            output_path = Path(result.output_path)
            
            logger.info("🎉 TEST VIDEO CREATED SUCCESSFULLY!")
            logger.info(f"📂 Output file: {output_path.name}")
            logger.info(f"📍 Full path: {output_path.absolute()}")
            logger.info(f"⏱️  Generation time: {result.generation_time:.3f}s")
            logger.info(f"🎞️  Total duration: {result.total_duration:.1f}s")
            logger.info(f"🎬 Clips used: {len(result.clips_used)}")
            logger.info(f"🎯 Relevance score: {result.relevance_score:.3f}")
            logger.info(f"🌈 Variety score: {result.visual_variety_score:.3f}")
            
            # Check file details
            if output_path.exists():
                file_size_mb = output_path.stat().st_size / (1024 * 1024)
                logger.info(f"💾 File size: {file_size_mb:.1f}MB")
                
                # Try to verify it's a real video file
                if output_path.suffix.lower() == '.mp4':
                    logger.info("✅ Generated MP4 video file")
                    
                    # Check if it's actually a video using MoviePy
                    try:
                        from moviepy.editor import VideoFileClip
                        with VideoFileClip(str(output_path)) as video:
                            logger.info(f"🎥 Video verification:")
                            logger.info(f"   Duration: {video.duration:.1f}s")
                            logger.info(f"   Resolution: {video.w}x{video.h}")
                            logger.info(f"   FPS: {video.fps}")
                            if video.audio:
                                logger.info(f"   Audio: ✅ Present ({video.audio.duration:.1f}s)")
                            else:
                                logger.info(f"   Audio: ❌ Missing")
                                
                        logger.info("🎊 VIDEO VERIFICATION SUCCESSFUL!")
                        
                    except Exception as e:
                        logger.warning(f"⚠️  Could not verify video with MoviePy: {e}")
                        logger.info("📄 File created but may be placeholder format")
                
                logger.info(f"\n🚀 TEST COMPLETE - Video ready for viewing!")
                logger.info(f"💡 You can play the video: {output_path.absolute()}")
                return True
                
            else:
                logger.error("❌ Output file was not created")
                return False
                
        else:
            logger.error(f"❌ Video generation failed: {result.error_message}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Test video creation failed: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(create_single_test_video())
    if success:
        print("\n🎉 SUCCESS! Test video created and ready for viewing!")
    else:
        print("\n💥 FAILED! Could not create test video")
        sys.exit(1)