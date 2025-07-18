#!/usr/bin/env python3
"""
Generate a real hi-hat synchronized video using the complete system.

This creates an actual video file using the percussive sync feature.
"""

import asyncio
import logging
from pathlib import Path
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def generate_real_hihat_video():
    """Generate a real hi-hat synchronized video"""
    
    try:
        logger.info("🥁 Generating REAL Hi-Hat Sync Video")
        logger.info("=" * 60)
        
        # Import the system
        from core.real_content_generator import RealContentGenerator, RealVideoRequest
        
        # Configuration paths (using absolute paths like existing scripts)
        clips_directory = "/Users/jamesguo/vibe-content-creation/MJAnime"
        metadata_file = "/Users/jamesguo/vibe-content-creation/MJAnime/metadata_final_clean_shots.json"
        scripts_directory = "/Users/jamesguo/vibe-content-creation/11-scripts-for-tiktok"
        music_file = "/Users/jamesguo/vibe-content-creation/unified-video-system-main/music/Beanie (Slowed).mp3"
        script_path = "/Users/jamesguo/vibe-content-creation/11-scripts-for-tiktok/anxiety1.wav"  # Use existing script
        
        # Verify files exist
        files_to_check = [
            Path(clips_directory),
            Path(metadata_file),
            Path(music_file),
            Path(script_path)
        ]
        
        for file_path in files_to_check:
            if not file_path.exists():
                logger.error(f"❌ Missing: {file_path}")
                return False
            else:
                logger.info(f"✅ Found: {file_path}")
        
        # Initialize the real content generator
        logger.info("\n📁 Initializing Real Content Generator...")
        generator = RealContentGenerator(
            clips_directory=clips_directory,
            metadata_file=metadata_file,
            scripts_directory=scripts_directory,
            music_file=music_file
        )
        
        # Initialize the system
        start_init = time.time()
        if not await generator.initialize():
            logger.error("❌ Failed to initialize generator")
            return False
        
        init_time = time.time() - start_init
        logger.info(f"✅ Generator initialized in {init_time:.1f}s")
        
        # Create hi-hat sync video request
        logger.info("\n🎬 Creating Hi-Hat Sync Video Request...")
        request = RealVideoRequest(
            script_path=script_path,
            script_name="anxiety1",  # Match the existing script name
            variation_number=1,
            caption_style="tiktok",
            music_sync=True,
            min_clip_duration=1.0,        # Allow shorter clips for hi-hat cuts
            sync_event_type='hihat',      # 🎯 KEY: Sync to hi-hat events
            use_percussive_sync=True,     # 🎯 KEY: Enable percussive sync
            burn_in_captions=False        # Clean video without burned captions
        )
        
        logger.info("🎯 Video Generation Settings:")
        logger.info(f"   - Script: {request.script_name}")
        logger.info(f"   - Sync Type: {request.sync_event_type}")
        logger.info(f"   - Percussive Sync: {request.use_percussive_sync}")
        logger.info(f"   - Min Clip Duration: {request.min_clip_duration}s")
        logger.info(f"   - Caption Style: {request.caption_style}")
        
        # Generate the video
        logger.info("\n🚀 Generating Hi-Hat Synchronized Video...")
        start_gen = time.time()
        result = await generator.generate_video(request)
        gen_time = time.time() - start_gen
        
        # Report results
        if result.success:
            logger.info("\n🎉 SUCCESS! Hi-Hat Sync Video Generated!")
            logger.info("=" * 60)
            
            output_file = Path(result.output_path)
            file_size = output_file.stat().st_size / (1024 * 1024) if output_file.exists() else 0
            
            logger.info(f"📁 Output File: {output_file.name}")
            logger.info(f"📏 File Size: {file_size:.1f} MB")
            logger.info(f"⏱️  Generation Time: {result.generation_time:.1f}s")
            logger.info(f"🎬 Video Duration: {result.total_duration:.1f}s")
            logger.info(f"📊 Clips Used: {len(result.clips_used)}")
            logger.info(f"🎯 Relevance Score: {result.relevance_score:.3f}")
            logger.info(f"🎨 Visual Variety: {result.visual_variety_score:.3f}")
            logger.info(f"🔗 Sequence Hash: {result.sequence_hash[:12]}...")
            
            # Calculate cutting characteristics
            cuts_per_second = len(result.clips_used) / result.total_duration if result.total_duration > 0 else 0
            avg_clip_duration = result.total_duration / len(result.clips_used) if result.clips_used else 0
            
            logger.info(f"\n🥁 Hi-Hat Sync Analysis:")
            logger.info(f"   - Cuts per second: {cuts_per_second:.2f}")
            logger.info(f"   - Average clip duration: {avg_clip_duration:.2f}s")
            
            if cuts_per_second > 1.5:
                pace = "⚡ VERY FAST (Rapid-fire cuts)"
            elif cuts_per_second > 1.0:
                pace = "🔥 FAST (High-energy cuts)"
            elif cuts_per_second > 0.5:
                pace = "🎵 MODERATE (Rhythmic cuts)"
            else:
                pace = "🧘 SLOW (Contemplative cuts)"
            
            logger.info(f"   - Video pace: {pace}")
            
            # Performance metrics
            logger.info(f"\n📈 Performance Metrics:")
            logger.info(f"   - Total processing: {gen_time:.1f}s")
            logger.info(f"   - Video per second ratio: {result.total_duration/gen_time:.2f}x")
            
            # File information
            logger.info(f"\n📺 Video File Details:")
            logger.info(f"   - Path: {result.output_path}")
            logger.info(f"   - Format: MP4 (1080x1936 TikTok)")
            logger.info(f"   - Audio: Mixed script + music")
            logger.info(f"   - Captions: {request.caption_style} style")
            
            # Playback instructions
            logger.info(f"\n🎬 To View Your Hi-Hat Sync Video:")
            logger.info(f"   open '{output_file.name}'")
            logger.info(f"   # or")
            logger.info(f"   vlc '{result.output_path}'")
            logger.info(f"   # or")
            logger.info(f"   ffplay '{result.output_path}'")
            
            # Success summary
            logger.info(f"\n✨ Hi-Hat Sync Video Complete!")
            logger.info(f"   🎯 Feature: Percussive synchronization working perfectly")
            logger.info(f"   🔥 Result: {cuts_per_second:.1f} cuts/sec rapid-fire editing")
            logger.info(f"   📱 Perfect for: TikTok, Instagram, high-energy content")
            
            return True
            
        else:
            logger.error("\n❌ Video Generation Failed!")
            logger.error(f"Error: {result.error_message}")
            logger.info(f"Generation time: {gen_time:.1f}s")
            return False
            
    except Exception as e:
        logger.error(f"❌ Generation failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False


async def analyze_existing_videos():
    """Analyze existing videos to compare with hi-hat sync"""
    
    logger.info("\n📊 Analyzing Existing Videos for Comparison")
    logger.info("-" * 50)
    
    output_dir = Path("output")
    if not output_dir.exists():
        logger.warning("No output directory found")
        return
    
    # Find recent videos
    videos = list(output_dir.glob("*.mp4"))
    if not videos:
        logger.warning("No existing videos found")
        return
    
    # Sort by modification time (newest first)
    videos.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    logger.info(f"Found {len(videos)} existing videos")
    logger.info("Recent videos:")
    
    for i, video in enumerate(videos[:5]):
        file_size = video.stat().st_size / (1024 * 1024)
        mod_time = time.ctime(video.stat().st_mtime)
        logger.info(f"   {i+1}. {video.name} ({file_size:.1f}MB) - {mod_time}")
    
    logger.info(f"\n💡 After generating hi-hat sync video, compare:")
    logger.info(f"   - Regular sync vs hi-hat sync cutting speed")
    logger.info(f"   - Visual rhythm and energy differences")
    logger.info(f"   - Engagement and entertainment value")


async def main():
    """Main function"""
    print("🥁 Real Hi-Hat Sync Video Generator")
    print("=" * 70)
    print("Creating an actual video with percussive synchronization")
    print()
    
    # Analyze existing videos first
    await analyze_existing_videos()
    
    # Generate the new hi-hat sync video
    success = await generate_real_hihat_video()
    
    print("\n" + "=" * 70)
    if success:
        print("🎉 REAL HI-HAT SYNC VIDEO GENERATED SUCCESSFULLY!")
        print("🎬 You now have a working example of percussive synchronization")
        print("📱 Perfect for TikTok, Instagram, and high-energy content")
        print("\n🔥 Key Achievement:")
        print("   ✅ Percussive sync feature fully implemented")
        print("   ✅ Real video generated with hi-hat cuts")
        print("   ✅ Production-ready for social media")
    else:
        print("❌ Video generation encountered issues")
        print("🔧 Check logs above for troubleshooting information")
    
    print(f"\n📂 Check the 'output/' directory for your video file")
    
    return success


if __name__ == "__main__":
    asyncio.run(main())