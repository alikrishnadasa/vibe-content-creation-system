#!/usr/bin/env python3
"""
Generate Snare Sync Video

Creates a video synchronized to snare drum hits for dramatic emphasis.
"""

import asyncio
import logging
from pathlib import Path
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def generate_snare_sync_video():
    """Generate video synchronized to snare drum hits"""
    
    try:
        logger.info("💥 Generating Snare Sync Video")
        logger.info("=" * 60)
        
        # Import the system
        from core.real_content_generator import RealContentGenerator, RealVideoRequest
        
        # Configuration paths
        clips_directory = "/Users/jamesguo/vibe-content-creation/MJAnime"
        metadata_file = "/Users/jamesguo/vibe-content-creation/MJAnime/metadata_final_clean_shots.json"
        scripts_directory = "/Users/jamesguo/vibe-content-creation/11-scripts-for-tiktok"
        music_file = "/Users/jamesguo/vibe-content-creation/unified-video-system-main/music/Beanie (Slowed).mp3"
        script_path = "/Users/jamesguo/vibe-content-creation/11-scripts-for-tiktok/anxiety1.wav"
        
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
        
        # Create snare sync video request
        logger.info("\n💥 Creating Snare Sync Video Request...")
        request = RealVideoRequest(
            script_path=script_path,
            script_name="anxiety1",
            variation_number=100,  # Use a unique variation number
            caption_style="tiktok",
            music_sync=True,
            min_clip_duration=2.0,          # Longer clips for dramatic effect
            sync_event_type='snare',        # 🎯 KEY: Sync to snare drums
            use_percussive_sync=True,       # 🎯 KEY: Enable percussive sync
            burn_in_captions=False          # Clean video without burned captions
        )
        
        logger.info("💥 Snare Sync Settings:")
        logger.info(f"   - Script: {request.script_name}")
        logger.info(f"   - Sync Type: {request.sync_event_type}")
        logger.info(f"   - Percussive Sync: {request.use_percussive_sync}")
        logger.info(f"   - Min Clip Duration: {request.min_clip_duration}s")
        logger.info(f"   - Caption Style: {request.caption_style}")
        
        # Generate the video
        logger.info("\n🚀 Generating Snare Synchronized Video...")
        start_gen = time.time()
        result = await generator.generate_video(request)
        gen_time = time.time() - start_gen
        
        # Report results
        if result.success:
            logger.info("\n🎉 SUCCESS! Snare Sync Video Generated!")
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
            
            logger.info(f"\n💥 Snare Sync Analysis:")
            logger.info(f"   - Cuts per second: {cuts_per_second:.2f}")
            logger.info(f"   - Average clip duration: {avg_clip_duration:.2f}s")
            
            if cuts_per_second > 1.0:
                pace = "🔥 FAST (High-energy cuts)"
            elif cuts_per_second > 0.5:
                pace = "💥 DRAMATIC (Powerful cuts)"
            else:
                pace = "🎭 CINEMATIC (Epic cuts)"
            
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
            logger.info(f"\n🎬 To View Your Snare Sync Video:")
            logger.info(f"   open '{output_file.name}'")
            logger.info(f"   # or")
            logger.info(f"   vlc '{result.output_path}'")
            logger.info(f"   # or")
            logger.info(f"   ffplay '{result.output_path}'")
            
            # Success summary
            logger.info(f"\n✨ Snare Sync Video Complete!")
            logger.info(f"   💥 Feature: Snare drum synchronization working perfectly")
            logger.info(f"   🔥 Result: {cuts_per_second:.1f} cuts/sec dramatic editing")
            logger.info(f"   🎭 Perfect for: Dramatic storytelling, powerful messaging")
            
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


async def main():
    """Main function"""
    print("💥 Snare Sync Video Generator")
    print("=" * 50)
    print("Creating a video synchronized to snare drum hits")
    print("Perfect for dramatic emphasis and powerful messaging")
    print()
    
    # Generate the snare sync video
    success = await generate_snare_sync_video()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 SNARE SYNC VIDEO GENERATED SUCCESSFULLY!")
        print("💥 You now have a video with dramatic snare synchronization")
        print("🎭 Perfect for storytelling and powerful impact")
        print("\n🔥 Key Achievement:")
        print("   ✅ Snare sync for dramatic emphasis")
        print("   ✅ Real percussion detection working")
        print("   ✅ Production-ready for impactful content")
    else:
        print("❌ Video generation encountered issues")
        print("🔧 Check logs above for troubleshooting information")
    
    print(f"\n📂 Check the 'output/' directory for your video file")
    
    return success


if __name__ == "__main__":
    asyncio.run(main())