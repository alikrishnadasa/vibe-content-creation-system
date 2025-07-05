#!/usr/bin/env python3
"""
Generate Optimized Combined Percussion Sync Video

Creates a video synchronized to BOTH snares AND hi-hats, but with optimized clip count
to avoid FFmpeg issues while maintaining the combined percussion energy.
"""

import asyncio
import logging
from pathlib import Path
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def generate_optimized_combined_percussion():
    """Generate optimized video synchronized to both snares and hi-hats"""
    
    try:
        logger.info("⚡ Generating Optimized Combined Percussion Sync Video")
        logger.info("=" * 60)
        logger.info("💥 Snare drums for dramatic emphasis")
        logger.info("🔥 Hi-hats for rapid-fire energy")
        logger.info("🎯 Optimized for reliable video generation")
        
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
        
        # Get the music manager to access percussion data
        music_manager = generator.content_database.music_manager
        track_info = music_manager.track_info
        
        if not track_info or not track_info.analyzed:
            logger.error("❌ Music track not analyzed")
            return False
        
        # Log detected percussion events
        logger.info(f"\n🎵 Detected Percussion Events:")
        logger.info(f"   💥 Snare events: {len(track_info.snare_times) if track_info.snare_times else 0}")
        logger.info(f"   🔥 Hi-hat events: {len(track_info.hihat_times) if track_info.hihat_times else 0}")
        
        # Combine snare and hi-hat events
        combined_events = []
        if track_info.snare_times:
            combined_events.extend(track_info.snare_times)
        if track_info.hihat_times:
            combined_events.extend(track_info.hihat_times)
        
        # Sort combined events by time
        combined_events.sort()
        
        logger.info(f"   ⚡ Combined events: {len(combined_events)}")
        
        # Optimize: Reduce clip count to avoid FFmpeg issues
        # Take every 2nd event to get ~72 clips instead of 144
        optimized_events = []
        for i in range(0, len(combined_events), 2):
            optimized_events.append(combined_events[i])
        
        logger.info(f"   🎯 Optimized events: {len(optimized_events)} (every 2nd event)")
        logger.info(f"   📊 Clips expected: ~{len(optimized_events)} (FFmpeg-friendly)")
        
        # Calculate average interval between optimized events
        if len(optimized_events) > 1:
            total_duration = optimized_events[-1] - optimized_events[0]
            avg_interval = total_duration / (len(optimized_events) - 1)
            logger.info(f"   ⏱️  Average interval: {avg_interval:.2f}s")
        
        # Temporarily modify the track_info to use optimized combined events
        original_hihat_times = track_info.hihat_times
        track_info.hihat_times = optimized_events
        
        try:
            # Create optimized combined percussion sync video request
            logger.info("\n⚡ Creating Optimized Combined Percussion Sync Video Request...")
            request = RealVideoRequest(
                script_path=script_path,
                script_name="anxiety1",
                variation_number=300,  # Use a unique variation number
                caption_style="tiktok",
                music_sync=True,
                min_clip_duration=1.0,          # Slightly longer clips for stability
                sync_event_type='hihat',        # Use hihat sync (now contains optimized combined events)
                use_percussive_sync=True,       # 🎯 KEY: Enable percussive sync
                burn_in_captions=False          # Clean video without burned captions
            )
            
            logger.info("⚡ Optimized Combined Percussion Sync Settings:")
            logger.info(f"   - Script: {request.script_name}")
            logger.info(f"   - Sync Type: Optimized Combined (snares + hi-hats)")
            logger.info(f"   - Total Events: {len(optimized_events)}")
            logger.info(f"   - Optimization: Every 2nd event (for stability)")
            logger.info(f"   - Percussive Sync: {request.use_percussive_sync}")
            logger.info(f"   - Min Clip Duration: {request.min_clip_duration}s")
            logger.info(f"   - Caption Style: {request.caption_style}")
            
            # Generate the video
            logger.info("\n🚀 Generating Optimized Combined Percussion Synchronized Video...")
            start_gen = time.time()
            result = await generator.generate_video(request)
            gen_time = time.time() - start_gen
            
        finally:
            # Restore original hi-hat times
            track_info.hihat_times = original_hihat_times
        
        # Report results
        if result.success:
            logger.info("\n🎉 SUCCESS! Optimized Combined Percussion Sync Video Generated!")
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
            
            logger.info(f"\n⚡ Optimized Combined Percussion Sync Analysis:")
            logger.info(f"   - Original combined events: {len(combined_events)}")
            logger.info(f"   - Optimized events used: {len(optimized_events)}")
            logger.info(f"   - Snare contributions: {len(track_info.snare_times) if track_info.snare_times else 0}")
            logger.info(f"   - Hi-hat contributions: {len(original_hihat_times) if original_hihat_times else 0}")
            logger.info(f"   - Cuts per second: {cuts_per_second:.2f}")
            logger.info(f"   - Average clip duration: {avg_clip_duration:.2f}s")
            
            # Determine video pace
            if cuts_per_second > 2.0:
                pace = "⚡ ULTRA FAST (Insane rapid-fire)"
            elif cuts_per_second > 1.5:
                pace = "🔥 VERY FAST (High-energy cuts)"
            elif cuts_per_second > 1.0:
                pace = "💥 FAST (Combined percussion power)"
            elif cuts_per_second > 0.5:
                pace = "🎭 DYNAMIC (Dramatic emphasis)"
            else:
                pace = "🎬 CINEMATIC (Epic storytelling)"
            
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
            
            # Comparison with individual sync types
            snare_count = len(track_info.snare_times) if track_info.snare_times else 0
            hihat_count = len(original_hihat_times) if original_hihat_times else 0
            
            logger.info(f"\n📊 Sync Type Comparison:")
            logger.info(f"   🔥 Hi-hat only: {hihat_count} events")
            logger.info(f"   💥 Snare only: {snare_count} events") 
            logger.info(f"   ⚡ Raw combined: {len(combined_events)} events")
            logger.info(f"   🎯 Optimized combined: {len(optimized_events)} events (stable generation)")
            
            # Playback instructions
            logger.info(f"\n🎬 To View Your Optimized Combined Percussion Sync Video:")
            logger.info(f"   open '{output_file.name}'")
            logger.info(f"   # or")
            logger.info(f"   vlc '{result.output_path}'")
            logger.info(f"   # or")
            logger.info(f"   ffplay '{result.output_path}'")
            
            # Success summary
            logger.info(f"\n✨ Optimized Combined Percussion Sync Video Complete!")
            logger.info(f"   ⚡ Feature: Snare + Hi-hat synchronization (optimized)")
            logger.info(f"   🔥 Result: {cuts_per_second:.1f} cuts/sec combined percussion editing")
            logger.info(f"   🎭 Perfect for: High engagement with reliable generation")
            logger.info(f"   📱 Ideal for: TikTok, Instagram Reels, YouTube Shorts")
            
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
    print("⚡ Optimized Combined Percussion Sync Video Generator")
    print("=" * 60)
    print("Creating a video synchronized to BOTH snares AND hi-hats")
    print("💥 Snare drums for dramatic emphasis")
    print("🔥 Hi-hats for rapid-fire energy") 
    print("🎯 Optimized for reliable video generation")
    print()
    
    # Generate the optimized combined percussion sync video
    success = await generate_optimized_combined_percussion()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 OPTIMIZED COMBINED PERCUSSION SYNC VIDEO GENERATED SUCCESSFULLY!")
        print("⚡ You now have the optimized percussion-synchronized video")
        print("🔥 Best of both worlds: snares AND hi-hats with stable generation!")
        print("\n🏆 Key Achievements:")
        print("   ✅ Combined snare + hi-hat synchronization")
        print("   ✅ Optimized clip count for reliable generation")
        print("   ✅ Perfect balance of drama and energy")
        print("   ✅ Production-ready social media content")
    else:
        print("❌ Video generation encountered issues")
        print("🔧 Check logs above for troubleshooting information")
    
    print(f"\n📂 Check the 'output/' directory for your video file")
    print("🎬 This version should generate successfully with combined percussion sync!")
    
    return success


if __name__ == "__main__":
    asyncio.run(main())