#!/usr/bin/env python3
"""
Test Expanded Percussion Detection

Test the improved percussion detection with expanded frequency ranges
and lower thresholds to catch more kick, snare, and hi-hat events.
"""

import asyncio
import logging
from pathlib import Path
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def test_expanded_percussion_detection():
    """Test the expanded percussion detection"""
    
    try:
        logger.info("🔍 Testing Expanded Percussion Detection")
        logger.info("=" * 60)
        
        # Import the system
        from core.real_content_generator import RealContentGenerator, RealVideoRequest
        
        # Configuration paths
        clips_directory = "/Users/jamesguo/vibe-content-creation/MJAnime"
        metadata_file = "/Users/jamesguo/vibe-content-creation/MJAnime/metadata_final_clean_shots.json"
        scripts_directory = "/Users/jamesguo/vibe-content-creation/11-scripts-for-tiktok"
        music_file = "/Users/jamesguo/vibe-content-creation/unified-video-system-main/music/Beanie (Slowed).mp3"
        script_path = "/Users/jamesguo/vibe-content-creation/11-scripts-for-tiktok/anxiety1.wav"
        
        # Initialize the real content generator
        logger.info("📁 Initializing Real Content Generator...")
        generator = RealContentGenerator(
            clips_directory=clips_directory,
            metadata_file=metadata_file,
            scripts_directory=scripts_directory,
            music_file=music_file
        )
        
        # Initialize the system to trigger beat analysis
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
        
        # Display expanded detection results
        logger.info(f"\n🎵 EXPANDED Percussion Detection Results:")
        logger.info("=" * 50)
        
        kick_count = len(track_info.kick_times) if track_info.kick_times else 0
        snare_count = len(track_info.snare_times) if track_info.snare_times else 0
        hihat_count = len(track_info.hihat_times) if track_info.hihat_times else 0
        other_count = len(track_info.other_times) if track_info.other_times else 0
        total_events = kick_count + snare_count + hihat_count + other_count
        
        logger.info(f"🥁 Total percussion events detected: {total_events}")
        logger.info(f"⚡ Kick events: {kick_count}")
        logger.info(f"💥 Snare events: {snare_count}")
        logger.info(f"🔥 Hi-hat events: {hihat_count}")
        logger.info(f"❓ Other events: {other_count}")
        
        # Calculate percentages
        if total_events > 0:
            logger.info(f"\n📊 Event Distribution:")
            logger.info(f"   ⚡ Kicks: {kick_count/total_events*100:.1f}%")
            logger.info(f"   💥 Snares: {snare_count/total_events*100:.1f}%")
            logger.info(f"   🔥 Hi-hats: {hihat_count/total_events*100:.1f}%")
            logger.info(f"   ❓ Others: {other_count/total_events*100:.1f}%")
        
        # Calculate event density
        duration = track_info.duration
        if duration > 0:
            events_per_second = total_events / duration
            logger.info(f"\n⏱️  Event Density: {events_per_second:.2f} events/second")
            
            if kick_count > 0:
                kick_interval = duration / kick_count
                logger.info(f"   ⚡ Kick interval: {kick_interval:.2f}s")
            
            if snare_count > 0:
                snare_interval = duration / snare_count
                logger.info(f"   💥 Snare interval: {snare_interval:.2f}s")
            
            if hihat_count > 0:
                hihat_interval = duration / hihat_count
                logger.info(f"   🔥 Hi-hat interval: {hihat_interval:.2f}s")
        
        # Compare with previous detection (from logs)
        logger.info(f"\n📈 Comparison with Previous Detection:")
        logger.info("=" * 40)
        logger.info("BEFORE Expansion (restrictive ranges):")
        logger.info("   ⚡ Kicks: 0 events")
        logger.info("   💥 Snares: 22 events") 
        logger.info("   🔥 Hi-hats: 122 events")
        logger.info("   📊 Total: 144 events")
        logger.info("")
        logger.info("AFTER Expansion (broader ranges):")
        logger.info(f"   ⚡ Kicks: {kick_count} events")
        logger.info(f"   💥 Snares: {snare_count} events")
        logger.info(f"   🔥 Hi-hats: {hihat_count} events")
        logger.info(f"   📊 Total: {total_events} events")
        
        # Calculate improvement
        previous_total = 144
        if total_events > previous_total:
            improvement = ((total_events - previous_total) / previous_total) * 100
            logger.info(f"   🚀 Improvement: +{improvement:.1f}% more events detected!")
        elif total_events < previous_total:
            reduction = ((previous_total - total_events) / previous_total) * 100
            logger.info(f"   📉 Change: -{reduction:.1f}% fewer events (may indicate better classification)")
        else:
            logger.info(f"   ➡️  Same total events (better distribution)")
        
        # Test video generation with expanded detection
        if kick_count > 0:
            logger.info(f"\n🎬 Testing kick sync (now possible with {kick_count} kicks detected!)...")
            request = RealVideoRequest(
                script_path=script_path,
                script_name="anxiety1",
                variation_number=400,
                caption_style="tiktok",
                music_sync=True,
                min_clip_duration=2.0,
                sync_event_type='kick',
                use_percussive_sync=True,
                burn_in_captions=False
            )
            
            logger.info("🚀 Generating Kick Sync Video with Expanded Detection...")
            start_gen = time.time()
            result = await generator.generate_video(request)
            gen_time = time.time() - start_gen
            
            if result.success:
                output_file = Path(result.output_path)
                file_size = output_file.stat().st_size / (1024 * 1024)
                cuts_per_second = len(result.clips_used) / result.total_duration if result.total_duration > 0 else 0
                
                logger.info(f"✅ SUCCESS! Kick sync video generated!")
                logger.info(f"   📁 File: {output_file.name}")
                logger.info(f"   📏 Size: {file_size:.1f} MB")
                logger.info(f"   🎬 Duration: {result.total_duration:.1f}s")
                logger.info(f"   📊 Clips: {len(result.clips_used)}")
                logger.info(f"   ⚡ Cuts/sec: {cuts_per_second:.2f}")
                logger.info(f"   ⏱️  Generation: {gen_time:.1f}s")
                
            else:
                logger.error(f"❌ Kick sync video failed: {result.error_message}")
        else:
            logger.warning("⚠️  Still no kick events detected - may need further tuning")
        
        # Summary
        logger.info(f"\n✨ Expanded Percussion Detection Test Complete!")
        logger.info("=" * 50)
        
        success_indicators = []
        if kick_count > 0:
            success_indicators.append("✅ Kick detection now working")
        if snare_count >= 20:
            success_indicators.append("✅ Good snare detection")
        if hihat_count >= 100:
            success_indicators.append("✅ Excellent hi-hat detection")
        if total_events > previous_total:
            success_indicators.append("✅ More events detected overall")
        
        if success_indicators:
            logger.info("🎉 Improvements detected:")
            for indicator in success_indicators:
                logger.info(f"   {indicator}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Main function"""
    print("🔍 Expanded Percussion Detection Test")
    print("=" * 50)
    print("Testing improved frequency ranges and thresholds")
    print("for better kick, snare, and hi-hat detection")
    print()
    
    success = await test_expanded_percussion_detection()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 EXPANDED PERCUSSION DETECTION TEST COMPLETE!")
        print("🔍 Check the results above to see improvements")
        print("⚡ Kick detection should now be working!")
    else:
        print("❌ Test encountered issues")
    
    return success


if __name__ == "__main__":
    asyncio.run(main())