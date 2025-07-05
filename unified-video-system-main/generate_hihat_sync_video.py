#!/usr/bin/env python3
"""
Generate test video with hi-hat synchronization.

This script demonstrates the new percussive sync feature by creating a video
where cuts are synchronized to hi-hat events for rapid-fire editing effect.
"""

import asyncio
import logging
from pathlib import Path
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def generate_hihat_sync_video():
    """Generate a test video synchronized to hi-hat events"""
    
    try:
        logger.info("🥁 Generating test video with hi-hat synchronization...")
        
        # Import required modules
        from core.real_content_generator import RealContentGenerator, RealVideoRequest
        
        # Configuration - adjust paths as needed
        clips_directory = "MJAnime"
        metadata_file = "MJAnime/metadata.json" 
        scripts_directory = "scripts"
        music_file = "music/Beanie (Slowed).mp3"
        
        # Check if we have the required files
        script_path = Path("scripts/spiritual_script.txt")
        music_path = Path(music_file)
        
        if not script_path.exists():
            logger.error(f"❌ Script file not found: {script_path}")
            logger.info("Available scripts:")
            scripts_dir = Path("scripts")
            if scripts_dir.exists():
                for script in scripts_dir.glob("*.txt"):
                    logger.info(f"  - {script.name}")
            return False
            
        if not music_path.exists():
            logger.error(f"❌ Music file not found: {music_path}")
            return False
        
        # Initialize content generator
        logger.info("📁 Initializing content generator...")
        generator = RealContentGenerator(
            clips_directory=clips_directory,
            metadata_file=metadata_file,
            scripts_directory=scripts_directory,
            music_file=music_file
        )
        
        # Initialize
        initialized = await generator.initialize()
        if not initialized:
            logger.error("❌ Failed to initialize content generator")
            return False
            
        logger.info("✅ Content generator initialized successfully")
        
        # Analyze music for percussive events
        logger.info("🎵 Analyzing music for hi-hat events...")
        music_manager = generator.content_database.music_manager
        
        # Load and analyze music
        await music_manager.load_music_track()
        await music_manager.analyze_beats()
        
        # Get track info and hi-hat events
        track_info = music_manager.get_track_info()
        hihat_events = music_manager.get_percussive_events('hihat')
        
        logger.info(f"📊 Music analysis results:")
        logger.info(f"   - Tempo: {track_info.get('tempo', 'N/A')} BPM")
        logger.info(f"   - Total beats: {track_info.get('beat_count', 0)}")
        logger.info(f"   - Hi-hat events: {track_info.get('hihat_count', 0)}")
        logger.info(f"   - Kick events: {track_info.get('kick_count', 0)}")
        logger.info(f"   - Snare events: {track_info.get('snare_count', 0)}")
        
        if hihat_events:
            # Calculate average interval between hi-hat hits
            if len(hihat_events) > 1:
                intervals = [hihat_events[i+1] - hihat_events[i] for i in range(len(hihat_events)-1)]
                avg_interval = sum(intervals) / len(intervals)
                logger.info(f"   - Hi-hat average interval: {avg_interval:.2f}s")
                logger.info(f"   - Expected video style: {'Very fast-paced' if avg_interval < 1.0 else 'Fast-paced'}")
                
                # Show first few hi-hat events
                logger.info(f"   - First 10 hi-hat events: {hihat_events[:10]}")
        else:
            logger.warning("⚠️  No hi-hat events detected - falling back to beat sync")
        
        # Create video request with hi-hat synchronization
        logger.info("🎬 Creating video with hi-hat synchronization...")
        request = RealVideoRequest(
            script_path=str(script_path),
            script_name=script_path.stem,
            variation_number=1,
            caption_style="tiktok",
            music_sync=True,
            min_clip_duration=1.0,  # Allow shorter clips for fast hi-hat cuts
            sync_event_type='hihat',
            use_percussive_sync=True,
            burn_in_captions=False
        )
        
        logger.info(f"🎯 Video generation settings:")
        logger.info(f"   - Script: {request.script_name}")
        logger.info(f"   - Sync type: {request.sync_event_type}")
        logger.info(f"   - Percussive sync: {request.use_percussive_sync}")
        logger.info(f"   - Min clip duration: {request.min_clip_duration}s")
        logger.info(f"   - Caption style: {request.caption_style}")
        
        # Generate the video
        start_time = time.time()
        result = await generator.generate_video(request)
        generation_time = time.time() - start_time
        
        # Report results
        if result.success:
            logger.info("🎉 Hi-hat sync video generated successfully!")
            logger.info(f"📁 Output file: {Path(result.output_path).name}")
            logger.info(f"⏱️  Generation time: {result.generation_time:.1f}s")
            logger.info(f"🎬 Video duration: {result.total_duration:.1f}s")
            logger.info(f"📊 Relevance score: {result.relevance_score:.3f}")
            logger.info(f"🎨 Visual variety: {result.visual_variety_score:.3f}")
            logger.info(f"📋 Clips used: {len(result.clips_used)}")
            logger.info(f"🔗 Sequence hash: {result.sequence_hash[:8]}...")
            
            # Analyze the sync characteristics
            if hihat_events:
                video_duration = result.total_duration
                clips_in_video = len(result.clips_used)
                avg_clip_duration = video_duration / clips_in_video if clips_in_video > 0 else 0
                
                logger.info(f"🥁 Hi-hat sync analysis:")
                logger.info(f"   - Clips per video: {clips_in_video}")
                logger.info(f"   - Average clip duration: {avg_clip_duration:.2f}s")
                
                # Estimate cuts per second
                cuts_per_second = clips_in_video / video_duration if video_duration > 0 else 0
                logger.info(f"   - Cuts per second: {cuts_per_second:.2f}")
                
                if cuts_per_second > 1.0:
                    logger.info("   - Style: Very fast-paced editing ⚡")
                elif cuts_per_second > 0.5:
                    logger.info("   - Style: Fast-paced editing 🔥")
                else:
                    logger.info("   - Style: Moderate-paced editing 🎵")
            
            # Provide playback instructions
            output_path = Path(result.output_path)
            logger.info(f"\n📺 To view the hi-hat sync video:")
            logger.info(f"   open '{output_path.name}'")
            logger.info(f"   # or")
            logger.info(f"   vlc '{output_path}'")
            
            return True
        else:
            logger.error(f"❌ Video generation failed: {result.error_message}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Hi-hat sync video generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def generate_comparison_videos():
    """Generate multiple videos to compare different sync types"""
    
    logger.info("🔄 Generating comparison videos with different sync types...")
    
    sync_types = [
        ('beat', 'Regular beat synchronization'),
        ('kick', 'Kick drum synchronization (powerful impact)'),
        ('hihat', 'Hi-hat synchronization (rapid cuts)'),
        ('snare', 'Snare synchronization (dramatic emphasis)')
    ]
    
    results = {}
    
    for sync_type, description in sync_types:
        logger.info(f"\n🎬 Generating {sync_type} sync video...")
        logger.info(f"   Description: {description}")
        
        try:
            # Import required modules
            from core.real_content_generator import RealContentGenerator, RealVideoRequest
            
            # Configuration
            clips_directory = "MJAnime"
            metadata_file = "MJAnime/metadata.json"
            scripts_directory = "scripts"
            music_file = "music/Beanie (Slowed).mp3"
            script_path = Path("scripts/spiritual_script.txt")
            
            if not script_path.exists():
                logger.warning(f"⚠️  Script not found, skipping {sync_type} sync")
                continue
            
            # Initialize generator (reuse if possible)
            generator = RealContentGenerator(
                clips_directory=clips_directory,
                metadata_file=metadata_file,
                scripts_directory=scripts_directory,
                music_file=music_file
            )
            
            if not await generator.initialize():
                logger.error(f"❌ Failed to initialize for {sync_type} sync")
                continue
            
            # Create request
            request = RealVideoRequest(
                script_path=str(script_path),
                script_name=f"{script_path.stem}_{sync_type}",
                variation_number=1,
                caption_style="tiktok",
                music_sync=True,
                min_clip_duration=1.5 if sync_type == 'hihat' else 2.5,
                sync_event_type=sync_type,
                use_percussive_sync=(sync_type != 'beat')
            )
            
            # Generate video
            result = await generator.generate_video(request)
            results[sync_type] = result
            
            if result.success:
                logger.info(f"✅ {sync_type.capitalize()} sync: {Path(result.output_path).name}")
                logger.info(f"   Duration: {result.total_duration:.1f}s, "
                           f"Clips: {len(result.clips_used)}, "
                           f"Relevance: {result.relevance_score:.3f}")
            else:
                logger.error(f"❌ {sync_type.capitalize()} sync failed: {result.error_message}")
                
        except Exception as e:
            logger.error(f"❌ Error generating {sync_type} sync: {e}")
    
    # Summary
    successful_videos = [k for k, v in results.items() if v.success]
    logger.info(f"\n📊 Comparison video generation summary:")
    logger.info(f"   - Total attempts: {len(sync_types)}")
    logger.info(f"   - Successful: {len(successful_videos)}")
    logger.info(f"   - Generated videos: {successful_videos}")
    
    if successful_videos:
        logger.info(f"\n🎬 Compare the different sync styles:")
        for sync_type in successful_videos:
            result = results[sync_type]
            clips_per_second = len(result.clips_used) / result.total_duration if result.total_duration > 0 else 0
            logger.info(f"   - {sync_type}: {clips_per_second:.2f} cuts/sec ({Path(result.output_path).name})")
    
    return len(successful_videos) > 0


async def main():
    """Main function"""
    print("🥁 Hi-Hat Sync Video Generator")
    print("=" * 50)
    
    # Check if we should generate comparison videos
    import sys
    generate_comparison = '--compare' in sys.argv or '--all' in sys.argv
    
    if generate_comparison:
        print("🔄 Generating comparison videos with all sync types...")
        success = await generate_comparison_videos()
    else:
        print("🎯 Generating single hi-hat sync video...")
        success = await generate_hihat_sync_video()
    
    if success:
        print("\n🎉 Video generation completed successfully!")
        print("📺 Check the output directory for your new video(s)")
        
        if not generate_comparison:
            print("\n💡 Tip: Run with --compare to generate videos with all sync types")
    else:
        print("\n❌ Video generation failed. Check the logs above for details.")
        
        # Provide troubleshooting info
        print("\n🔧 Troubleshooting:")
        print("   1. Ensure MJAnime clips are available")
        print("   2. Check that scripts/spiritual_script.txt exists")
        print("   3. Verify music/Beanie (Slowed).mp3 is present")
        print("   4. Run the basic system tests first")


if __name__ == "__main__":
    asyncio.run(main())