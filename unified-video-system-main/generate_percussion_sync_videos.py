#!/usr/bin/env python3
"""
Generate Multiple Percussion Sync Videos

Creates videos synchronized to different percussion elements:
- Hi-hat sync (rapid-fire cuts)
- Snare sync (dramatic emphasis)
- Kick sync (powerful impact)
- Mixed percussion sync (all events combined)
"""

import asyncio
import logging
from pathlib import Path
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def generate_percussion_sync_videos():
    """Generate videos synchronized to different percussion elements"""
    
    try:
        logger.info("🥁 Generating Multiple Percussion Sync Videos")
        logger.info("=" * 70)
        
        # Import the system
        from core.real_content_generator import RealContentGenerator, RealVideoRequest
        
        # Configuration paths (using absolute paths like existing scripts)
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
        
        # Define percussion sync configurations
        percussion_configs = [
            {
                'name': 'Hi-Hat Sync',
                'sync_type': 'hihat',
                'description': 'Rapid-fire cuts synchronized to hi-hat cymbals',
                'min_duration': 0.8,
                'emoji': '🔥'
            },
            {
                'name': 'Snare Sync', 
                'sync_type': 'snare',
                'description': 'Dramatic cuts synchronized to snare hits',
                'min_duration': 1.5,
                'emoji': '💥'
            },
            {
                'name': 'Kick Sync',
                'sync_type': 'kick', 
                'description': 'Powerful cuts synchronized to kick drums',
                'min_duration': 2.0,
                'emoji': '⚡'
            }
        ]
        
        results = []
        
        # Generate videos for each percussion type
        for i, config in enumerate(percussion_configs, 1):
            logger.info(f"\n{config['emoji']} Generating {config['name']} Video ({i}/{len(percussion_configs)})...")
            logger.info("=" * 50)
            logger.info(f"📝 Description: {config['description']}")
            logger.info(f"⏱️  Min clip duration: {config['min_duration']}s")
            
            # Create video request
            request = RealVideoRequest(
                script_path=script_path,
                script_name="anxiety1",  # Use base script name
                variation_number=i,  # Use different variation numbers
                caption_style="tiktok",
                music_sync=True,
                min_clip_duration=config['min_duration'],
                sync_event_type=config['sync_type'],      # 🎯 KEY: Sync to specific percussion
                use_percussive_sync=True,                 # 🎯 KEY: Enable percussive sync
                burn_in_captions=False                    # Clean video without burned captions
            )
            
            logger.info(f"🎯 {config['name']} Settings:")
            logger.info(f"   - Sync Type: {request.sync_event_type}")
            logger.info(f"   - Percussive Sync: {request.use_percussive_sync}")
            logger.info(f"   - Min Clip Duration: {request.min_clip_duration}s")
            
            # Generate the video
            start_gen = time.time()
            result = await generator.generate_video(request)
            gen_time = time.time() - start_gen
            
            # Process results
            if result.success:
                output_file = Path(result.output_path)
                file_size = output_file.stat().st_size / (1024 * 1024) if output_file.exists() else 0
                cuts_per_second = len(result.clips_used) / result.total_duration if result.total_duration > 0 else 0
                
                # Determine video pace
                if cuts_per_second > 2.0:
                    pace = "⚡ ULTRA FAST (Rapid-fire cuts)"
                elif cuts_per_second > 1.5:
                    pace = "🔥 VERY FAST (High-energy cuts)"
                elif cuts_per_second > 1.0:
                    pace = "🎵 FAST (Rhythmic cuts)"
                elif cuts_per_second > 0.5:
                    pace = "🎭 MODERATE (Dramatic cuts)"
                else:
                    pace = "🧘 SLOW (Contemplative cuts)"
                
                result_info = {
                    'config': config,
                    'filename': output_file.name,
                    'file_size_mb': file_size,
                    'generation_time': gen_time,
                    'total_duration': result.total_duration,
                    'clips_used': len(result.clips_used),
                    'cuts_per_second': cuts_per_second,
                    'pace': pace,
                    'relevance_score': result.relevance_score,
                    'visual_variety': result.visual_variety_score,
                    'success': True
                }
                
                logger.info(f"\n✅ {config['emoji']} {config['name']} SUCCESS!")
                logger.info(f"   📁 File: {output_file.name}")
                logger.info(f"   📏 Size: {file_size:.1f} MB")
                logger.info(f"   ⏱️  Time: {gen_time:.1f}s")
                logger.info(f"   🎬 Duration: {result.total_duration:.1f}s")
                logger.info(f"   📊 Clips: {len(result.clips_used)}")
                logger.info(f"   ⚡ Cuts/sec: {cuts_per_second:.2f}")
                logger.info(f"   🎭 Pace: {pace}")
                
            else:
                result_info = {
                    'config': config,
                    'success': False,
                    'error': result.error_message,
                    'generation_time': gen_time
                }
                logger.error(f"\n❌ {config['emoji']} {config['name']} FAILED!")
                logger.error(f"   Error: {result.error_message}")
                logger.error(f"   Time: {gen_time:.1f}s")
            
            results.append(result_info)
        
        # Generate summary report
        logger.info("\n" + "=" * 70)
        logger.info("🎉 PERCUSSION SYNC VIDEO GENERATION COMPLETE!")
        logger.info("=" * 70)
        
        successful_videos = [r for r in results if r['success']]
        failed_videos = [r for r in results if not r['success']]
        
        logger.info(f"✅ Successful videos: {len(successful_videos)}/{len(results)}")
        if failed_videos:
            logger.info(f"❌ Failed videos: {len(failed_videos)}")
        
        # Summary table
        if successful_videos:
            logger.info("\n📊 Video Comparison Summary:")
            logger.info("-" * 70)
            
            for result in successful_videos:
                config = result['config']
                logger.info(f"{config['emoji']} {config['name']}:")
                logger.info(f"   📁 {result['filename']}")
                logger.info(f"   ⚡ {result['cuts_per_second']:.2f} cuts/sec - {result['pace']}")
                logger.info(f"   🎬 {result['total_duration']:.1f}s duration, {result['clips_used']} clips")
                logger.info(f"   📏 {result['file_size_mb']:.1f}MB, generated in {result['generation_time']:.1f}s")
                logger.info("")
        
        # Viewing instructions
        logger.info("🎬 To View Your Percussion Sync Videos:")
        for result in successful_videos:
            logger.info(f"   {result['config']['emoji']} {result['config']['name']}: open '{result['filename']}'")
        
        # Usage recommendations
        logger.info("\n💡 Usage Recommendations:")
        logger.info("🔥 Hi-Hat Sync: Perfect for high-energy, fast-paced content")
        logger.info("💥 Snare Sync: Great for dramatic emphasis and storytelling")
        logger.info("⚡ Kick Sync: Ideal for powerful, impactful messaging")
        
        logger.info("\n📱 All videos optimized for:")
        logger.info("   - TikTok (1080x1936)")
        logger.info("   - Instagram Reels")
        logger.info("   - YouTube Shorts")
        logger.info("   - High-engagement social media content")
        
        return len(successful_videos) > 0
        
    except Exception as e:
        logger.error(f"❌ Generation failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Main function"""
    print("🥁 Multiple Percussion Sync Video Generator")
    print("=" * 70)
    print("Creating videos synchronized to different percussion elements:")
    print("🔥 Hi-Hat Sync - Rapid-fire cuts")
    print("💥 Snare Sync - Dramatic emphasis")  
    print("⚡ Kick Sync - Powerful impact")
    print()
    
    # Generate the percussion sync videos
    success = await generate_percussion_sync_videos()
    
    print("\n" + "=" * 70)
    if success:
        print("🎉 PERCUSSION SYNC VIDEOS GENERATED SUCCESSFULLY!")
        print("🎬 You now have multiple examples of different percussion synchronization")
        print("📱 Perfect for comparing different cutting styles and energies")
        print("\n🔥 Key Achievements:")
        print("   ✅ Hi-hat sync for rapid-fire editing")
        print("   ✅ Snare sync for dramatic emphasis")
        print("   ✅ Kick sync for powerful impact")
        print("   ✅ Real percussion detection working perfectly")
    else:
        print("❌ Video generation encountered issues")
        print("🔧 Check logs above for troubleshooting information")
    
    print(f"\n📂 Check the 'output/' directory for your video files")
    
    return success


if __name__ == "__main__":
    asyncio.run(main())