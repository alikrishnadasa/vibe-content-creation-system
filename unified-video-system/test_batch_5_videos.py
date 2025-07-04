#!/usr/bin/env python3
"""
Test batch generation with 5 videos to verify caption fixes
"""

import asyncio
import time
from pathlib import Path
from core.real_content_generator import RealContentGenerator, RealVideoRequest

async def test_5_videos():
    """Generate 5 test videos to verify captions and uniqueness fixes"""
    
    # Create output directory for test
    test_output_dir = Path("output/test_5_videos")
    test_output_dir.mkdir(exist_ok=True)
    
    print(f"🎬 Testing 5 videos with improved uniqueness and captions...")
    print(f"📂 Output directory: {test_output_dir}")
    
    generator = RealContentGenerator(
        clips_directory='../MJAnime',
        metadata_file='../MJAnime/metadata_final_clean_shots.json',
        scripts_directory='../11-scripts-for-tiktok',
        music_file='music/Beanie (Slowed).mp3'
    )
    
    # Initialize generator
    print("🔧 Initializing generator...")
    await generator.initialize()
    
    # Test with first 5 scripts
    test_scripts = ['anxiety1', 'safe1', 'miserable1', 'before', 'adhd']
    
    successful_videos = 0
    failed_videos = 0
    start_time = time.time()
    
    for script_idx, script_name in enumerate(test_scripts):
        try:
            # Generate unique variation number with high entropy
            variation_num = (script_idx * 1000) + int(time.time() * 1000) % 1000
            
            request = RealVideoRequest(
                script_path=f'../11-scripts-for-tiktok/{script_name}.wav',
                script_name=script_name,
                variation_number=variation_num,
                caption_style='tiktok',
                output_path=str(test_output_dir / f"{script_name}_test_{int(time.time())}.mp4"),
                burn_in_captions=True  # Enable captions
            )
            
            print(f"📹 Generating {script_name}...")
            result = await generator.generate_video(request)
            
            if result.success:
                successful_videos += 1
                print(f"✅ Success: {Path(result.output_path).name}")
                print(f"   Duration: {result.total_duration:.1f}s")
                print(f"   Clips: {len(result.clips_used)}")
                print(f"   Generation time: {result.generation_time:.2f}s")
            else:
                failed_videos += 1
                print(f"❌ Failed: {result.error_message}")
                
        except Exception as e:
            failed_videos += 1
            print(f"💥 Exception for {script_name}: {e}")
    
    elapsed_time = time.time() - start_time
    
    print(f"\n🎉 Test complete!")
    print(f"✅ Successful: {successful_videos}")
    print(f"❌ Failed: {failed_videos}")
    print(f"⏱️  Total time: {elapsed_time:.1f}s")
    print(f"📂 Videos saved to: {test_output_dir}")
    
    if successful_videos > 0:
        avg_time = elapsed_time / successful_videos
        print(f"📊 Average time per video: {avg_time:.1f}s")

if __name__ == "__main__":
    asyncio.run(test_5_videos())