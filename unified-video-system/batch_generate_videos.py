#!/usr/bin/env python3
"""
Batch generate videos with final caption styling - configurable number
"""

import asyncio
import time
import argparse
from pathlib import Path
from core.real_content_generator import RealContentGenerator, RealVideoRequest

async def batch_generate_videos(num_videos: int = 60, output_suffix: str = None):
    """Generate specified number of videos across all 11 scripts with final styling
    
    Args:
        num_videos: Number of videos to generate (default: 60)
        output_suffix: Optional suffix for output directory name
    """
    
    # Create output directory for batch
    dir_name = f"batch_{num_videos}_videos" + (f"_{output_suffix}" if output_suffix else "")
    batch_output_dir = Path(f"output/{dir_name}")
    batch_output_dir.mkdir(exist_ok=True)
    
    print(f"🎬 Starting batch generation of {num_videos} videos...")
    print(f"📂 Output directory: {batch_output_dir}")
    
    generator = RealContentGenerator(
        clips_directory='../MJAnime',
        metadata_file='../MJAnime/metadata_final_clean_shots.json',
        scripts_directory='../11-scripts-for-tiktok',
        music_file='music/Beanie (Slowed).mp3'
    )
    
    # Initialize generator
    print("🔧 Initializing generator...")
    await generator.initialize()
    
    # Available scripts
    scripts = [
        'anxiety1', 'safe1', 'miserable1', 'before', 'adhd', 
        'deadinside', 'diewithphone', '4', '6', '500friends', 'phone1'
    ]
    
    # Calculate variations per script
    videos_per_script = num_videos // len(scripts)
    extra_videos = num_videos % len(scripts)
    
    print(f"📊 Generating {videos_per_script} videos per script")
    print(f"📊 First {extra_videos} scripts get 1 extra video")
    
    successful_videos = 0
    failed_videos = 0
    start_time = time.time()
    
    for script_idx, script_name in enumerate(scripts):
        # Determine how many videos for this script
        num_videos = videos_per_script + (1 if script_idx < extra_videos else 0)
        
        print(f"\n🎯 Processing {script_name}: {num_videos} videos")
        
        for variation in range(1, num_videos + 1):
            try:
                # Generate unique variation number with timestamp entropy
                variation_num = (script_idx * 100) + variation + int(time.time() * 1000) % 1000
                
                request = RealVideoRequest(
                    script_path=f'../11-scripts-for-tiktok/{script_name}.wav',
                    script_name=script_name,
                    variation_number=variation_num,
                    caption_style='tiktok',
                    output_path=str(batch_output_dir / f"{script_name}_v{variation}_{int(time.time())}.mp4"),
                    burn_in_captions=True  # Enable captions for batch videos
                )
                
                print(f"  📹 Generating {script_name} variation {variation}...")
                result = await generator.generate_video(request)
                
                if result.success:
                    successful_videos += 1
                    print(f"  ✅ Success: {Path(result.output_path).name}")
                else:
                    failed_videos += 1
                    print(f"  ❌ Failed: {result.error_message}")
                    
            except Exception as e:
                failed_videos += 1
                print(f"  💥 Exception: {e}")
                
        # Progress update
        total_progress = successful_videos + failed_videos
        print(f"📈 Progress: {total_progress}/{num_videos} videos ({successful_videos} successful, {failed_videos} failed)")
    
    elapsed_time = time.time() - start_time
    
    print(f"\n🎉 Batch generation complete!")
    print(f"✅ Successful: {successful_videos}")
    print(f"❌ Failed: {failed_videos}")
    print(f"⏱️  Total time: {elapsed_time/60:.1f} minutes")
    print(f"📂 Videos saved to: {batch_output_dir}")
    
    if successful_videos > 0:
        avg_time = elapsed_time / successful_videos
        print(f"📊 Average time per video: {avg_time:.1f}s")

def main():
    """Main function with command line argument parsing"""
    parser = argparse.ArgumentParser(description='Batch generate videos with configurable count')
    parser.add_argument('--count', '-c', type=int, default=60, 
                       help='Number of videos to generate (default: 60)')
    parser.add_argument('--suffix', '-s', type=str, default=None,
                       help='Optional suffix for output directory name')
    
    args = parser.parse_args()
    
    # Validate count
    if args.count <= 0:
        print("❌ Error: Video count must be positive")
        return
    
    print(f"🎯 Configured to generate {args.count} videos")
    if args.suffix:
        print(f"📁 Output suffix: {args.suffix}")
    
    asyncio.run(batch_generate_videos(args.count, args.suffix))

if __name__ == "__main__":
    main()