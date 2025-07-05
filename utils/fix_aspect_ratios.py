#!/usr/bin/env python3
"""
Fix Aspect Ratios - Eliminate Black Bars from Video Clips
Simple script to normalize video aspect ratios using MoviePy
"""

from moviepy import VideoFileClip, CompositeVideoClip
from pathlib import Path
import sys

def fix_aspect_ratio(input_path, output_path=None, target_width=1080, target_height=1920):
    """
    Fix aspect ratio by scaling and cropping to eliminate black bars
    
    Args:
        input_path: Path to input video
        output_path: Path for output (optional)
        target_width: Target width (default 1080)
        target_height: Target height (default 1920 for 9:16)
    """
    input_path = Path(input_path)
    
    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_fixed{input_path.suffix}"
    
    print(f"🎬 Fixing aspect ratio for: {input_path.name}")
    
    # Load video
    clip = VideoFileClip(str(input_path))
    
    # Get current dimensions
    current_w, current_h = clip.w, clip.h
    print(f"   Current: {current_w}x{current_h}")
    print(f"   Target:  {target_width}x{target_height}")
    
    # Calculate scale factors
    scale_w = target_width / current_w
    scale_h = target_height / current_h
    
    # Use the larger scale factor to fill the target dimensions
    scale_factor = max(scale_w, scale_h)
    
    # Scale the clip
    scaled_w = int(current_w * scale_factor)
    scaled_h = int(current_h * scale_factor)
    scaled_clip = clip.resized((scaled_w, scaled_h))
    
    # Crop to exact target dimensions (center crop)
    if scaled_w > target_width:
        # Crop width
        x_offset = (scaled_w - target_width) // 2
        y_offset = 0
    else:
        # Crop height  
        x_offset = 0
        y_offset = (scaled_h - target_height) // 2
    
    final_clip = scaled_clip.cropped(
        x1=x_offset, 
        y1=y_offset,
        x2=x_offset + target_width, 
        y2=y_offset + target_height
    )
    
    # Export
    print(f"   Exporting to: {output_path}")
    final_clip.write_videofile(
        str(output_path),
        codec='libx264',
        audio_codec='aac',
        fps=clip.fps
    )
    
    # Clean up
    clip.close()
    scaled_clip.close()
    final_clip.close()
    
    print(f"   ✅ Fixed: {output_path}")
    return str(output_path)

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 fix_aspect_ratios.py <input_video> [output_video]")
        print("       python3 fix_aspect_ratios.py <input_directory>")
        sys.exit(1)
    
    input_arg = sys.argv[1]
    input_path = Path(input_arg)
    
    if input_path.is_dir():
        # Batch process directory
        video_files = list(input_path.glob("*.mp4"))
        if not video_files:
            print("No MP4 files found in directory")
            sys.exit(1)
        
        output_dir = input_path / "fixed"
        output_dir.mkdir(exist_ok=True)
        
        print(f"🎬 Processing {len(video_files)} videos...")
        
        for i, video_file in enumerate(video_files, 1):
            print(f"\n[{i}/{len(video_files)}]")
            output_file = output_dir / f"{video_file.stem}_fixed{video_file.suffix}"
            
            try:
                fix_aspect_ratio(video_file, output_file)
            except Exception as e:
                print(f"   ❌ Error: {e}")
                continue
        
        print(f"\n✅ Processed {len(video_files)} videos. Check the 'fixed' directory.")
    else:
        # Single file
        output_file = sys.argv[2] if len(sys.argv) > 2 else None
        fix_aspect_ratio(input_path, output_file)

if __name__ == "__main__":
    main()
