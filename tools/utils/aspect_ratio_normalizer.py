#!/usr/bin/env python3
"""
Aspect Ratio Normalizer for Video Clips
Eliminates black bars by normalizing video clips to consistent aspect ratios
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any
import json
from moviepy.editor import VideoFileClip, CompositeVideoClip
from dataclasses import dataclass
from enum import Enum

class CropStrategy(Enum):
    """Different strategies for handling aspect ratio normalization"""
    CENTER_CROP = "center_crop"        # Crop from center, may lose content
    SMART_CROP = "smart_crop"          # AI-guided cropping to preserve important content
    SCALE_TO_FIT = "scale_to_fit"      # Scale to fit target, may distort
    SCALE_CROP = "scale_crop"          # Scale then crop for best quality
    PAD_BLUR = "pad_blur"              # Pad with blurred background instead of black

@dataclass
class AspectRatioConfig:
    """Configuration for aspect ratio normalization"""
    target_width: int = 1080
    target_height: int = 1920  # 9:16 for TikTok/Instagram Reels
    strategy: CropStrategy = CropStrategy.SCALE_CROP
    quality: str = "high"  # "high", "medium", "fast"
    preserve_duration: bool = True
    blur_strength: float = 15.0  # For PAD_BLUR strategy
    
    @property
    def target_aspect_ratio(self) -> float:
        return self.target_width / self.target_height

class AspectRatioNormalizer:
    """Normalizes video clips to eliminate black bars and ensure consistent aspect ratios"""
    
    def __init__(self, config: AspectRatioConfig = None):
        self.config = config or AspectRatioConfig()
        self.processed_clips = {}
        
    def normalize_clip(self, input_path: str, output_path: str = None) -> str:
        """
        Normalize a single video clip to the target aspect ratio
        
        Args:
            input_path: Path to input video file
            output_path: Path for output file (optional)
            
        Returns:
            Path to the normalized video file
        """
        input_path = Path(input_path)
        
        if output_path is None:
            output_path = input_path.parent / f"{input_path.stem}_normalized{input_path.suffix}"
        
        print(f"🎬 Normalizing {input_path.name} using {self.config.strategy.value}")
        
        # Load video
        clip = VideoFileClip(str(input_path))
        
        # Get current dimensions
        current_w, current_h = clip.w, clip.h
        current_aspect = current_w / current_h
        target_aspect = self.config.target_aspect_ratio
        
        print(f"   Current: {current_w}x{current_h} (aspect: {current_aspect:.3f})")
        print(f"   Target:  {self.config.target_width}x{self.config.target_height} (aspect: {target_aspect:.3f})")
        
        # Apply normalization strategy
        if self.config.strategy == CropStrategy.CENTER_CROP:
            normalized_clip = self._center_crop(clip)
        elif self.config.strategy == CropStrategy.SMART_CROP:
            normalized_clip = self._smart_crop(clip)
        elif self.config.strategy == CropStrategy.SCALE_TO_FIT:
            normalized_clip = self._scale_to_fit(clip)
        elif self.config.strategy == CropStrategy.SCALE_CROP:
            normalized_clip = self._scale_crop(clip)
        elif self.config.strategy == CropStrategy.PAD_BLUR:
            normalized_clip = self._pad_blur(clip)
        else:
            normalized_clip = self._scale_crop(clip)  # Default
        
        # Export normalized clip
        normalized_clip.write_videofile(
            str(output_path),
            codec='libx264',
            audio_codec='aac',
            fps=clip.fps,
            verbose=False,
            logger=None
        )
        
        # Clean up
        clip.close()
        normalized_clip.close()
        
        print(f"   ✅ Saved: {output_path}")
        return str(output_path)
    
    def _center_crop(self, clip: VideoFileClip) -> VideoFileClip:
        """Center crop to target aspect ratio"""
        current_w, current_h = clip.w, clip.h
        target_aspect = self.config.target_aspect_ratio
        
        # Calculate crop dimensions
        if current_w / current_h > target_aspect:
            # Video is too wide, crop width
            new_w = int(current_h * target_aspect)
            new_h = current_h
            x_offset = (current_w - new_w) // 2
            y_offset = 0
        else:
            # Video is too tall, crop height
            new_w = current_w
            new_h = int(current_w / target_aspect)
            x_offset = 0
            y_offset = (current_h - new_h) // 2
        
        # Apply crop
        cropped = clip.crop(x1=x_offset, y1=y_offset, 
                           x2=x_offset + new_w, y2=y_offset + new_h)
        
        # Resize to target dimensions
        return cropped.resize((self.config.target_width, self.config.target_height))
    
    def _smart_crop(self, clip: VideoFileClip) -> VideoFileClip:
        """
        Smart crop that tries to preserve important content
        Uses center crop as fallback (can be enhanced with AI detection)
        """
        # For now, use center crop - can be enhanced with face/object detection
        return self._center_crop(clip)
    
    def _scale_to_fit(self, clip: VideoFileClip) -> VideoFileClip:
        """Scale video to fit target dimensions (may distort)"""
        return clip.resize((self.config.target_width, self.config.target_height))
    
    def _scale_crop(self, clip: VideoFileClip) -> VideoFileClip:
        """Scale to fill target dimensions, then crop excess (recommended)"""
        current_w, current_h = clip.w, clip.h
        target_w, target_h = self.config.target_width, self.config.target_height
        
        # Calculate scale factors
        scale_w = target_w / current_w
        scale_h = target_h / current_h
        
        # Use the larger scale factor to ensure we fill the target dimensions
        scale_factor = max(scale_w, scale_h)
        
        # Scale the clip
        scaled_w = int(current_w * scale_factor)
        scaled_h = int(current_h * scale_factor)
        scaled_clip = clip.resize((scaled_w, scaled_h))
        
        # Crop to exact target dimensions
        if scaled_w > target_w:
            # Crop width
            x_offset = (scaled_w - target_w) // 2
            y_offset = 0
        else:
            # Crop height
            x_offset = 0
            y_offset = (scaled_h - target_h) // 2
        
        return scaled_clip.crop(x1=x_offset, y1=y_offset,
                               x2=x_offset + target_w, y2=y_offset + target_h)
    
    def _pad_blur(self, clip: VideoFileClip) -> VideoFileClip:
        """Pad video with blurred background instead of black bars"""
        current_w, current_h = clip.w, clip.h
        target_w, target_h = self.config.target_width, self.config.target_height
        target_aspect = target_w / target_h
        current_aspect = current_w / current_h
        
        if abs(current_aspect - target_aspect) < 0.01:
            # Already correct aspect ratio, just resize
            return clip.resize((target_w, target_h))
        
        # Scale to fit within target dimensions
        if current_aspect > target_aspect:
            # Video is wider, scale by height
            scale_factor = target_h / current_h
        else:
            # Video is taller, scale by width
            scale_factor = target_w / current_w
        
        scaled_w = int(current_w * scale_factor)
        scaled_h = int(current_h * scale_factor)
        scaled_clip = clip.resize((scaled_w, scaled_h))
        
        # Create blurred background
        blurred_bg = (scaled_clip.resize((target_w, target_h))
                     .fx(lambda gf, t: cv2.GaussianBlur(gf(t), 
                                                       (int(self.config.blur_strength * 2 + 1), 
                                                        int(self.config.blur_strength * 2 + 1)), 
                                                       self.config.blur_strength)))
        
        # Position scaled clip on blurred background
        x_pos = (target_w - scaled_w) // 2
        y_pos = (target_h - scaled_h) // 2
        
        return CompositeVideoClip([blurred_bg, scaled_clip.set_position((x_pos, y_pos))],
                                 size=(target_w, target_h))
    
    def batch_normalize(self, input_directory: str, output_directory: str = None,
                       file_pattern: str = "*.mp4") -> Dict[str, str]:
        """
        Batch normalize all video files in a directory
        
        Args:
            input_directory: Directory containing video files
            output_directory: Directory for normalized files (optional)
            file_pattern: Pattern to match video files
            
        Returns:
            Dictionary mapping input paths to output paths
        """
        input_dir = Path(input_directory)
        
        if output_directory is None:
            output_dir = input_dir / "normalized"
        else:
            output_dir = Path(output_directory)
        
        output_dir.mkdir(exist_ok=True)
        
        # Find all video files
        video_files = list(input_dir.glob(file_pattern))
        
        if not video_files:
            print(f"⚠️  No video files found matching pattern: {file_pattern}")
            return {}
        
        print(f"🎬 Found {len(video_files)} video files to normalize")
        
        results = {}
        
        for i, input_file in enumerate(video_files, 1):
            print(f"\n[{i}/{len(video_files)}] Processing: {input_file.name}")
            
            output_file = output_dir / f"{input_file.stem}_normalized{input_file.suffix}"
            
            try:
                normalized_path = self.normalize_clip(str(input_file), str(output_file))
                results[str(input_file)] = normalized_path
            except Exception as e:
                print(f"   ❌ Error processing {input_file.name}: {e}")
                continue
        
        print(f"\n✅ Successfully normalized {len(results)} out of {len(video_files)} files")
        return results
    
    def update_metadata(self, metadata_path: str, normalized_mappings: Dict[str, str]):
        """
        Update metadata file with normalized clip information
        
        Args:
            metadata_path: Path to metadata JSON file
            normalized_mappings: Dictionary mapping original paths to normalized paths
        """
        metadata_file = Path(metadata_path)
        
        if not metadata_file.exists():
            print(f"⚠️  Metadata file not found: {metadata_path}")
            return
        
        # Load existing metadata
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Update clips with normalized versions
        updated_clips = []
        
        for clip in metadata.get('clips', []):
            original_filename = clip.get('filename', '')
            
            # Check if we have a normalized version
            normalized_path = None
            for orig_path, norm_path in normalized_mappings.items():
                if Path(orig_path).name == original_filename:
                    normalized_path = norm_path
                    break
            
            if normalized_path:
                # Create updated clip entry
                updated_clip = clip.copy()
                updated_clip['filename'] = Path(normalized_path).name
                updated_clip['resolution'] = f"{self.config.target_width}x{self.config.target_height}"
                updated_clip['aspect_ratio'] = f"{self.config.target_aspect_ratio:.3f}"
                updated_clip['normalized'] = True
                updated_clip['original_filename'] = original_filename
                updated_clip['normalization_strategy'] = self.config.strategy.value
                
                updated_clips.append(updated_clip)
            else:
                # Keep original clip
                updated_clips.append(clip)
        
        # Update metadata
        metadata['clips'] = updated_clips
        metadata['normalization_info'] = {
            'target_resolution': f"{self.config.target_width}x{self.config.target_height}",
            'target_aspect_ratio': self.config.target_aspect_ratio,
            'strategy': self.config.strategy.value,
            'normalized_count': len([c for c in updated_clips if c.get('normalized', False)]),
            'total_count': len(updated_clips)
        }
        
        # Save updated metadata
        backup_path = metadata_file.with_suffix('.backup.json')
        metadata_file.rename(backup_path)
        print(f"📄 Created backup: {backup_path}")
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"📄 Updated metadata: {metadata_file}")

def main():
    """CLI interface for aspect ratio normalization"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Normalize video aspect ratios to eliminate black bars")
    parser.add_argument("input", help="Input video file or directory")
    parser.add_argument("-o", "--output", help="Output file or directory")
    parser.add_argument("-s", "--strategy", choices=[s.value for s in CropStrategy], 
                       default="scale_crop", help="Normalization strategy")
    parser.add_argument("-w", "--width", type=int, default=1080, help="Target width")
    parser.add_argument("--height", type=int, default=1920, help="Target height")
    parser.add_argument("--batch", action="store_true", help="Process directory in batch mode")
    parser.add_argument("--update-metadata", help="Update metadata file with normalized info")
    
    args = parser.parse_args()
    
    # Create configuration
    config = AspectRatioConfig(
        target_width=args.width,
        target_height=args.height,
        strategy=CropStrategy(args.strategy)
    )
    
    # Create normalizer
    normalizer = AspectRatioNormalizer(config)
    
    if args.batch or Path(args.input).is_dir():
        # Batch processing
        results = normalizer.batch_normalize(args.input, args.output)
        
        # Update metadata if requested
        if args.update_metadata:
            normalizer.update_metadata(args.update_metadata, results)
    else:
        # Single file processing
        normalizer.normalize_clip(args.input, args.output)

if __name__ == "__main__":
    main() 