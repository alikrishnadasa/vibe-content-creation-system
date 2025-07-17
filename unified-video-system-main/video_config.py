#!/usr/bin/env python3
"""
Video Generation Configuration
Modular configuration settings for video generation
Preserves all settings from generate_single_video.py
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional

@dataclass
class VideoConfig:
    """
    Centralized configuration for video generation
    Preserves 2:3 aspect ratio and caption settings from single video generator
    """
    
    # Directory paths - extracted from generate_single_video.py
    clips_directory: str = "/Users/jamesguo/vibe-content-creation/MJAnime/fixed"
    metadata_file: str = "/Users/jamesguo/vibe-content-creation/MJAnime/mjanime_metadata.json"
    scripts_directory: str = "/Users/jamesguo/vibe-content-creation/11-scripts-for-tiktok"
    music_file: str = "/Users/jamesguo/vibe-content-creation/unified-video-system-main/music/Beanie (Slowed).mp3"
    output_directory: str = "/Users/jamesguo/vibe-content-creation/unified-video-system-main/output"
    
    # Caption cache directory - modular support for any folder
    caption_cache_directory: str = "/Users/jamesguo/vibe-content-creation/unified-video-system-main/cache/pregenerated_captions"
    
    # Video settings - 2:3 aspect ratio (1080x1620) from FFmpeg processor
    target_resolution: Tuple[int, int] = (1080, 1620)  # 2:3 aspect ratio
    target_fps: int = 24
    
    # Caption settings - from generate_single_video.py and preset_manager.py
    caption_style: str = "default"  # Uses font_size=18 from preset_manager.py
    burn_in_captions: bool = True   # Burn captions directly into video
    
    # Audio/Music settings
    music_sync: bool = True
    
    # Generation settings
    clear_uniqueness_cache_frequency: int = 10  # Clear cache every N videos for variety
    
    def get_generator_kwargs(self) -> dict:
        """Get keyword arguments for RealContentGenerator initialization"""
        return {
            "clips_directory": self.clips_directory,
            "metadata_file": self.metadata_file,
            "scripts_directory": self.scripts_directory,
            "music_file": self.music_file,
            "output_directory": self.output_directory
        }
    
    def get_video_request_kwargs(self, script_path: str, script_name: str, variation_number: int) -> dict:
        """Get keyword arguments for RealVideoRequest"""
        return {
            "script_path": script_path,
            "script_name": script_name,
            "variation_number": variation_number,
            "caption_style": self.caption_style,
            "music_sync": self.music_sync,
            "burn_in_captions": self.burn_in_captions
        }
    
    def get_available_scripts(self) -> list:
        """Get list of available script files"""
        scripts_dir = Path(self.scripts_directory)
        return list(scripts_dir.glob("*.wav"))
    
    def get_caption_cache_path(self, script_name: str, style: str = None) -> Path:
        """Get path to cached caption file"""
        if style is None:
            style = self.caption_style
        cache_filename = f"{script_name}_{style}_captions.json"
        return Path(self.caption_cache_directory) / cache_filename
    
    def has_cached_captions(self, script_name: str, style: str = None) -> bool:
        """Check if cached captions exist for a script and style"""
        cache_path = self.get_caption_cache_path(script_name, style)
        return cache_path.exists()
    
    def get_available_cached_scripts(self, style: str = None) -> list:
        """Get list of scripts that have cached captions"""
        if style is None:
            style = self.caption_style
        
        cache_dir = Path(self.caption_cache_directory)
        if not cache_dir.exists():
            return []
        
        # Find all cache files for the specified style
        pattern = f"*_{style}_captions.json"
        cache_files = list(cache_dir.glob(pattern))
        
        # Extract script names from filenames
        script_names = []
        for cache_file in cache_files:
            # Remove the style suffix and .json extension
            script_name = cache_file.stem.replace(f"_{style}_captions", "")
            script_names.append(script_name)
        
        return script_names
    
    def validate_paths(self) -> bool:
        """Validate that all required paths exist"""
        required_paths = [
            Path(self.clips_directory),
            Path(self.metadata_file),
            Path(self.scripts_directory),
        ]
        
        for path in required_paths:
            if not path.exists():
                print(f"❌ Required path does not exist: {path}")
                return False
        
        # Check music file relative to current directory
        music_path = Path(self.music_file)
        if not music_path.exists():
            print(f"❌ Music file does not exist: {music_path}")
            return False
            
        return True

# Default configuration instance
DEFAULT_CONFIG = VideoConfig()

def create_custom_config(**kwargs) -> VideoConfig:
    """Create a custom configuration with overrides"""
    config_dict = {
        "clips_directory": kwargs.get("clips_directory", DEFAULT_CONFIG.clips_directory),
        "metadata_file": kwargs.get("metadata_file", DEFAULT_CONFIG.metadata_file),
        "scripts_directory": kwargs.get("scripts_directory", DEFAULT_CONFIG.scripts_directory),
        "music_file": kwargs.get("music_file", DEFAULT_CONFIG.music_file),
        "output_directory": kwargs.get("output_directory", DEFAULT_CONFIG.output_directory),
        "caption_cache_directory": kwargs.get("caption_cache_directory", DEFAULT_CONFIG.caption_cache_directory),
        "target_resolution": kwargs.get("target_resolution", DEFAULT_CONFIG.target_resolution),
        "target_fps": kwargs.get("target_fps", DEFAULT_CONFIG.target_fps),
        "caption_style": kwargs.get("caption_style", DEFAULT_CONFIG.caption_style),
        "burn_in_captions": kwargs.get("burn_in_captions", DEFAULT_CONFIG.burn_in_captions),
        "music_sync": kwargs.get("music_sync", DEFAULT_CONFIG.music_sync),
        "clear_uniqueness_cache_frequency": kwargs.get("clear_uniqueness_cache_frequency", DEFAULT_CONFIG.clear_uniqueness_cache_frequency)
    }
    
    return VideoConfig(**config_dict)

if __name__ == "__main__":
    # Test configuration
    config = DEFAULT_CONFIG
    print("🔧 Video Generation Configuration:")
    print(f"   📁 Clips: {config.clips_directory}")
    print(f"   📄 Metadata: {config.metadata_file}")
    print(f"   🎵 Scripts: {config.scripts_directory}")
    print(f"   🎶 Music: {config.music_file}")
    print(f"   📤 Output: {config.output_directory}")
    print(f"   📐 Resolution: {config.target_resolution} (2:3 aspect ratio)")
    print(f"   📝 Caption style: {config.caption_style}")
    print(f"   🔥 Burn captions: {config.burn_in_captions}")
    print(f"   🎵 Music sync: {config.music_sync}")
    
    print(f"\n📋 Available scripts: {len(config.get_available_scripts())}")
    print(f"✅ Paths valid: {config.validate_paths()}")