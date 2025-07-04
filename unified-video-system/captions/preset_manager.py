"""
Caption Preset Manager
Manages modular caption styles and configurations
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
from enum import Enum
import json
from pathlib import Path


class CaptionDisplayMode(Enum):
    """Caption display modes"""
    ONE_WORD = "one_word"
    TWO_WORDS = "two_words"
    FULL_SENTENCE = "full_sentence"
    PHRASE_BASED = "phrase_based"
    KARAOKE = "karaoke"
    TYPEWRITER = "typewriter"


class CaptionPosition(Enum):
    """Caption positioning options"""
    CENTER = "center"
    BOTTOM = "bottom"
    TOP = "top"
    BOTTOM_LEFT = "bottom_left"
    BOTTOM_RIGHT = "bottom_right"
    CUSTOM = "custom"


@dataclass
class CaptionStyle:
    """Complete caption style configuration"""
    # Font settings
    font_family: str = "HelveticaTextNow-ExtraBold"
    font_size: int = 90
    font_color: str = "white"
    font_weight: str = "extra-bold"
    
    # Display settings
    display_mode: CaptionDisplayMode = CaptionDisplayMode.ONE_WORD
    position: CaptionPosition = CaptionPosition.CENTER
    
    # Styling
    background_color: Optional[str] = None
    background_opacity: float = 0.0
    outline_color: Optional[str] = None
    outline_width: int = 0
    shadow_color: Optional[str] = None
    shadow_offset: tuple[int, int] = (0, 0)
    shadow_blur: int = 0
    
    # Animation
    fade_in_duration: float = 0.0
    fade_out_duration: float = 0.0
    animation_type: Optional[str] = None  # "slide", "pop", "fade", etc.
    
    # Layout
    margin: int = 50
    line_height: float = 1.2
    letter_spacing: float = 0.0
    max_width: Optional[int] = None
    
    # Advanced
    uppercase: bool = False
    blur_background: bool = False
    custom_position: Optional[tuple[int, int]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        result = asdict(self)
        # Convert enums to values
        result['display_mode'] = self.display_mode.value
        result['position'] = self.position.value
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CaptionStyle':
        """Create from dictionary"""
        # Convert string values back to enums
        if 'display_mode' in data:
            data['display_mode'] = CaptionDisplayMode(data['display_mode'])
        if 'position' in data:
            data['position'] = CaptionPosition(data['position'])
        
        return cls(**data)


class CaptionPresetManager:
    """Manage caption style presets"""
    
    def __init__(self):
        self.presets = self._load_default_presets()
        self.custom_presets_path = Path("cache/caption_presets.json")
        self._load_custom_presets()
    
    def _load_default_presets(self) -> Dict[str, CaptionStyle]:
        """Load built-in presets"""
        return {
            # DEFAULT PRESET - Your specification
            "default": CaptionStyle(
                font_family="HelveticaTextNow-ExtraBold",
                font_size=100,
                font_color="white",
                display_mode=CaptionDisplayMode.ONE_WORD,
                position=CaptionPosition.CENTER
            ),
            
            # YouTube style
            "youtube": CaptionStyle(
                font_family="Roboto-Bold",
                font_size=72,
                font_color="white",
                display_mode=CaptionDisplayMode.TWO_WORDS,
                position=CaptionPosition.BOTTOM,
                background_color="black",
                background_opacity=0.75,
                margin=100
            ),
            
            # TikTok style
            "tiktok": CaptionStyle(
                font_family="HelveticaTextNow-ExtraBold",
                font_size=85,
                font_color="white",
                display_mode=CaptionDisplayMode.ONE_WORD,
                position=CaptionPosition.CENTER,
                uppercase=False,
                shadow_blur=2
            ),
            
            # Cinematic
            "cinematic": CaptionStyle(
                font_family="Futura-Medium",
                font_size=60,
                font_color="#FFFFCC",
                display_mode=CaptionDisplayMode.PHRASE_BASED,
                position=CaptionPosition.BOTTOM,
                fade_in_duration=0.3,
                fade_out_duration=0.3,
                letter_spacing=0.1,
                margin=150
            ),
            
            # Minimal
            "minimal": CaptionStyle(
                font_family="Inter-Light",
                font_size=48,
                font_color="white",
                display_mode=CaptionDisplayMode.FULL_SENTENCE,
                position=CaptionPosition.BOTTOM,
                fade_in_duration=0.2,
                fade_out_duration=0.2
            ),
            
            # Bold Impact
            "impact": CaptionStyle(
                font_family="Impact",
                font_size=120,
                font_color="yellow",
                display_mode=CaptionDisplayMode.ONE_WORD,
                position=CaptionPosition.CENTER,
                outline_color="black",
                outline_width=5,
                animation_type="pop"
            ),
            
            # Karaoke
            "karaoke": CaptionStyle(
                font_family="Arial-Bold",
                font_size=80,
                font_color="white",
                display_mode=CaptionDisplayMode.KARAOKE,
                position=CaptionPosition.BOTTOM,
                background_color="blue",
                background_opacity=0.8,
                margin=50
            )
        }
    
    def _load_custom_presets(self):
        """Load custom presets from disk"""
        if self.custom_presets_path.exists():
            try:
                with open(self.custom_presets_path, 'r') as f:
                    custom_data = json.load(f)
                    
                for name, data in custom_data.items():
                    self.presets[name] = CaptionStyle.from_dict(data)
                    
            except Exception as e:
                print(f"Warning: Could not load custom presets: {e}")
    
    def _save_custom_presets(self):
        """Save custom presets to disk"""
        # Only save non-default presets
        default_names = {
            "default", "youtube", "tiktok", "cinematic", 
            "minimal", "impact", "karaoke"
        }
        
        custom_presets = {
            name: style.to_dict()
            for name, style in self.presets.items()
            if name not in default_names
        }
        
        if custom_presets:
            self.custom_presets_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.custom_presets_path, 'w') as f:
                json.dump(custom_presets, f, indent=2)
    
    def get_preset(self, name: str = "default") -> CaptionStyle:
        """Get a preset by name"""
        return self.presets.get(name, self.presets["default"])
    
    def save_custom_preset(self, name: str, style: CaptionStyle):
        """Save a custom preset"""
        self.presets[name] = style
        self._save_custom_presets()
    
    def list_presets(self) -> List[str]:
        """List all available presets"""
        return list(self.presets.keys())
    
    def get_preset_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a preset"""
        if name in self.presets:
            style = self.presets[name]
            return {
                'name': name,
                'font_family': style.font_family,
                'font_size': style.font_size,
                'font_color': style.font_color,
                'display_mode': style.display_mode.value,
                'position': style.position.value,
                'has_outline': bool(style.outline_color),
                'has_background': bool(style.background_color),
                'has_animation': bool(style.animation_type or 
                                   style.fade_in_duration > 0 or 
                                   style.fade_out_duration > 0),
                'uppercase': style.uppercase
            }
        return None
    
    def create_variation(self, base_preset: str, modifications: Dict[str, Any]) -> CaptionStyle:
        """Create a variation of an existing preset"""
        base_style = self.get_preset(base_preset)
        style_dict = base_style.to_dict()
        
        # Apply modifications
        for key, value in modifications.items():
            if hasattr(base_style, key):
                style_dict[key] = value
        
        return CaptionStyle.from_dict(style_dict)


# Test function
def test_preset_manager():
    """Test preset manager functionality"""
    print("Testing Caption Preset Manager...")
    
    manager = CaptionPresetManager()
    
    # Test listing presets
    presets = manager.list_presets()
    print(f"Available presets: {presets}")
    
    # Test getting presets
    default_style = manager.get_preset("default")
    print(f"Default style: {default_style.font_family}, {default_style.font_size}px")
    
    tiktok_style = manager.get_preset("tiktok")
    print(f"TikTok style: {tiktok_style.font_family}, outline: {tiktok_style.outline_width}px")
    
    # Test preset info
    info = manager.get_preset_info("tiktok")
    print(f"TikTok info: {info}")
    
    # Test creating variation
    custom_style = manager.create_variation("default", {
        'font_size': 120,
        'font_color': 'red',
        'outline_color': 'black',
        'outline_width': 2
    })
    print(f"Custom variation: {custom_style.font_size}px, {custom_style.font_color}")
    
    # Test saving custom preset
    manager.save_custom_preset("my_custom", custom_style)
    print("Saved custom preset")
    
    print("✅ Preset manager tests passed!")


if __name__ == "__main__":
    test_preset_manager() 