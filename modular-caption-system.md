# Modular Caption System - Flexible Configuration Architecture

## Overview
A highly modular caption system that allows easy configuration switching while maintaining perfect synchronization. Default configuration uses HelveticaTextNow-ExtraBold at 100px, white color, with one word displayed at a time.

---

## 1. CORE CAPTION SYSTEM ARCHITECTURE

### 1.1 Caption Configuration Classes
```python
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Literal
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
    BOTTOM_LEFT = "bottom-left"
    BOTTOM_RIGHT = "bottom-right"
    CUSTOM = "custom"

@dataclass
class CaptionStyle:
    """Complete caption style configuration"""
    # Font settings
    font_family: str = "HelveticaTextNow-ExtraBold"
    font_size: int = 100
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
        return {
            'font_family': self.font_family,
            'font_size': self.font_size,
            'font_color': self.font_color,
            'font_weight': self.font_weight,
            'display_mode': self.display_mode.value,
            'position': self.position.value,
            'background_color': self.background_color,
            'background_opacity': self.background_opacity,
            'outline_color': self.outline_color,
            'outline_width': self.outline_width,
            'shadow_color': self.shadow_color,
            'shadow_offset': self.shadow_offset,
            'shadow_blur': self.shadow_blur,
            'fade_in_duration': self.fade_in_duration,
            'fade_out_duration': self.fade_out_duration,
            'animation_type': self.animation_type,
            'margin': self.margin,
            'line_height': self.line_height,
            'letter_spacing': self.letter_spacing,
            'max_width': self.max_width,
            'uppercase': self.uppercase,
            'blur_background': self.blur_background,
            'custom_position': self.custom_position
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CaptionStyle':
        """Create from dictionary"""
        data['display_mode'] = CaptionDisplayMode(data.get('display_mode', 'one_word'))
        data['position'] = CaptionPosition(data.get('position', 'center'))
        return cls(**data)
```

### 1.2 Preset Configuration Manager
```python
class CaptionPresetManager:
    """Manage caption style presets"""
    
    def __init__(self):
        self.presets = self._load_default_presets()
        self.custom_presets_path = Path("caption_presets.json")
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
                outline_color="black",
                outline_width=3,
                uppercase=True
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
```

---

## 2. MODULAR CAPTION ENGINE

### 2.1 Main Caption Engine
```python
class ModularCaptionEngine:
    """Modular caption engine with configurable styles"""
    
    def __init__(self, style: Optional[CaptionStyle] = None):
        # Use default style if none provided
        self.style = style or CaptionPresetManager().get_preset("default")
        
        # Initialize components
        self.sync_engine = CaptionSyncEngine()
        self.renderer = self._create_renderer()
        self.animator = CaptionAnimator(self.style)
        self.validator = CaptionValidator()
        
    def _create_renderer(self) -> 'BaseCaptionRenderer':
        """Create appropriate renderer based on style"""
        if self.style.display_mode == CaptionDisplayMode.KARAOKE:
            return KaraokeCaptionRenderer(self.style)
        elif self.style.display_mode == CaptionDisplayMode.TYPEWRITER:
            return TypewriterCaptionRenderer(self.style)
        else:
            return StandardCaptionRenderer(self.style)
    
    def configure(self, **kwargs):
        """Update configuration dynamically"""
        for key, value in kwargs.items():
            if hasattr(self.style, key):
                setattr(self.style, key, value)
        
        # Recreate renderer if display mode changed
        if 'display_mode' in kwargs:
            self.renderer = self._create_renderer()
    
    def use_preset(self, preset_name: str):
        """Switch to a preset configuration"""
        preset_manager = CaptionPresetManager()
        self.style = preset_manager.get_preset(preset_name)
        self.renderer = self._create_renderer()
        self.animator = CaptionAnimator(self.style)
    
    def create_captions(self, audio_path: str, transcript: str, 
                       video_duration: float) -> List['Caption']:
        """Create captions with current configuration"""
        
        # Step 1: Generate sync timings
        sync_data = self.sync_engine.generate_sync(
            audio_path, 
            transcript, 
            self.style.display_mode
        )
        
        # Step 2: Create caption objects
        captions = self._create_caption_objects(sync_data, transcript)
        
        # Step 3: Apply styling
        styled_captions = self.renderer.apply_styling(captions)
        
        # Step 4: Add animations if configured
        if self.style.animation_type:
            animated_captions = self.animator.add_animations(styled_captions)
        else:
            animated_captions = styled_captions
        
        # Step 5: Validate
        validation_result = self.validator.validate(
            animated_captions, 
            video_duration
        )
        
        if not validation_result.is_valid:
            print(f"⚠️ Caption validation warnings: {validation_result.warnings}")
        
        return animated_captions
```

### 2.2 Caption Sync Engine
```python
class CaptionSyncEngine:
    """Handle synchronization for different display modes"""
    
    def __init__(self):
        self.phoneme_detector = PhonemeDetector()
        self.word_aligner = ForcedWordAligner()
        self.phrase_detector = PhraseDetector()
        
    def generate_sync(self, audio_path: str, transcript: str, 
                     display_mode: CaptionDisplayMode) -> List[SyncSegment]:
        """Generate sync data based on display mode"""
        
        # Load audio
        audio, sr = librosa.load(audio_path, sr=None)
        
        # Get word-level alignment first
        word_timings = self.word_aligner.align(audio, sr, transcript)
        
        # Process based on display mode
        if display_mode == CaptionDisplayMode.ONE_WORD:
            return self._create_one_word_segments(word_timings)
            
        elif display_mode == CaptionDisplayMode.TWO_WORDS:
            return self._create_two_word_segments(word_timings)
            
        elif display_mode == CaptionDisplayMode.FULL_SENTENCE:
            return self._create_sentence_segments(word_timings, transcript)
            
        elif display_mode == CaptionDisplayMode.PHRASE_BASED:
            phrases = self.phrase_detector.detect_phrases(transcript, audio)
            return self._create_phrase_segments(word_timings, phrases)
            
        elif display_mode == CaptionDisplayMode.KARAOKE:
            return self._create_karaoke_segments(word_timings, audio, sr)
            
        elif display_mode == CaptionDisplayMode.TYPEWRITER:
            return self._create_typewriter_segments(word_timings)
    
    def _create_one_word_segments(self, word_timings: List[WordTiming]) -> List[SyncSegment]:
        """Create segments for one-word display"""
        segments = []
        
        for timing in word_timings:
            segments.append(SyncSegment(
                text=timing.word,
                start_time=timing.start,
                end_time=timing.end,
                words=[timing.word],
                segment_type="single_word"
            ))
            
        return segments
    
    def _create_two_word_segments(self, word_timings: List[WordTiming]) -> List[SyncSegment]:
        """Create segments for two-word display"""
        segments = []
        
        for i in range(0, len(word_timings), 2):
            words = [word_timings[i]]
            if i + 1 < len(word_timings):
                words.append(word_timings[i + 1])
            
            segment_text = " ".join(w.word for w in words)
            segments.append(SyncSegment(
                text=segment_text,
                start_time=words[0].start,
                end_time=words[-1].end,
                words=[w.word for w in words],
                segment_type="two_word"
            ))
            
        return segments
```

---

## 3. CAPTION RENDERERS

### 3.1 Base Renderer Interface
```python
from abc import ABC, abstractmethod

class BaseCaptionRenderer(ABC):
    """Base class for all caption renderers"""
    
    def __init__(self, style: CaptionStyle):
        self.style = style
        self.font_cache = FontCache()
        
    @abstractmethod
    def render_caption(self, text: str, frame: np.ndarray, 
                      time: float) -> np.ndarray:
        """Render caption on frame"""
        pass
    
    def apply_styling(self, captions: List[Caption]) -> List[Caption]:
        """Apply style configuration to captions"""
        for caption in captions:
            caption.style = self.style
        return captions
```

### 3.2 Standard Caption Renderer
```python
class StandardCaptionRenderer(BaseCaptionRenderer):
    """Standard caption renderer for most display modes"""
    
    def __init__(self, style: CaptionStyle):
        super().__init__(style)
        self.setup_moviepy_style()
        
    def setup_moviepy_style(self):
        """Prepare MoviePy text parameters"""
        self.text_params = {
            'fontsize': self.style.font_size,
            'color': self.style.font_color,
            'font': self.style.font_family,
            'stroke_color': self.style.outline_color,
            'stroke_width': self.style.outline_width,
            'method': 'caption'
        }
        
        # Add shadow if configured
        if self.style.shadow_color:
            self.text_params['shadow'] = True
            self.text_params['shadow_color'] = self.style.shadow_color
            self.text_params['shadow_offset'] = self.style.shadow_offset
    
    def render_caption(self, text: str, frame: np.ndarray, 
                      time: float) -> np.ndarray:
        """Render caption using MoviePy"""
        
        # Apply text transformations
        if self.style.uppercase:
            text = text.upper()
        
        # Create text clip
        txt_clip = TextClip(text, **self.text_params)
        
        # Position the text
        if self.style.position == CaptionPosition.CENTER:
            txt_clip = txt_clip.set_position('center')
        elif self.style.position == CaptionPosition.BOTTOM:
            txt_clip = txt_clip.set_position(('center', 'bottom'))
        elif self.style.position == CaptionPosition.TOP:
            txt_clip = txt_clip.set_position(('center', 'top'))
        elif self.style.position == CaptionPosition.CUSTOM:
            txt_clip = txt_clip.set_position(self.style.custom_position)
        
        # Add background if configured
        if self.style.background_color and self.style.background_opacity > 0:
            bg_clip = self._create_background(txt_clip.size)
            txt_clip = CompositeVideoClip([bg_clip, txt_clip])
        
        # Composite onto frame
        frame_clip = ImageClip(frame)
        composite = CompositeVideoClip([frame_clip, txt_clip])
        
        return composite.get_frame(0)
    
    def _create_background(self, text_size: tuple) -> VideoClip:
        """Create background for caption"""
        width, height = text_size
        padding = 20
        
        bg_size = (width + padding * 2, height + padding)
        color = self._hex_to_rgb(self.style.background_color)
        
        return (ColorClip(bg_size, color)
                .set_opacity(self.style.background_opacity))
```

### 3.3 GPU-Accelerated Renderer
```python
class GPUCaptionRenderer(BaseCaptionRenderer):
    """GPU-accelerated caption renderer for performance"""
    
    def __init__(self, style: CaptionStyle):
        super().__init__(style)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.font_atlas = self._create_font_atlas()
        self.shader = self._compile_shader()
        
    def _create_font_atlas(self) -> Dict:
        """Pre-render font characters to GPU texture"""
        import freetype
        
        face = freetype.Face(f"fonts/{self.style.font_family}.ttf")
        face.set_pixel_sizes(0, self.style.font_size)
        
        # Create texture atlas
        atlas_size = 2048
        atlas = torch.zeros((atlas_size, atlas_size, 4), device=self.device)
        char_map = {}
        
        x, y = 0, 0
        row_height = 0
        
        for char_code in range(32, 127):  # ASCII printable
            face.load_char(char_code)
            bitmap = face.glyph.bitmap
            
            if bitmap.buffer and len(bitmap.buffer) > 0:
                # Convert to tensor
                char_data = np.array(bitmap.buffer).reshape(bitmap.rows, bitmap.width)
                char_tensor = torch.from_numpy(char_data).float() / 255.0
                
                # Check space
                if x + bitmap.width > atlas_size:
                    x = 0
                    y += row_height + 2
                    row_height = 0
                
                # Place in atlas
                atlas[y:y+bitmap.rows, x:x+bitmap.width, 3] = char_tensor
                
                # Set color channels
                if self.style.font_color == "white":
                    atlas[y:y+bitmap.rows, x:x+bitmap.width, :3] = 1.0
                
                char_map[chr(char_code)] = {
                    'x': x, 'y': y,
                    'width': bitmap.width,
                    'height': bitmap.rows,
                    'advance': face.glyph.advance.x >> 6,
                    'bearing_x': face.glyph.bitmap_left,
                    'bearing_y': face.glyph.bitmap_top
                }
                
                x += bitmap.width + 2
                row_height = max(row_height, bitmap.rows)
                
        return {'atlas': atlas, 'char_map': char_map}
    
    def render_caption(self, text: str, frame: np.ndarray, 
                      time: float) -> np.ndarray:
        """Render caption on GPU"""
        
        # Convert frame to GPU
        frame_tensor = torch.from_numpy(frame).to(self.device).float() / 255.0
        
        # Calculate text position
        text_width = sum(self.font_atlas['char_map'].get(c, {}).get('advance', 0) 
                        for c in text)
        
        if self.style.position == CaptionPosition.CENTER:
            x = (frame.shape[1] - text_width) // 2
            y = frame.shape[0] // 2
        else:
            x, y = self._calculate_position(frame.shape, text_width)
        
        # Render each character
        for char in text:
            if char in self.font_atlas['char_map']:
                char_info = self.font_atlas['char_map'][char]
                
                # Extract character from atlas
                char_tex = self.font_atlas['atlas'][
                    char_info['y']:char_info['y']+char_info['height'],
                    char_info['x']:char_info['x']+char_info['width']
                ]
                
                # Composite onto frame
                self._composite_char(frame_tensor, char_tex, x, y, char_info)
                
                x += char_info['advance']
        
        # Convert back to numpy
        return (frame_tensor * 255).byte().cpu().numpy()
```

---

## 4. USAGE EXAMPLES

### 4.1 Basic Usage with Default Settings
```python
# Create engine with default settings
# (HelveticaTextNow-ExtraBold, 100px, white, one word at a time)
caption_engine = ModularCaptionEngine()

# Generate captions
captions = caption_engine.create_captions(
    audio_path="narration.mp3",
    transcript="Like water flowing through ancient stones",
    video_duration=30.0
)

# Apply to video
video = VideoFileClip("input.mp4")
captioned_video = apply_captions_to_video(video, captions, caption_engine)
captioned_video.write_videofile("output_with_captions.mp4")
```

### 4.2 Using Different Presets
```python
# Initialize engine
caption_engine = ModularCaptionEngine()

# Switch to TikTok style
caption_engine.use_preset("tiktok")

# Or YouTube style
caption_engine.use_preset("youtube")

# List available presets
presets = CaptionPresetManager().list_presets()
print(f"Available presets: {presets}")
```

### 4.3 Custom Configuration
```python
# Create custom style
custom_style = CaptionStyle(
    font_family="Montserrat-Black",
    font_size=90,
    font_color="#FFD700",  # Gold
    display_mode=CaptionDisplayMode.TWO_WORDS,
    position=CaptionPosition.BOTTOM,
    outline_color="black",
    outline_width=2,
    fade_in_duration=0.2,
    fade_out_duration=0.2,
    animation_type="slide"
)

# Use custom style
caption_engine = ModularCaptionEngine(custom_style)

# Or modify existing configuration
caption_engine.configure(
    font_size=120,
    font_color="red",
    display_mode=CaptionDisplayMode.ONE_WORD
)
```

### 4.4 Complete Pipeline Integration
```python
class VideoProductionPipeline:
    """Integration with video production pipeline"""
    
    def __init__(self):
        # Initialize with default caption style
        self.caption_engine = ModularCaptionEngine()
        
    def create_video_with_captions(self, script: str, video_clips: List[str],
                                  caption_preset: str = "default") -> str:
        """Create video with specified caption style"""
        
        # Set caption style
        self.caption_engine.use_preset(caption_preset)
        
        # Generate audio
        audio_path = self.generate_audio(script)
        
        # Create video from clips
        video = self.assemble_video(video_clips)
        
        # Generate and apply captions
        captions = self.caption_engine.create_captions(
            audio_path=audio_path,
            transcript=script,
            video_duration=video.duration
        )
        
        # Apply captions with current style
        final_video = self.apply_captions_with_style(video, captions)
        
        # Add audio and export
        final_video = final_video.set_audio(AudioFileClip(audio_path))
        output_path = "final_video_with_styled_captions.mp4"
        final_video.write_videofile(output_path)
        
        return output_path
    
    def apply_captions_with_style(self, video: VideoFileClip, 
                                 captions: List[Caption]) -> VideoFileClip:
        """Apply styled captions to video"""
        
        def add_captions(get_frame, t):
            frame = get_frame(t)
            
            # Find active caption
            for caption in captions:
                if caption.start_time <= t <= caption.end_time:
                    # Render with current style
                    frame = self.caption_engine.renderer.render_caption(
                        caption.text, frame, t
                    )
                    break
                    
            return frame
            
        return video.fl(add_captions)

# Usage
pipeline = VideoProductionPipeline()

# Create video with default captions
video1 = pipeline.create_video_with_captions(
    script="Your script here",
    video_clips=["clip1.mp4", "clip2.mp4"],
    caption_preset="default"  # Your specified style
)

# Create another with TikTok style
video2 = pipeline.create_video_with_captions(
    script="Another script",
    video_clips=["clip3.mp4", "clip4.mp4"],
    caption_preset="tiktok"
)
```

### 4.5 Saving and Loading Custom Presets
```python
# Create a custom preset
my_style = CaptionStyle(
    font_family="HelveticaTextNow-ExtraBold",
    font_size=110,
    font_color="white",
    display_mode=CaptionDisplayMode.ONE_WORD,
    position=CaptionPosition.CENTER,
    shadow_color="black",
    shadow_offset=(2, 2),
    shadow_blur=4
)

# Save it
preset_manager = CaptionPresetManager()
preset_manager.save_custom_preset("my_brand_style", my_style)

# Use it later
caption_engine = ModularCaptionEngine()
caption_engine.use_preset("my_brand_style")
```

---

## 5. CONFIGURATION FILE FORMAT

### 5.1 JSON Configuration
```json
{
  "presets": {
    "company_default": {
      "font_family": "HelveticaTextNow-ExtraBold",
      "font_size": 100,
      "font_color": "white",
      "display_mode": "one_word",
      "position": "center",
      "outline_width": 0,
      "fade_in_duration": 0.0,
      "fade_out_duration": 0.0
    },
    "social_media_vertical": {
      "font_family": "HelveticaTextNow-ExtraBold",
      "font_size": 95,
      "font_color": "white",
      "display_mode": "one_word",
      "position": "center",
      "outline_color": "black",
      "outline_width": 2,
      "uppercase": true
    }
  }
}
```

---

## 6. KEY FEATURES

1. **Modular Design**: Easy to swap components and add new features
2. **Preset System**: Quick switching between different caption styles
3. **Perfect Sync**: Maintains frame-perfect synchronization regardless of style
4. **GPU Acceleration**: Optional GPU rendering for performance
5. **Extensible**: Easy to add new display modes or rendering engines
6. **Configuration**: Simple API for runtime configuration changes
7. **Validation**: Built-in validation to ensure captions display correctly

This system gives you complete control over caption appearance while maintaining the high-quality synchronization from the advanced sync engine.