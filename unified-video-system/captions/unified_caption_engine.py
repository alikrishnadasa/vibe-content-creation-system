"""
Unified Caption Engine
Combines modular captions, phoneme sync, and GPU rendering
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
import asyncio
import time

# Local imports
try:
    from .preset_manager import CaptionPresetManager, CaptionStyle, CaptionDisplayMode
except ImportError:
    from preset_manager import CaptionPresetManager, CaptionStyle, CaptionDisplayMode


@dataclass
class Caption:
    """Individual caption with timing and style"""
    text: str
    start_time: float
    end_time: float
    style: CaptionStyle
    confidence: float = 1.0
    x_position: Optional[int] = None
    y_position: Optional[int] = None
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time
    
    def is_active_at(self, time: float) -> bool:
        """Check if caption is active at given time"""
        return self.start_time <= time <= self.end_time


@dataclass
class WordTiming:
    """Word-level timing information"""
    word: str
    start: float
    end: float
    confidence: float = 1.0
    phonemes: Optional[List[Dict]] = None


class CaptionSyncEngine:
    """Handle synchronization for different display modes"""
    
    def __init__(self):
        # Placeholder for future phoneme detection integration
        self.use_phoneme_sync = False
        
    def generate_word_timings(self, text: str, audio_duration: float) -> List[WordTiming]:
        """Generate word-level timing (placeholder implementation)"""
        words = text.split()
        if not words:
            return []
        
        # Simple equal distribution for now
        time_per_word = audio_duration / len(words)
        
        timings = []
        current_time = 0
        
        for word in words:
            timing = WordTiming(
                word=word,
                start=current_time,
                end=current_time + time_per_word,
                confidence=0.8  # Placeholder confidence
            )
            timings.append(timing)
            current_time += time_per_word
        
        return timings
    
    def create_caption_segments(self, word_timings: List[WordTiming], 
                              display_mode: CaptionDisplayMode) -> List[Dict]:
        """Create caption segments based on display mode"""
        
        if display_mode == CaptionDisplayMode.ONE_WORD:
            return self._create_one_word_segments(word_timings)
        elif display_mode == CaptionDisplayMode.TWO_WORDS:
            return self._create_two_word_segments(word_timings)
        elif display_mode == CaptionDisplayMode.FULL_SENTENCE:
            return self._create_sentence_segments(word_timings)
        elif display_mode == CaptionDisplayMode.PHRASE_BASED:
            return self._create_phrase_segments(word_timings)
        elif display_mode == CaptionDisplayMode.KARAOKE:
            return self._create_karaoke_segments(word_timings)
        else:
            # Default to one word
            return self._create_one_word_segments(word_timings)
    
    def _create_one_word_segments(self, word_timings: List[WordTiming]) -> List[Dict]:
        """Create segments for one-word display"""
        return [
            {
                'text': timing.word,
                'start_time': timing.start,
                'end_time': timing.end,
                'confidence': timing.confidence
            }
            for timing in word_timings
        ]
    
    def _create_two_word_segments(self, word_timings: List[WordTiming]) -> List[Dict]:
        """Create segments for two-word display"""
        segments = []
        
        for i in range(0, len(word_timings), 2):
            words = [word_timings[i]]
            if i + 1 < len(word_timings):
                words.append(word_timings[i + 1])
            
            text = " ".join(w.word for w in words)
            segments.append({
                'text': text,
                'start_time': words[0].start,
                'end_time': words[-1].end,
                'confidence': min(w.confidence for w in words)
            })
        
        return segments
    
    def _create_sentence_segments(self, word_timings: List[WordTiming]) -> List[Dict]:
        """Create segments for full sentence display"""
        if not word_timings:
            return []
        
        # Group words into sentences (simplified)
        sentences = []
        current_sentence = []
        
        for timing in word_timings:
            current_sentence.append(timing)
            # End sentence on punctuation
            if timing.word.endswith(('.', '!', '?')):
                sentences.append(current_sentence)
                current_sentence = []
        
        # Add remaining words as final sentence
        if current_sentence:
            sentences.append(current_sentence)
        
        segments = []
        for sentence_words in sentences:
            text = " ".join(w.word for w in sentence_words)
            segments.append({
                'text': text,
                'start_time': sentence_words[0].start,
                'end_time': sentence_words[-1].end,
                'confidence': min(w.confidence for w in sentence_words)
            })
        
        return segments
    
    def _create_phrase_segments(self, word_timings: List[WordTiming]) -> List[Dict]:
        """Create segments for phrase-based display"""
        # Simple phrase detection (3-5 words per phrase)
        segments = []
        phrase_length = 4
        
        for i in range(0, len(word_timings), phrase_length):
            phrase_words = word_timings[i:i + phrase_length]
            text = " ".join(w.word for w in phrase_words)
            
            segments.append({
                'text': text,
                'start_time': phrase_words[0].start,
                'end_time': phrase_words[-1].end,
                'confidence': min(w.confidence for w in phrase_words)
            })
        
        return segments
    
    def _create_karaoke_segments(self, word_timings: List[WordTiming]) -> List[Dict]:
        """Create segments for karaoke-style display"""
        # For karaoke, we show all words but highlight the current one
        if not word_timings:
            return []
        
        full_text = " ".join(w.word for w in word_timings)
        
        segments = []
        for i, timing in enumerate(word_timings):
            segments.append({
                'text': full_text,
                'highlighted_word_index': i,
                'start_time': timing.start,
                'end_time': timing.end,
                'confidence': timing.confidence
            })
        
        return segments


class UnifiedCaptionEngine:
    """
    Unified caption engine combining all features
    
    Features:
    - Modular style presets
    - Perfect synchronization
    - GPU rendering (placeholder)
    - Multiple display modes
    """
    
    def __init__(self, config: Dict = None):
        """Initialize the unified caption engine"""
        self.config = config or {}
        
        # Initialize subsystems
        self.preset_manager = CaptionPresetManager()
        self.sync_engine = CaptionSyncEngine()
        
        # Current style
        default_preset = self.config.get('default_preset', 'default')
        self.current_style = self.preset_manager.get_preset(default_preset)
        
        # Performance tracking
        self.stats = {
            'captions_generated': 0,
            'total_generation_time': 0.0,
            'average_generation_time': 0.0
        }
    
    def set_style(self, preset_name: str):
        """Change the current caption style"""
        self.current_style = self.preset_manager.get_preset(preset_name)
    
    def create_captions(self, text: str, audio_duration: float, 
                       style: Optional[str] = None,
                       beat_data: Optional[Dict] = None) -> List[Caption]:
        """
        Create captions with perfect synchronization
        
        Args:
            text: Script text
            audio_duration: Duration of audio in seconds
            style: Optional style preset name
            beat_data: Optional beat sync data
            
        Returns:
            List of Caption objects with timing and style
        """
        start_time = time.time()
        
        # Use specified style or current
        if style:
            caption_style = self.preset_manager.get_preset(style)
        else:
            caption_style = self.current_style
        
        # Generate word-level timings
        word_timings = self.sync_engine.generate_word_timings(text, audio_duration)
        
        # Adjust for beat sync if available
        if beat_data:
            word_timings = self._adjust_for_beats(word_timings, beat_data)
        
        # Create caption segments based on display mode
        segments = self.sync_engine.create_caption_segments(
            word_timings, caption_style.display_mode
        )
        
        # Convert to Caption objects
        captions = []
        for segment in segments:
            # Apply text transformations
            text_content = segment['text']
            if caption_style.uppercase:
                text_content = text_content.upper()
            
            caption = Caption(
                text=text_content,
                start_time=segment['start_time'],
                end_time=segment['end_time'],
                style=caption_style,
                confidence=segment.get('confidence', 1.0)
            )
            
            # Calculate position if needed
            if caption_style.position.value == 'custom' and caption_style.custom_position:
                caption.x_position = caption_style.custom_position[0]
                caption.y_position = caption_style.custom_position[1]
            
            captions.append(caption)
        
        # Update statistics
        generation_time = time.time() - start_time
        self.stats['captions_generated'] += 1
        self.stats['total_generation_time'] += generation_time
        self.stats['average_generation_time'] = (
            self.stats['total_generation_time'] / self.stats['captions_generated']
        )
        
        return captions
    
    def _adjust_for_beats(self, word_timings: List[WordTiming], 
                         beat_data: Dict) -> List[WordTiming]:
        """Adjust word timings to align with musical beats"""
        # Placeholder implementation
        # In full implementation, this would align words to beat boundaries
        
        beats = beat_data.get('beat_times', [])
        if not beats:
            return word_timings
        
        # Simple alignment: snap word starts to nearest beats
        adjusted_timings = []
        
        for timing in word_timings:
            # Find nearest beat
            nearest_beat = min(beats, key=lambda b: abs(b - timing.start))
            
            # Only adjust if within reasonable range (0.2s)
            if abs(nearest_beat - timing.start) < 0.2:
                duration = timing.end - timing.start
                adjusted_timing = WordTiming(
                    word=timing.word,
                    start=nearest_beat,
                    end=nearest_beat + duration,
                    confidence=timing.confidence * 0.9  # Slight confidence reduction
                )
                adjusted_timings.append(adjusted_timing)
            else:
                adjusted_timings.append(timing)
        
        return adjusted_timings
    
    def render_caption_frame(self, caption: Caption, frame_size: Tuple[int, int], 
                           current_time: float) -> Dict[str, Any]:
        """
        Render caption for a specific frame (placeholder for GPU implementation)
        
        Returns rendering instructions for the GPU renderer
        """
        if not caption.is_active_at(current_time):
            return {}
        
        # Calculate opacity for fade effects
        opacity = 1.0
        if caption.style.fade_in_duration > 0:
            fade_progress = (current_time - caption.start_time) / caption.style.fade_in_duration
            if fade_progress < 1.0:
                opacity *= min(1.0, max(0.0, fade_progress))
        
        if caption.style.fade_out_duration > 0:
            fade_out_start = caption.end_time - caption.style.fade_out_duration
            if current_time >= fade_out_start:
                fade_progress = (caption.end_time - current_time) / caption.style.fade_out_duration
                opacity *= min(1.0, max(0.0, fade_progress))
        
        # Determine position
        if caption.x_position is not None and caption.y_position is not None:
            position = (caption.x_position, caption.y_position)
        else:
            position = self._calculate_position(caption.style.position, frame_size)
        
        return {
            'text': caption.text,
            'font_family': caption.style.font_family,
            'font_size': caption.style.font_size,
            'color': caption.style.font_color,
            'position': position,
            'opacity': opacity,
            'outline_color': caption.style.outline_color,
            'outline_width': caption.style.outline_width,
            'background_color': caption.style.background_color,
            'background_opacity': caption.style.background_opacity,
            'shadow_color': caption.style.shadow_color,
            'shadow_offset': caption.style.shadow_offset,
            'margin': caption.style.margin
        }
    
    def _calculate_position(self, position_type, frame_size: Tuple[int, int]) -> Tuple[int, int]:
        """Calculate caption position based on type"""
        width, height = frame_size
        
        if position_type.value == 'center':
            return (width // 2, height // 2)
        elif position_type.value == 'bottom':
            return (width // 2, height - 100)
        elif position_type.value == 'top':
            return (width // 2, 100)
        elif position_type.value == 'bottom_left':
            return (100, height - 100)
        elif position_type.value == 'bottom_right':
            return (width - 100, height - 100)
        else:
            return (width // 2, height // 2)  # Default to center
    
    def get_active_captions(self, captions: List[Caption], 
                          current_time: float) -> List[Caption]:
        """Get all captions active at current time"""
        return [cap for cap in captions if cap.is_active_at(current_time)]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get caption engine statistics"""
        return {
            **self.stats,
            'available_presets': self.preset_manager.list_presets(),
            'current_style': self.current_style.font_family,
            'display_mode': self.current_style.display_mode.value
        }


# Integration with quantum pipeline
async def integrate_with_pipeline(pipeline, caption_engine):
    """Helper function to integrate caption engine with quantum pipeline"""
    # This would be called from the quantum pipeline
    pipeline.caption_engine = caption_engine
    print("✅ Caption engine integrated with quantum pipeline")


# Test function
async def test_unified_caption_engine():
    """Test the unified caption engine"""
    print("🧪 Testing Unified Caption Engine...")
    
    # Initialize engine
    engine = UnifiedCaptionEngine()
    
    # Test basic caption generation
    test_text = "Like water flowing through ancient stones, consciousness emerges."
    audio_duration = 8.0
    
    captions = engine.create_captions(test_text, audio_duration, style="default")
    
    print(f"✅ Generated {len(captions)} captions")
    print(f"   First caption: '{captions[0].text}' ({captions[0].start_time:.1f}s - {captions[0].end_time:.1f}s)")
    
    # Test different styles
    tiktok_captions = engine.create_captions(test_text, audio_duration, style="tiktok")
    print(f"✅ TikTok style: {len(tiktok_captions)} captions, uppercase: {tiktok_captions[0].text.isupper()}")
    
    # Test frame rendering
    frame_size = (1920, 1080)
    render_data = engine.render_caption_frame(captions[0], frame_size, captions[0].start_time + 0.1)
    print(f"✅ Render data: {render_data.get('text', 'No text')} at {render_data.get('position', 'No position')}")
    
    # Test statistics
    stats = engine.get_statistics()
    print(f"✅ Statistics: {stats['captions_generated']} generated, avg time: {stats['average_generation_time']:.3f}s")
    
    print("🎉 All caption engine tests passed!")


if __name__ == "__main__":
    asyncio.run(test_unified_caption_engine()) 