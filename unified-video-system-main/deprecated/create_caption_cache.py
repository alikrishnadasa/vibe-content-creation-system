#!/usr/bin/env python3
"""
Create Caption Cache Files
Generate proper caption cache files that the video system expects
"""

import json
import logging
from pathlib import Path
from captions.whisper_transcriber import WhisperTranscriber
from captions.unified_caption_engine import CaptionSyncEngine, WordTiming
from captions.preset_manager import CaptionDisplayMode

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_caption_cache_for_script(script_name: str = "anxiety1"):
    """Create caption cache files for a script"""
    
    # Initialize components
    transcriber = WhisperTranscriber()
    caption_sync = CaptionSyncEngine()
    
    # Get Whisper transcription
    if script_name not in transcriber.transcription_cache:
        logger.error(f"No transcription found for {script_name}")
        return False
    
    transcription = transcriber.transcription_cache[script_name]
    logger.info(f"Found transcription with {len(transcription.words)} words")
    
    # Convert to WordTiming objects
    word_timings = []
    for whisper_word in transcription.words:
        word_timing = WordTiming(
            word=whisper_word.word,
            start=whisper_word.start,
            end=whisper_word.end,
            confidence=whisper_word.confidence
        )
        word_timings.append(word_timing)
    
    # Define styles and their display modes
    styles = {
        'default': CaptionDisplayMode.ONE_WORD,
        'tiktok': CaptionDisplayMode.ONE_WORD,
        'youtube': CaptionDisplayMode.TWO_WORDS,
        'cinematic': CaptionDisplayMode.PHRASE_BASED,
        'minimal': CaptionDisplayMode.FULL_SENTENCE,
        'karaoke': CaptionDisplayMode.KARAOKE
    }
    
    # Create cache directory
    cache_dir = Path("cache/pregenerated_captions")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate cache for each style
    for style_name, display_mode in styles.items():
        logger.info(f"Generating {style_name} captions...")
        
        # Generate caption segments
        caption_segments = caption_sync.create_caption_segments(
            word_timings=word_timings,
            display_mode=display_mode
        )
        
        # Convert to the format expected by the video system
        captions_data = []
        for segment in caption_segments:
            caption_entry = {
                "text": segment['text'].upper() if style_name == 'tiktok' else segment['text'],
                "start_time": segment['start_time'],
                "end_time": segment['end_time'],
                "confidence": segment.get('confidence', 1.0),
                "style": style_name
            }
            captions_data.append(caption_entry)
        
        # Create the cache file format
        cache_data = {
            "script_name": script_name,
            "style": style_name,
            "total_captions": len(captions_data),
            "total_duration": transcription.duration,
            "captions": captions_data,
            "metadata": {
                "display_mode": display_mode.value,
                "word_count": len(word_timings),
                "generated_from": "whisper_transcription"
            }
        }
        
        # Save to cache file
        cache_filename = f"{script_name}_{style_name}_captions.json"
        cache_path = cache_dir / cache_filename
        
        with open(cache_path, 'w') as f:
            json.dump(cache_data, f, indent=2)
        
        logger.info(f"✅ Saved {len(captions_data)} {style_name} captions to {cache_filename}")
    
    return True

if __name__ == "__main__":
    success = create_caption_cache_for_script("anxiety1")
    if success:
        print("🎉 Caption cache files created successfully!")
    else:
        print("❌ Failed to create caption cache files")