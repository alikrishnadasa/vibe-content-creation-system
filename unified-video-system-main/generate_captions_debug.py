#!/usr/bin/env python3
"""
Debug Caption Generation
Generate captions directly from Whisper transcription to test the system
"""

import logging
import json
from pathlib import Path
from captions.whisper_transcriber import WhisperTranscriber
from captions.unified_caption_engine import UnifiedCaptionEngine, WordTiming, CaptionSyncEngine
from captions.preset_manager import CaptionDisplayMode

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_captions_for_script(script_name: str = "anxiety1"):
    """Generate captions for a specific script"""
    
    # Initialize components
    transcriber = WhisperTranscriber()
    caption_sync = CaptionSyncEngine()
    
    # Get Whisper transcription
    if script_name not in transcriber.transcription_cache:
        logger.error(f"No transcription found for {script_name}")
        return
    
    transcription = transcriber.transcription_cache[script_name]
    logger.info(f"Found transcription with {len(transcription.words)} words")
    
    # Convert Whisper words to WordTiming objects
    word_timings = []
    for whisper_word in transcription.words:
        word_timing = WordTiming(
            word=whisper_word.word,
            start=whisper_word.start,
            end=whisper_word.end,
            confidence=whisper_word.confidence
        )
        word_timings.append(word_timing)
    
    logger.info(f"Converted to {len(word_timings)} word timings")
    
    # Generate different caption styles
    styles = {
        'one_word': CaptionDisplayMode.ONE_WORD,
        'two_words': CaptionDisplayMode.TWO_WORDS,
        'karaoke': CaptionDisplayMode.KARAOKE,
        'phrase': CaptionDisplayMode.PHRASE_BASED,
        'sentence': CaptionDisplayMode.FULL_SENTENCE
    }
    
    for style_name, display_mode in styles.items():
        logger.info(f"\nGenerating {style_name} captions...")
        
        try:
            caption_segments = caption_sync.create_caption_segments(
                word_timings=word_timings,
                display_mode=display_mode
            )
            
            logger.info(f"Generated {len(caption_segments)} {style_name} caption segments")
            
            # Show first few captions
            for i, segment in enumerate(caption_segments[:5]):
                logger.info(f"  {i+1}: '{segment['text']}' ({segment['start_time']:.1f}s - {segment['end_time']:.1f}s)")
            
            # Save captions as SRT for testing
            srt_content = ""
            for i, segment in enumerate(caption_segments):
                start_srt = f"{int(segment['start_time']//3600):02d}:{int((segment['start_time']%3600)//60):02d}:{int(segment['start_time']%60):02d},000"
                end_srt = f"{int(segment['end_time']//3600):02d}:{int((segment['end_time']%3600)//60):02d}:{int(segment['end_time']%60):02d},000"
                srt_content += f"{i+1}\n{start_srt} --> {end_srt}\n{segment['text']}\n\n"
            
            srt_file = f"debug_{script_name}_{style_name}.srt"
            with open(srt_file, 'w') as f:
                f.write(srt_content)
            logger.info(f"Saved SRT: {srt_file}")
            
        except Exception as e:
            logger.error(f"Failed to generate {style_name} captions: {e}")

if __name__ == "__main__":
    generate_captions_for_script("anxiety1")