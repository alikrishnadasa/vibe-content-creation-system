#!/usr/bin/env python3
"""
Script Caption Cache
Pregenerate and cache word-by-word captions for all 11 audio scripts
"""

import json
import logging
import asyncio
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

# Local imports
try:
    from .unified_caption_engine import UnifiedCaptionEngine, WordTiming, Caption
    from .preset_manager import CaptionDisplayMode
    from .whisper_transcriber import WhisperTranscriber
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).parent))
    from unified_caption_engine import UnifiedCaptionEngine, WordTiming, Caption
    from preset_manager import CaptionDisplayMode
    from whisper_transcriber import WhisperTranscriber

logger = logging.getLogger(__name__)

@dataclass 
class CachedScriptCaptions:
    """Cached caption data for a script"""
    script_name: str
    audio_duration: float
    word_timings: List[Dict[str, Any]]  # Serialized WordTiming objects
    one_word_segments: List[Dict[str, Any]]
    two_word_segments: List[Dict[str, Any]]
    phrase_segments: List[Dict[str, Any]]
    sentence_segments: List[Dict[str, Any]]
    karaoke_segments: List[Dict[str, Any]]
    generated_timestamp: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CachedScriptCaptions':
        """Create from dictionary"""
        return cls(**data)

class ScriptCaptionCache:
    """Manages pregenerated caption data for all scripts"""
    
    def __init__(self, cache_file: str = "cache/script_captions.json", use_whisper: bool = True):
        self.cache_file = Path(cache_file)
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        self.cache: Dict[str, CachedScriptCaptions] = {}
        self.caption_engine = UnifiedCaptionEngine()
        
        # Initialize Whisper transcriber if requested
        self.use_whisper = use_whisper
        if use_whisper:
            try:
                self.whisper_transcriber = WhisperTranscriber(model_size="base")
                logger.info("🧠 Whisper transcriber initialized for precise word timing")
            except Exception as e:
                logger.warning(f"Whisper initialization failed: {e}")
                logger.info("📝 Falling back to simple timing distribution")
                self.use_whisper = False
                self.whisper_transcriber = None
        else:
            self.whisper_transcriber = None
        
        # Load existing cache
        self._load_cache()
    
    def _load_cache(self):
        """Load caption cache from disk"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    cache_data = json.load(f)
                
                for script_name, data in cache_data.items():
                    self.cache[script_name] = CachedScriptCaptions.from_dict(data)
                
                logger.info(f"📚 Loaded caption cache for {len(self.cache)} scripts")
            except Exception as e:
                logger.warning(f"Could not load caption cache: {e}")
    
    def _save_cache(self):
        """Save caption cache to disk"""
        try:
            cache_data = {
                script_name: cached.to_dict() 
                for script_name, cached in self.cache.items()
            }
            
            with open(self.cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
            
            logger.info(f"💾 Saved caption cache for {len(self.cache)} scripts")
        except Exception as e:
            logger.error(f"Failed to save caption cache: {e}")
    
    def get_audio_duration(self, audio_path: str) -> float:
        """Get audio duration using librosa or fallback"""
        try:
            import librosa
            duration = librosa.get_duration(path=audio_path)
            return duration
        except ImportError:
            logger.warning("librosa not available, using file size estimate")
            file_size = Path(audio_path).stat().st_size
            # Rough estimate: ~1MB per minute for compressed audio
            return (file_size / 1024 / 1024) * 60
        except Exception as e:
            logger.error(f"Could not get audio duration for {audio_path}: {e}")
            return 60.0  # Default fallback
    
    def extract_text_from_script_name(self, script_name: str) -> str:
        """Extract meaningful text from script filename"""
        # Map script names to actual text content
        script_text_map = {
            'anxiety1': "When anxiety overwhelms you, remember that this feeling is temporary. You have the strength to overcome this moment and find peace within yourself.",
            'safe1': "You are safe. You are loved. You are exactly where you need to be in this moment. Trust in your journey and embrace your inner strength.",
            'miserable1': "Even in the darkest moments, there is light within you waiting to shine. You are stronger than you know and worthy of happiness.",
            'before': "Before you make that decision, take a breath. Consider the possibilities that await you. Your future self is counting on this moment.",
            'adhd': "Your mind works differently, and that's your superpower. Embrace your unique way of thinking and channel your energy into greatness.",
            'deadinside': "Feeling numb doesn't mean you're broken. It means you're protecting yourself while you heal. Be patient with your process.",
            'diewithphone': "Put down the phone. Look up. The real world is waiting for your presence. Authentic connections matter more than digital validation.",
            'phone1': "Your worth isn't measured by likes or follows. You are valuable just as you are. Real relationships happen beyond the screen.",
            '4': "Four simple words can change everything: You are not alone. In your struggles and triumphs, remember that connection heals.",
            '6': "Six breaths. In and out. Feel yourself returning to the present moment. Peace is always available to you right here, right now.",
            '500friends': "Five hundred friends online, but do you feel truly connected? Quality over quantity in relationships. One real friend is worth more than a thousand followers."
        }
        
        return script_text_map.get(script_name, f"Inspirational message about {script_name}")
    
    async def generate_script_captions(self, script_name: str, audio_path: str, 
                                     force_regenerate: bool = False) -> CachedScriptCaptions:
        """Generate and cache caption data for a single script"""
        
        # Check if already cached
        if not force_regenerate and script_name in self.cache:
            logger.info(f"📋 Using cached captions for {script_name}")
            return self.cache[script_name]
        
        logger.info(f"🎯 Generating captions for {script_name}...")
        start_time = time.time()
        
        # Generate word timings using Whisper or fallback
        if self.use_whisper and self.whisper_transcriber:
            logger.info(f"🧠 Using Whisper for precise word timing...")
            
            # Get Whisper transcription with word-level timing
            transcription = self.whisper_transcriber.transcribe_audio(audio_path)
            
            logger.info(f"   Whisper text: '{transcription.text[:50]}...'")
            logger.info(f"   Audio duration: {transcription.duration:.1f}s")
            logger.info(f"   Words detected: {len(transcription.words)}")
            
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
            
            audio_duration = transcription.duration
            
        else:
            logger.info(f"📝 Using simple timing distribution...")
            
            # Get audio duration
            audio_duration = self.get_audio_duration(audio_path)
            logger.info(f"   Audio duration: {audio_duration:.1f}s")
            
            # Get script text
            script_text = self.extract_text_from_script_name(script_name)
            logger.info(f"   Script text: '{script_text[:50]}...'")
            
            # Generate word timings using the caption engine
            word_timings = self.caption_engine.sync_engine.generate_word_timings(
                script_text, audio_duration
            )
        
        # Generate all display mode segments
        one_word_segments = self.caption_engine.sync_engine.create_caption_segments(
            word_timings, CaptionDisplayMode.ONE_WORD
        )
        
        two_word_segments = self.caption_engine.sync_engine.create_caption_segments(
            word_timings, CaptionDisplayMode.TWO_WORDS
        )
        
        phrase_segments = self.caption_engine.sync_engine.create_caption_segments(
            word_timings, CaptionDisplayMode.PHRASE_BASED
        )
        
        sentence_segments = self.caption_engine.sync_engine.create_caption_segments(
            word_timings, CaptionDisplayMode.FULL_SENTENCE
        )
        
        karaoke_segments = self.caption_engine.sync_engine.create_caption_segments(
            word_timings, CaptionDisplayMode.KARAOKE
        )
        
        # Create cached object
        cached_captions = CachedScriptCaptions(
            script_name=script_name,
            audio_duration=audio_duration,
            word_timings=[{
                'word': w.word,
                'start': w.start,
                'end': w.end,
                'confidence': w.confidence
            } for w in word_timings],
            one_word_segments=one_word_segments,
            two_word_segments=two_word_segments,
            phrase_segments=phrase_segments,
            sentence_segments=sentence_segments,
            karaoke_segments=karaoke_segments,
            generated_timestamp=time.time()
        )
        
        # Cache it
        self.cache[script_name] = cached_captions
        
        generation_time = time.time() - start_time
        logger.info(f"✅ Generated captions for {script_name} in {generation_time:.3f}s")
        logger.info(f"   Words: {len(word_timings)}, One-word segments: {len(one_word_segments)}")
        
        return cached_captions
    
    async def generate_all_script_captions(self, scripts_directory: str) -> bool:
        """Generate captions for all 11 scripts"""
        
        scripts_dir = Path(scripts_directory)
        if not scripts_dir.exists():
            logger.error(f"Scripts directory not found: {scripts_directory}")
            return False
        
        # Find all .wav files
        audio_files = list(scripts_dir.glob("*.wav"))
        logger.info(f"🎵 Found {len(audio_files)} audio scripts")
        
        if len(audio_files) == 0:
            logger.error("No .wav files found in scripts directory")
            return False
        
        success_count = 0
        start_time = time.time()
        
        # Process each script
        for audio_file in audio_files:
            script_name = audio_file.stem  # filename without extension
            
            try:
                await self.generate_script_captions(script_name, str(audio_file))
                success_count += 1
            except Exception as e:
                logger.error(f"Failed to generate captions for {script_name}: {e}")
        
        # Save cache
        self._save_cache()
        
        total_time = time.time() - start_time
        logger.info("=" * 60)
        logger.info("📊 CAPTION GENERATION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"✅ Processed: {success_count}/{len(audio_files)} scripts")
        logger.info(f"⚡ Total time: {total_time:.2f}s")
        logger.info(f"📈 Average per script: {total_time/len(audio_files):.3f}s")
        logger.info(f"💾 Cache saved to: {self.cache_file}")
        logger.info("=" * 60)
        
        return success_count == len(audio_files)
    
    def get_captions_for_script(self, script_name: str, 
                               display_mode: CaptionDisplayMode = CaptionDisplayMode.ONE_WORD) -> List[Dict[str, Any]]:
        """Get cached caption segments for a script"""
        
        if script_name not in self.cache:
            logger.warning(f"No cached captions found for {script_name}")
            return []
        
        cached = self.cache[script_name]
        
        # Return appropriate segments based on display mode
        if display_mode == CaptionDisplayMode.ONE_WORD:
            return cached.one_word_segments
        elif display_mode == CaptionDisplayMode.TWO_WORDS:
            return cached.two_word_segments
        elif display_mode == CaptionDisplayMode.PHRASE_BASED:
            return cached.phrase_segments
        elif display_mode == CaptionDisplayMode.FULL_SENTENCE:
            return cached.sentence_segments
        elif display_mode == CaptionDisplayMode.KARAOKE:
            return cached.karaoke_segments
        else:
            return cached.one_word_segments  # Default
    
    def get_cache_status(self) -> Dict[str, Any]:
        """Get status of caption cache"""
        return {
            'cached_scripts': list(self.cache.keys()),
            'total_cached': len(self.cache),
            'cache_file': str(self.cache_file),
            'cache_file_exists': self.cache_file.exists(),
            'latest_generation': max(
                (cached.generated_timestamp for cached in self.cache.values()), 
                default=0
            ) if self.cache else 0
        }

# Main execution function
async def main():
    """Main function to generate all script captions"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    cache = ScriptCaptionCache()
    
    # Generate captions for all scripts
    scripts_directory = "../11-scripts-for-tiktok"
    success = await cache.generate_all_script_captions(scripts_directory)
    
    if success:
        print("\n🎉 SUCCESS! All script captions generated and cached!")
        print("📋 Caption cache ready for ultra-fast video generation")
        
        # Show cache status
        status = cache.get_cache_status()
        print(f"📊 Cached scripts: {status['total_cached']}")
        print(f"💾 Cache file: {status['cache_file']}")
    else:
        print("\n💥 FAILED! Some script captions could not be generated")
        return False
    
    return True

if __name__ == "__main__":
    success = asyncio.run(main())