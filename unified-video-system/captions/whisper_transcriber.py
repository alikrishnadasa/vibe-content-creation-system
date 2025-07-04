#!/usr/bin/env python3
"""
Whisper Transcriber
Precise word-level transcription with timing using OpenAI Whisper
"""

import whisper
import logging
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class WhisperWord:
    """Word with precise timing from Whisper"""
    word: str
    start: float
    end: float
    confidence: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'word': self.word,
            'start': self.start,
            'end': self.end,
            'confidence': self.confidence
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WhisperWord':
        """Create from dictionary"""
        return cls(**data)

@dataclass
class WhisperTranscription:
    """Complete transcription result from Whisper"""
    text: str
    words: List[WhisperWord]
    language: str
    duration: float
    model_used: str
    processing_time: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'text': self.text,
            'words': [word.to_dict() for word in self.words],
            'language': self.language,
            'duration': self.duration,
            'model_used': self.model_used,
            'processing_time': self.processing_time
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WhisperTranscription':
        """Create from dictionary"""
        words = [WhisperWord.from_dict(w) for w in data['words']]
        return cls(
            text=data['text'],
            words=words,
            language=data['language'],
            duration=data['duration'],
            model_used=data['model_used'],
            processing_time=data['processing_time']
        )

class WhisperTranscriber:
    """Handles precise word-level transcription using Whisper"""
    
    def __init__(self, model_size: str = "base", cache_dir: str = "cache/whisper"):
        """
        Initialize Whisper transcriber
        
        Args:
            model_size: Whisper model size (tiny, base, small, medium, large)
            cache_dir: Directory to cache transcriptions
        """
        self.model_size = model_size
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load Whisper model
        logger.info(f"🔄 Loading Whisper {model_size} model...")
        start_time = time.time()
        self.model = whisper.load_model(model_size)
        load_time = time.time() - start_time
        logger.info(f"✅ Whisper {model_size} model loaded in {load_time:.2f}s")
        
        # Cache for transcriptions
        self.transcription_cache = {}
        self._load_cache()
    
    def _get_cache_path(self, audio_file: str) -> Path:
        """Get cache file path for audio file"""
        audio_path = Path(audio_file)
        cache_filename = f"{audio_path.stem}_{self.model_size}.json"
        return self.cache_dir / cache_filename
    
    def _load_cache(self):
        """Load existing transcription cache"""
        cache_files = list(self.cache_dir.glob(f"*_{self.model_size}.json"))
        
        for cache_file in cache_files:
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                    script_name = cache_file.stem.replace(f"_{self.model_size}", "")
                    self.transcription_cache[script_name] = WhisperTranscription.from_dict(data)
            except Exception as e:
                logger.warning(f"Could not load cache file {cache_file}: {e}")
        
        if self.transcription_cache:
            logger.info(f"📚 Loaded {len(self.transcription_cache)} cached transcriptions")
    
    def _save_transcription(self, audio_file: str, transcription: WhisperTranscription):
        """Save transcription to cache"""
        cache_path = self._get_cache_path(audio_file)
        
        try:
            with open(cache_path, 'w') as f:
                json.dump(transcription.to_dict(), f, indent=2)
            
            # Update memory cache
            script_name = Path(audio_file).stem
            self.transcription_cache[script_name] = transcription
            
            logger.info(f"💾 Saved transcription cache: {cache_path.name}")
        except Exception as e:
            logger.error(f"Failed to save transcription cache: {e}")
    
    def transcribe_audio(self, audio_file: str, force_retranscribe: bool = False) -> WhisperTranscription:
        """
        Transcribe audio file with word-level timing
        
        Args:
            audio_file: Path to audio file
            force_retranscribe: Force re-transcription even if cached
            
        Returns:
            WhisperTranscription with word-level timing
        """
        audio_path = Path(audio_file)
        script_name = audio_path.stem
        
        # Check cache first
        if not force_retranscribe and script_name in self.transcription_cache:
            logger.info(f"📋 Using cached Whisper transcription for {script_name}")
            return self.transcription_cache[script_name]
        
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_file}")
        
        logger.info(f"🎯 Transcribing {script_name} with Whisper {self.model_size}...")
        start_time = time.time()
        
        try:
            # Transcribe with word-level timestamps
            result = self.model.transcribe(
                str(audio_file),
                word_timestamps=True,
                verbose=False
            )
            
            processing_time = time.time() - start_time
            
            # Extract word-level data
            words = []
            if 'segments' in result:
                for segment in result['segments']:
                    if 'words' in segment:
                        for word_data in segment['words']:
                            # Clean word text
                            word_text = word_data['word'].strip()
                            if word_text:  # Skip empty words
                                words.append(WhisperWord(
                                    word=word_text,
                                    start=word_data['start'],
                                    end=word_data['end'],
                                    confidence=word_data.get('probability', 1.0)
                                ))
            
            # Get audio duration
            audio_duration = result.get('segments', [])
            if audio_duration:
                duration = max(seg['end'] for seg in audio_duration)
            else:
                duration = 0.0
            
            # Create transcription object
            transcription = WhisperTranscription(
                text=result['text'].strip(),
                words=words,
                language=result.get('language', 'en'),
                duration=duration,
                model_used=self.model_size,
                processing_time=processing_time
            )
            
            # Save to cache
            self._save_transcription(audio_file, transcription)
            
            logger.info(f"✅ Transcribed {script_name} in {processing_time:.2f}s")
            logger.info(f"   Text: '{transcription.text[:50]}...'")
            logger.info(f"   Words: {len(transcription.words)}")
            logger.info(f"   Duration: {transcription.duration:.1f}s")
            logger.info(f"   Language: {transcription.language}")
            
            return transcription
            
        except Exception as e:
            logger.error(f"Whisper transcription failed for {script_name}: {e}")
            raise
    
    def transcribe_all_scripts(self, scripts_directory: str, 
                              force_retranscribe: bool = False) -> Dict[str, WhisperTranscription]:
        """
        Transcribe all audio scripts in directory
        
        Args:
            scripts_directory: Directory containing .wav files
            force_retranscribe: Force re-transcription of all files
            
        Returns:
            Dictionary mapping script names to transcriptions
        """
        scripts_dir = Path(scripts_directory)
        if not scripts_dir.exists():
            raise FileNotFoundError(f"Scripts directory not found: {scripts_directory}")
        
        # Find all audio files
        audio_files = list(scripts_dir.glob("*.wav"))
        if not audio_files:
            raise ValueError(f"No .wav files found in {scripts_directory}")
        
        logger.info(f"🎵 Found {len(audio_files)} audio files to transcribe")
        
        transcriptions = {}
        total_start_time = time.time()
        
        for audio_file in audio_files:
            try:
                transcription = self.transcribe_audio(str(audio_file), force_retranscribe)
                transcriptions[audio_file.stem] = transcription
            except Exception as e:
                logger.error(f"Failed to transcribe {audio_file.name}: {e}")
        
        total_time = time.time() - total_start_time
        
        logger.info("=" * 60)
        logger.info("📊 WHISPER TRANSCRIPTION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"✅ Transcribed: {len(transcriptions)}/{len(audio_files)} files")
        logger.info(f"⚡ Total time: {total_time:.2f}s")
        logger.info(f"📈 Average per file: {total_time/len(audio_files):.2f}s")
        logger.info(f"🧠 Model used: {self.model_size}")
        logger.info("=" * 60)
        
        return transcriptions
    
    def get_word_timings(self, script_name: str) -> List[WhisperWord]:
        """Get word timings for a script"""
        if script_name in self.transcription_cache:
            return self.transcription_cache[script_name].words
        return []
    
    def get_transcription_text(self, script_name: str) -> str:
        """Get transcribed text for a script"""
        if script_name in self.transcription_cache:
            return self.transcription_cache[script_name].text
        return ""
    
    def get_cache_status(self) -> Dict[str, Any]:
        """Get transcription cache status"""
        return {
            'model_size': self.model_size,
            'cached_scripts': list(self.transcription_cache.keys()),
            'total_cached': len(self.transcription_cache),
            'cache_dir': str(self.cache_dir),
            'cache_files': len(list(self.cache_dir.glob(f"*_{self.model_size}.json")))
        }
    
    def compare_with_expected_text(self, script_name: str, expected_text: str) -> Dict[str, Any]:
        """Compare Whisper transcription with expected text"""
        if script_name not in self.transcription_cache:
            return {'error': f'No transcription found for {script_name}'}
        
        transcription = self.transcription_cache[script_name]
        whisper_text = transcription.text.lower().strip()
        expected_text = expected_text.lower().strip()
        
        # Simple word-level comparison
        whisper_words = whisper_text.split()
        expected_words = expected_text.split()
        
        # Calculate similarity
        matches = sum(1 for w1, w2 in zip(whisper_words, expected_words) if w1 == w2)
        total_words = max(len(whisper_words), len(expected_words))
        accuracy = matches / total_words if total_words > 0 else 0
        
        return {
            'script_name': script_name,
            'whisper_text': transcription.text,
            'expected_text': expected_text,
            'whisper_words': len(whisper_words),
            'expected_words': len(expected_words),
            'accuracy': accuracy,
            'word_count_match': len(whisper_words) == len(expected_words)
        }

# Test function
async def test_whisper_transcriber():
    """Test Whisper transcriber functionality"""
    print("🧪 Testing Whisper Transcriber...")
    
    transcriber = WhisperTranscriber(model_size="base")
    
    # Test with one file
    test_audio = "../11-scripts-for-tiktok/anxiety1.wav"
    if Path(test_audio).exists():
        transcription = transcriber.transcribe_audio(test_audio)
        
        print(f"✅ Transcription: '{transcription.text}'")
        print(f"   Words: {len(transcription.words)}")
        print(f"   Duration: {transcription.duration:.1f}s")
        print(f"   Processing time: {transcription.processing_time:.2f}s")
        
        # Show first few words with timing
        print("   First 5 words:")
        for i, word in enumerate(transcription.words[:5]):
            print(f"     {i+1}. '{word.word}' ({word.start:.2f}s - {word.end:.2f}s)")
    
    print("🎉 Whisper transcriber test complete!")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_whisper_transcriber())