"""
Beat Sync Engine for musical synchronization.

This engine provides:
- Beat detection using LibrosaBeatDetector (Python 3.13 compatible)
- Tempo estimation
- Downbeat detection
- Musical phrase analysis
- Caption-to-beat alignment
- Visual effect synchronization
"""

import asyncio
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from datetime import datetime

# Import our Python 3.13 compatible beat detector
from .librosa_beat_detection import LibrosaBeatDetector

logger = logging.getLogger(__name__)


@dataclass
class BeatInfo:
    """Information about a detected beat."""
    time: float  # Time in seconds
    strength: float  # Beat strength (0-1)
    is_downbeat: bool  # Whether this is a downbeat
    measure_position: int  # Position within measure (1-4 for 4/4 time)
    tempo: float  # Current tempo at this beat


@dataclass
class MusicalPhrase:
    """Represents a musical phrase or section."""
    start_time: float
    end_time: float
    phrase_type: str  # 'intro', 'verse', 'chorus', 'bridge', 'outro'
    energy_level: float  # 0-1, energy/intensity of phrase
    beat_indices: List[int]  # Indices of beats in this phrase


@dataclass
class BeatSyncResult:
    """Complete beat synchronization results."""
    beats: List[BeatInfo]
    tempo: float  # Average tempo in BPM
    time_signature: Tuple[int, int]  # e.g., (4, 4)
    phrases: List[MusicalPhrase]
    onset_times: List[float]  # Note onset times
    energy_curve: np.ndarray  # Energy over time
    processing_time: float


class BeatSyncEngine:
    """
    Advanced beat synchronization engine using LibrosaBeatDetector for music analysis.
    Python 3.13 compatible implementation.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the beat sync engine."""
        self.config = config or {}
        
        # Configuration
        self.min_tempo = self.config.get('min_tempo', 60)
        self.max_tempo = self.config.get('max_tempo', 180)
        self.beat_threshold = self.config.get('beat_threshold', 0.3)
        self.phrase_min_duration = self.config.get('phrase_min_duration', 4.0)
        self.sample_rate = self.config.get('sample_rate', 22050)
        self.hop_length = self.config.get('hop_length', 512)
        
        # Initialize LibrosaBeatDetector
        self.beat_detector = LibrosaBeatDetector(
            sr=self.sample_rate,
            hop_length=self.hop_length
        )
        
        # Cache
        self._cache = {}
        
        # Statistics
        self.stats = {
            'files_processed': 0,
            'total_beats_detected': 0,
            'average_processing_time': 0,
            'cache_hits': 0
        }
        
        logger.info("BeatSyncEngine initialized with LibrosaBeatDetector")
    
    async def analyze_audio(self, audio_path: Path) -> BeatSyncResult:
        """
        Analyze audio file for beat synchronization.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            BeatSyncResult with complete analysis
        """
        start_time = datetime.now()
        
        # Check cache
        cache_key = str(audio_path)
        if cache_key in self._cache:
            self.stats['cache_hits'] += 1
            return self._cache[cache_key]
        
        # Load audio
        audio_data, sr = await self._load_audio(audio_path)
        
        # Detect beats
        beats = await self._detect_beats(audio_path, audio_data)
        
        # Estimate tempo
        tempo, time_signature = await self._estimate_tempo(audio_path, beats)
        
        # Detect phrases
        phrases = await self._detect_phrases(audio_data, beats)
        
        # Detect onsets
        onsets = await self._detect_onsets(audio_path)
        
        # Calculate energy curve
        energy = await self._calculate_energy(audio_data)
        
        # Create result
        processing_time = (datetime.now() - start_time).total_seconds()
        result = BeatSyncResult(
            beats=beats,
            tempo=tempo,
            time_signature=time_signature,
            phrases=phrases,
            onset_times=onsets,
            energy_curve=energy,
            processing_time=processing_time
        )
        
        # Update statistics
        self.stats['files_processed'] += 1
        self.stats['total_beats_detected'] += len(beats)
        self.stats['average_processing_time'] = (
            (self.stats['average_processing_time'] * (self.stats['files_processed'] - 1) + 
             processing_time) / self.stats['files_processed']
        )
        
        # Cache result
        self._cache[cache_key] = result
        
        return result
    
    async def _load_audio(self, audio_path: Path) -> Tuple[np.ndarray, int]:
        """Load audio file using librosa."""
        import librosa
        
        # Load audio with librosa
        y, sr = librosa.load(str(audio_path), sr=self.sample_rate)
        return y, sr
    
    async def _detect_beats(self, audio_path: Path, audio_data: np.ndarray) -> List[BeatInfo]:
        """Detect beats in audio using LibrosaBeatDetector."""
        # Use LibrosaBeatDetector to detect beats
        tempo, beat_times = self.beat_detector.detect_beats(str(audio_path), units='time')
        
        # Get beat strength for more detailed analysis
        beat_strength = self.beat_detector.get_beat_strength(str(audio_path))
        
        # Create BeatInfo objects
        beat_infos = []
        for i, beat_time in enumerate(beat_times):
            # Determine if this is a downbeat (every 4th beat in 4/4 time)
            is_downbeat = i % 4 == 0
            
            # Calculate measure position (1-4 for 4/4 time)
            measure_position = (i % 4) + 1
            
            # Get beat strength at this time
            strength_idx = int(beat_time * len(beat_strength) / (len(audio_data) / self.sample_rate))
            strength_idx = min(strength_idx, len(beat_strength) - 1)
            strength = float(beat_strength[strength_idx])
            
            # Normalize strength to 0-1 range
            if beat_strength.max() > 0:
                strength = strength / beat_strength.max()
            
            # Boost strength for downbeats
            if is_downbeat:
                strength = min(1.0, strength * 1.5)
            
            beat_infos.append(BeatInfo(
                time=float(beat_time),
                strength=strength,
                is_downbeat=is_downbeat,
                measure_position=measure_position,
                tempo=tempo
            ))
        
        return beat_infos
    
    async def _estimate_tempo(self, audio_path: Path, beats: List[BeatInfo]) -> Tuple[float, Tuple[int, int]]:
        """Estimate tempo and time signature."""
        # If we already have tempo from beat detection, use it
        if beats and len(beats) > 0:
            # Use the median tempo from all beats
            tempos = [beat.tempo for beat in beats if beat.tempo > 0]
            if tempos:
                tempo = np.median(tempos)
            else:
                # Calculate from beat intervals
                tempo = self.beat_detector.tempo_from_beats(
                    np.array([beat.time for beat in beats])
                )
        else:
            # Fallback tempo
            tempo = 120.0
        
        # For now, assume 4/4 time (can be enhanced later)
        time_signature = (4, 4)
        
        return tempo, time_signature
    
    async def _detect_phrases(self, audio_data: np.ndarray, 
                            beats: List[BeatInfo]) -> List[MusicalPhrase]:
        """Detect musical phrases."""
        if not beats:
            return []
        
        # Calculate energy for phrase detection
        energy = await self._calculate_energy(audio_data)
        
        # Simple phrase detection based on energy changes
        phrases = []
        current_start = 0.0
        current_energy = []
        beat_indices = []
        
        for i, beat in enumerate(beats):
            # Sample energy at beat time
            energy_idx = int(beat.time * len(energy) / (len(audio_data) / 44100.0))
            energy_idx = min(energy_idx, len(energy) - 1)
            current_energy.append(energy[energy_idx])
            beat_indices.append(i)
            
            # Check for phrase boundary (energy drop or 8 bars)
            if (len(current_energy) >= 16 or  # 16 beats = 4 bars in 4/4
                (len(current_energy) > 4 and 
                 current_energy[-1] < np.mean(current_energy[:-1]) * 0.7)):
                
                # Create phrase
                avg_energy = np.mean(current_energy)
                phrase_type = self._classify_phrase(avg_energy, len(phrases))
                
                phrases.append(MusicalPhrase(
                    start_time=current_start,
                    end_time=beat.time,
                    phrase_type=phrase_type,
                    energy_level=avg_energy,
                    beat_indices=beat_indices.copy()
                ))
                
                # Reset for next phrase
                current_start = beat.time
                current_energy = []
                beat_indices = []
        
        # Add final phrase
        if current_energy:
            phrases.append(MusicalPhrase(
                start_time=current_start,
                end_time=beats[-1].time,
                phrase_type='outro',
                energy_level=np.mean(current_energy),
                beat_indices=beat_indices
            ))
        
        return phrases
    
    def _classify_phrase(self, energy: float, phrase_index: int) -> str:
        """Classify phrase type based on energy and position."""
        if phrase_index == 0:
            return 'intro'
        elif energy > 0.8:
            return 'chorus'
        elif energy > 0.6:
            return 'verse'
        elif energy > 0.4:
            return 'bridge'
        else:
            return 'outro'
    
    async def _detect_onsets(self, audio_path: Path) -> List[float]:
        """Detect note onsets using LibrosaBeatDetector."""
        # Use LibrosaBeatDetector to detect onsets
        onset_times, _ = self.beat_detector.detect_onset_beats(
            str(audio_path), 
            threshold=self.beat_threshold
        )
        
        return list(onset_times)
    
    async def _calculate_energy(self, audio_data: np.ndarray) -> np.ndarray:
        """Calculate energy curve of audio."""
        # Simple RMS energy calculation
        window_size = 2048
        hop_size = 512
        
        energy = []
        for i in range(0, len(audio_data) - window_size, hop_size):
            window = audio_data[i:i + window_size]
            rms = np.sqrt(np.mean(window ** 2))
            energy.append(rms)
        
        energy = np.array(energy)
        
        # Normalize
        if energy.max() > 0:
            energy = energy / energy.max()
        
        return energy
    
    def align_captions_to_beats(self, captions: List[Dict], 
                               beat_result: BeatSyncResult) -> List[Dict]:
        """
        Align caption timings to musical beats.
        
        Args:
            captions: List of caption dictionaries with 'start' and 'end' times
            beat_result: Beat analysis results
            
        Returns:
            Updated captions aligned to beats
        """
        aligned_captions = []
        
        for caption in captions:
            # Find nearest beat to start time
            start_beat_idx = self._find_nearest_beat(
                caption['start'], 
                beat_result.beats
            )
            
            # Align to beat
            if start_beat_idx is not None:
                aligned_caption = caption.copy()
                aligned_caption['start'] = beat_result.beats[start_beat_idx].time
                aligned_caption['beat_aligned'] = True
                aligned_caption['beat_strength'] = beat_result.beats[start_beat_idx].strength
                aligned_captions.append(aligned_caption)
            else:
                aligned_captions.append(caption)
        
        return aligned_captions
    
    def _find_nearest_beat(self, time: float, beats: List[BeatInfo]) -> Optional[int]:
        """Find index of nearest beat to given time."""
        if not beats:
            return None
        
        min_diff = float('inf')
        nearest_idx = None
        
        for i, beat in enumerate(beats):
            diff = abs(beat.time - time)
            if diff < min_diff:
                min_diff = diff
                nearest_idx = i
        
        # Only align if within threshold
        if min_diff < 0.1:  # 100ms threshold
            return nearest_idx
        
        return None
    
    def get_visual_effects_timing(self, beat_result: BeatSyncResult) -> List[Dict]:
        """
        Generate timing for visual effects based on beat analysis.
        
        Returns:
            List of effect timing dictionaries
        """
        effects = []
        
        # Add effects on strong beats
        for beat in beat_result.beats:
            if beat.is_downbeat:
                effects.append({
                    'time': beat.time,
                    'effect': 'flash',
                    'duration': 0.1,
                    'intensity': 1.0
                })
            elif beat.strength > 0.8:
                effects.append({
                    'time': beat.time,
                    'effect': 'pulse',
                    'duration': 0.2,
                    'intensity': beat.strength
                })
        
        # Add phrase transitions
        for i in range(len(beat_result.phrases) - 1):
            transition_time = beat_result.phrases[i].end_time
            effects.append({
                'time': transition_time,
                'effect': 'transition',
                'duration': 1.0,
                'from_phrase': beat_result.phrases[i].phrase_type,
                'to_phrase': beat_result.phrases[i + 1].phrase_type
            })
        
        return effects
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get engine statistics."""
        return {
            **self.stats,
            'backend': 'LibrosaBeatDetector',
            'python_version': 'Python 3.13 compatible',
            'cache_size': len(self._cache),
            'supported_features': {
                'beat_detection': True,
                'tempo_estimation': True,
                'downbeat_detection': True,
                'phrase_detection': True,
                'onset_detection': True,
                'energy_analysis': True,
                'audio_formats': ['mp3', 'wav', 'flac', 'ogg', 'm4a']
            }
        }


# Example usage and testing
if __name__ == "__main__":
    async def test_beat_sync():
        """Test the beat sync engine."""
        from rich.console import Console
        from rich.table import Table
        
        console = Console()
        
        # Create engine
        engine = BeatSyncEngine({
            'min_tempo': 80,
            'max_tempo': 160
        })
        
        console.print("[bold green]Beat Sync Engine Test[/bold green]")
        console.print(f"Backend: LibrosaBeatDetector (Python 3.13 compatible)")
        
        # Test with dummy audio
        result = await engine.analyze_audio(Path("test_audio.mp3"))
        
        # Display results
        table = Table(title="Beat Analysis Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Tempo (BPM)", f"{result.tempo:.1f}")
        table.add_row("Time Signature", f"{result.time_signature[0]}/{result.time_signature[1]}")
        table.add_row("Total Beats", str(len(result.beats)))
        table.add_row("Downbeats", str(sum(1 for b in result.beats if b.is_downbeat)))
        table.add_row("Phrases", str(len(result.phrases)))
        table.add_row("Onsets", str(len(result.onset_times)))
        table.add_row("Processing Time", f"{result.processing_time:.3f}s")
        
        console.print(table)
        
        # Show phrase breakdown
        phrase_table = Table(title="Musical Phrases")
        phrase_table.add_column("Type", style="cyan")
        phrase_table.add_column("Start", style="yellow")
        phrase_table.add_column("Duration", style="green")
        phrase_table.add_column("Energy", style="magenta")
        
        for phrase in result.phrases:
            phrase_table.add_row(
                phrase.phrase_type,
                f"{phrase.start_time:.2f}s",
                f"{phrase.end_time - phrase.start_time:.2f}s",
                f"{phrase.energy_level:.2f}"
            )
        
        console.print(phrase_table)
        
        # Test caption alignment
        test_captions = [
            {'text': 'Hello', 'start': 0.5, 'end': 1.0},
            {'text': 'World', 'start': 1.2, 'end': 1.8},
            {'text': 'Music', 'start': 2.1, 'end': 2.7}
        ]
        
        aligned = engine.align_captions_to_beats(test_captions, result)
        
        console.print("\n[bold]Caption Alignment Test[/bold]")
        for i, (orig, aligned) in enumerate(zip(test_captions, aligned)):
            if aligned.get('beat_aligned'):
                console.print(f"Caption {i}: {orig['start']:.2f}s → {aligned['start']:.2f}s "
                            f"(strength: {aligned['beat_strength']:.2f})")
        
        # Show statistics
        stats = engine.get_statistics()
        console.print(f"\n[bold]Engine Statistics:[/bold]")
        console.print(f"Files processed: {stats['files_processed']}")
        console.print(f"Cache hits: {stats['cache_hits']}")
        console.print(f"Average processing time: {stats['average_processing_time']:.3f}s")
    
    # Run test
    asyncio.run(test_beat_sync())