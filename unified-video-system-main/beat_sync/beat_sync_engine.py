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
    # Percussive event classifications
    kick_times: List[float]  # Kick drum onset times
    snare_times: List[float]  # Snare drum onset times
    hihat_times: List[float]  # Hi-hat onset times
    other_times: List[float]  # Other percussion onset times


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
        
        # Classify percussive events
        kick_times, snare_times, hihat_times, other_times = await self._classify_percussive_events(
            audio_path, audio_data, onsets
        )
        
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
            processing_time=processing_time,
            kick_times=kick_times,
            snare_times=snare_times,
            hihat_times=hihat_times,
            other_times=other_times
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
    
    async def _classify_percussive_events(self, audio_path: Path, audio_data: np.ndarray, 
                                        onset_times: List[float]) -> Tuple[List[float], List[float], List[float], List[float]]:
        """Classify onset events into kick, snare, hi-hat, and other categories."""
        import librosa
        
        kick_times = []
        snare_times = []
        hihat_times = []
        other_times = []
        
        if not onset_times:
            return kick_times, snare_times, hihat_times, other_times
        
        # Load audio with higher sample rate for better frequency analysis
        y, sr = librosa.load(str(audio_path), sr=44100)
        
        # Extract spectral features for each onset
        for onset_time in onset_times:
            # Get audio segment around onset (±50ms)
            onset_sample = int(onset_time * sr)
            window_size = int(0.05 * sr)  # 50ms window
            start_sample = max(0, onset_sample - window_size // 2)
            end_sample = min(len(y), onset_sample + window_size // 2)
            
            if end_sample <= start_sample:
                other_times.append(onset_time)
                continue
            
            segment = y[start_sample:end_sample]
            
            # Compute spectral features
            spectral_centroid = librosa.feature.spectral_centroid(y=segment, sr=sr)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=segment, sr=sr)[0]
            zero_crossing_rate = librosa.feature.zero_crossing_rate(segment)[0]
            
            # Compute frequency bands energy
            stft = librosa.stft(segment)
            magnitude = np.abs(stft)
            
            # Define frequency bands (in Hz) - EXPANDED RANGES for better detection
            freqs = librosa.fft_frequencies(sr=sr)
            
            # Low frequency energy (20-250 Hz) - expanded for kicks
            low_freq_mask = (freqs >= 20) & (freqs <= 250)
            low_energy = np.sum(magnitude[low_freq_mask, :]) if np.any(low_freq_mask) else 0
            
            # Mid frequency energy (150-2000 Hz) - expanded for snares
            mid_freq_mask = (freqs >= 150) & (freqs <= 2000)
            mid_energy = np.sum(magnitude[mid_freq_mask, :]) if np.any(mid_freq_mask) else 0
            
            # High frequency energy (1000+ Hz) - lowered threshold for hi-hats
            high_freq_mask = (freqs >= 1000)
            high_energy = np.sum(magnitude[high_freq_mask, :]) if np.any(high_freq_mask) else 0
            
            # Additional frequency bands for better classification
            # Very low (20-80 Hz) - deep kicks
            very_low_mask = (freqs >= 20) & (freqs <= 80)
            very_low_energy = np.sum(magnitude[very_low_mask, :]) if np.any(very_low_mask) else 0
            
            # Upper mid (800-4000 Hz) - snare harmonics and hi-hat fundamentals
            upper_mid_mask = (freqs >= 800) & (freqs <= 4000)
            upper_mid_energy = np.sum(magnitude[upper_mid_mask, :]) if np.any(upper_mid_mask) else 0
            
            # Very high (5000+ Hz) - hi-hat sizzle
            very_high_mask = (freqs >= 5000)
            very_high_energy = np.sum(magnitude[very_high_mask, :]) if np.any(very_high_mask) else 0
            
            # Total energy for normalization
            total_energy = np.sum(magnitude)
            
            if total_energy == 0:
                other_times.append(onset_time)
                continue
            
            # Normalize energies with expanded ranges
            low_ratio = low_energy / total_energy
            mid_ratio = mid_energy / total_energy
            high_ratio = high_energy / total_energy
            very_low_ratio = very_low_energy / total_energy
            upper_mid_ratio = upper_mid_energy / total_energy
            very_high_ratio = very_high_energy / total_energy
            
            # Classification based on spectral characteristics
            avg_centroid = np.mean(spectral_centroid)
            avg_rolloff = np.mean(spectral_rolloff)
            avg_zcr = np.mean(zero_crossing_rate)
            
            # EXPANDED CLASSIFICATION with lower thresholds and multiple criteria
            
            # Kick drum classification - EXPANDED
            if (
                # Primary: Strong low frequencies
                (low_ratio > 0.15 and avg_centroid < 300) or
                # Secondary: Very deep kicks
                (very_low_ratio > 0.25 and avg_centroid < 150) or
                # Tertiary: Low centroid with some low energy
                (low_ratio > 0.10 and avg_centroid < 200 and avg_rolloff < 1500)
            ):
                kick_times.append(onset_time)
            
            # Hi-hat classification - EXPANDED  
            elif (
                # Primary: High frequency content
                (high_ratio > 0.15 and avg_centroid > 1500) or
                # Secondary: Very high frequency sizzle
                (very_high_ratio > 0.10 and avg_zcr > 0.05) or
                # Tertiary: Upper mid content with high ZCR
                (upper_mid_ratio > 0.20 and avg_zcr > 0.08 and avg_centroid > 2000)
            ):
                hihat_times.append(onset_time)
            
            # Snare classification - EXPANDED
            elif (
                # Primary: Mid frequency content
                (mid_ratio > 0.15 and 150 <= avg_centroid <= 3000) or
                # Secondary: Upper mid with some noise
                (upper_mid_ratio > 0.25 and avg_rolloff > 800) or
                # Tertiary: Balanced frequency with good rolloff
                (mid_ratio > 0.10 and avg_rolloff > 1200 and avg_centroid > 200)
            ):
                snare_times.append(onset_time)
            
            else:
                other_times.append(onset_time)
        
        logger.info(f"Classified percussive events: {len(kick_times)} kicks, {len(snare_times)} snares, "
                   f"{len(hihat_times)} hi-hats, {len(other_times)} others")
        
        return kick_times, snare_times, hihat_times, other_times
    
    def get_onset_times(self, event_type: str, beat_result: BeatSyncResult) -> List[float]:
        """Get onset times for a specific percussive event type.
        
        Args:
            event_type: Type of percussive event ('kick', 'snare', 'hihat', 'other', 'beat')
            beat_result: Beat analysis results
            
        Returns:
            List of onset times for the specified event type
        """
        if event_type == 'kick':
            return beat_result.kick_times
        elif event_type == 'snare':
            return beat_result.snare_times
        elif event_type == 'hihat':
            return beat_result.hihat_times
        elif event_type == 'other':
            return beat_result.other_times
        elif event_type == 'beat':
            return [beat.time for beat in beat_result.beats]
        else:
            raise ValueError(f"Unknown event type: {event_type}. "
                           f"Valid types: 'kick', 'snare', 'hihat', 'other', 'beat'")
    
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
                'percussive_classification': True,
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