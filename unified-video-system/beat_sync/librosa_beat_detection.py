"""
Alternative beat detection using librosa - compatible with Python 3.13
Replaces madmom functionality for the unified video system
"""

import librosa
import numpy as np
from typing import Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)


class LibrosaBeatDetector:
    """
    Beat detection using librosa - madmom alternative for Python 3.13
    """
    
    def __init__(self, 
                 sr: int = 22050,
                 hop_length: int = 512,
                 tempo_method: str = 'beat_track'):
        """
        Initialize the beat detector
        
        Args:
            sr: Sample rate for audio processing
            hop_length: Number of samples between successive frames
            tempo_method: Method for tempo detection ('beat_track' or 'tempo')
        """
        self.sr = sr
        self.hop_length = hop_length
        self.tempo_method = tempo_method
        
    def detect_beats(self, 
                    audio_path: str,
                    units: str = 'time') -> Tuple[float, np.ndarray]:
        """
        Detect beats in an audio file
        
        Args:
            audio_path: Path to audio file
            units: Return format ('time' for seconds, 'frames' for frame indices)
            
        Returns:
            Tuple of (tempo_bpm, beat_times)
        """
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=self.sr)
            logger.info(f"Loaded audio: {audio_path}, duration: {len(y)/sr:.2f}s")
            
            # Detect tempo and beats
            if self.tempo_method == 'beat_track':
                tempo, beats = librosa.beat.beat_track(
                    y=y, 
                    sr=sr, 
                    hop_length=self.hop_length,
                    units=units
                )
            else:
                # Alternative method using onset strength
                onset_envelope = librosa.onset.onset_strength(
                    y=y, sr=sr, hop_length=self.hop_length
                )
                tempo = librosa.feature.tempo(
                    onset_envelope=onset_envelope, 
                    sr=sr, 
                    hop_length=self.hop_length
                )[0]
                beats = librosa.beat.beat_track(
                    onset_envelope=onset_envelope,
                    sr=sr,
                    hop_length=self.hop_length,
                    units=units
                )[1]
            
            tempo_scalar = float(tempo) if np.isscalar(tempo) else float(tempo[0])
            logger.info(f"Detected tempo: {tempo_scalar:.1f} BPM, beats: {len(beats)}")
            return tempo_scalar, beats
            
        except Exception as e:
            logger.error(f"Error detecting beats: {e}")
            raise
    
    def detect_onset_beats(self, 
                          audio_path: str,
                          threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect onsets and beats separately for more precise control
        
        Args:
            audio_path: Path to audio file
            threshold: Threshold for onset detection
            
        Returns:
            Tuple of (onset_times, beat_times)
        """
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=self.sr)
            
            # Detect onsets
            onset_frames = librosa.onset.onset_detect(
                y=y, 
                sr=sr, 
                hop_length=self.hop_length,
                threshold=threshold,
                units='time'
            )
            
            # Detect beats
            tempo, beat_frames = librosa.beat.beat_track(
                y=y, 
                sr=sr, 
                hop_length=self.hop_length,
                units='time'
            )
            
            logger.info(f"Detected {len(onset_frames)} onsets, {len(beat_frames)} beats")
            return onset_frames, beat_frames
            
        except Exception as e:
            logger.error(f"Error detecting onsets/beats: {e}")
            raise
    
    def get_beat_strength(self, audio_path: str) -> np.ndarray:
        """
        Get beat strength/activation function similar to madmom
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Beat strength array
        """
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=self.sr)
            
            # Compute onset strength (similar to beat activation)
            onset_strength = librosa.onset.onset_strength(
                y=y, 
                sr=sr, 
                hop_length=self.hop_length
            )
            
            return onset_strength
            
        except Exception as e:
            logger.error(f"Error computing beat strength: {e}")
            raise
    
    def tempo_from_beats(self, beat_times: np.ndarray) -> float:
        """
        Calculate tempo from beat times
        
        Args:
            beat_times: Array of beat times in seconds
            
        Returns:
            Estimated tempo in BPM
        """
        if len(beat_times) < 2:
            return 120.0  # Default tempo
            
        # Calculate intervals between beats
        intervals = np.diff(beat_times)
        
        # Use median interval to avoid outliers
        median_interval = np.median(intervals)
        
        # Convert to BPM
        tempo = 60.0 / median_interval
        
        return tempo


def create_beat_detector(**kwargs) -> LibrosaBeatDetector:
    """
    Factory function to create a beat detector
    """
    return LibrosaBeatDetector(**kwargs)


# Example usage and testing
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
        detector = LibrosaBeatDetector()
        
        try:
            tempo, beats = detector.detect_beats(audio_file)
            print(f"Tempo: {tempo:.1f} BPM")
            print(f"Beat times: {beats}")
            print(f"Number of beats: {len(beats)}")
            
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("Usage: python librosa_beat_detection.py <audio_file>") 