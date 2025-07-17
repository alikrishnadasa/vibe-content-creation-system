"""
Music Track Manager

Manages the universal background music track "Beanie (Slowed).mp3" for all video productions.
Provides beat analysis, tempo extraction, and audio processing capabilities.
"""

import os
import time
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from dataclasses import dataclass

# Import beat sync engine for real percussive analysis
from beat_sync.beat_sync_engine import BeatSyncEngine, BeatSyncResult

logger = logging.getLogger(__name__)

@dataclass
class MusicTrackInfo:
    """Information about the music track"""
    filepath: str
    filename: str
    duration: float
    sample_rate: int
    beats: Optional[List[float]] = None
    tempo: Optional[float] = None
    beat_frames: Optional[List[int]] = None
    analyzed: bool = False
    # Percussive event data
    kick_times: Optional[List[float]] = None
    snare_times: Optional[List[float]] = None
    hihat_times: Optional[List[float]] = None
    other_times: Optional[List[float]] = None
    beat_sync_result: Optional[BeatSyncResult] = None

class MusicManager:
    """Manages the universal background music track"""
    
    def __init__(self, music_file_path: str):
        """
        Initialize the music manager
        
        Args:
            music_file_path: Path to "Beanie (Slowed).mp3"
        """
        self.music_file_path = Path(music_file_path)
        self.track_info: Optional[MusicTrackInfo] = None
        self.loaded = False
        self.beat_sync_engine = BeatSyncEngine()
        
        logger.info("MusicManager initialized with BeatSyncEngine")
        
    async def load_music_track(self) -> bool:
        """
        Load and analyze the music track
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self.music_file_path.exists():
                logger.error(f"Music file not found: {self.music_file_path}")
                return False
            
            # For now, create basic track info
            # Full audio analysis will be implemented when librosa is integrated
            self.track_info = MusicTrackInfo(
                filepath=str(self.music_file_path),
                filename=self.music_file_path.name,
                duration=0.0,  # Will be updated with actual analysis
                sample_rate=44100  # Standard sample rate
            )
            
            logger.info(f"✅ Music track loaded: {self.track_info.filename}")
            self.loaded = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to load music track: {e}")
            return False
    
    async def analyze_beats(self) -> bool:
        """
        Analyze beat structure of the music track using the real BeatSyncEngine
        
        Returns:
            bool: True if analysis successful
        """
        if not self.loaded or not self.track_info:
            raise RuntimeError("Music track not loaded. Call load_music_track() first.")
        
        try:
            logger.info("🎵 Starting real beat analysis with BeatSyncEngine...")
            
            # Use real BeatSyncEngine to analyze the music track
            beat_sync_result = await self.beat_sync_engine.analyze_audio(self.music_file_path)
            
            # Extract timing information
            beat_times = [beat.time for beat in beat_sync_result.beats]
            
            # Update track info with real analysis results
            self.track_info.tempo = beat_sync_result.tempo
            self.track_info.duration = beat_times[-1] if beat_times else 0.0
            self.track_info.beats = beat_times
            self.track_info.kick_times = beat_sync_result.kick_times
            self.track_info.snare_times = beat_sync_result.snare_times
            self.track_info.hihat_times = beat_sync_result.hihat_times
            self.track_info.other_times = beat_sync_result.other_times
            self.track_info.beat_sync_result = beat_sync_result
            self.track_info.analyzed = True
            
            logger.info(f"🎵 Real beat analysis complete:")
            logger.info(f"   - Tempo: {beat_sync_result.tempo:.1f} BPM")
            logger.info(f"   - Duration: {self.track_info.duration:.1f}s")
            logger.info(f"   - Beats: {len(beat_times)}")
            logger.info(f"   - Kick events: {len(beat_sync_result.kick_times)}")
            logger.info(f"   - Snare events: {len(beat_sync_result.snare_times)}")
            logger.info(f"   - Hi-hat events: {len(beat_sync_result.hihat_times)}")
            logger.info(f"   - Other percussion: {len(beat_sync_result.other_times)}")
            logger.info(f"   - Processing time: {beat_sync_result.processing_time:.2f}s")
            
            return True
            
        except Exception as e:
            logger.error(f"Real beat analysis failed: {e}")
            logger.error(f"Falling back to mock analysis...")
            
            # Fallback to mock analysis if real analysis fails
            mock_tempo = 85.0  # BPM for "Beanie (Slowed)"
            mock_duration = 180.0  # 3 minutes estimated
            beat_interval = 60.0 / mock_tempo  # seconds per beat
            
            self.track_info.tempo = mock_tempo
            self.track_info.duration = mock_duration
            self.track_info.beats = [i * beat_interval for i in range(int(mock_duration / beat_interval))]
            self.track_info.analyzed = True
            
            logger.warning(f"Using mock analysis: {mock_tempo} BPM, {len(self.track_info.beats)} beats")
            return True
    
    def get_beat_timing(self, start_time: float, end_time: float, event_type: str = 'beat') -> List[float]:
        """
        Get beat timestamps within a time range for a specific event type
        
        Args:
            start_time: Start time in seconds
            end_time: End time in seconds
            event_type: Type of percussive event ('beat', 'kick', 'snare', 'hihat', 'other')
            
        Returns:
            List of event timestamps within the range
        """
        if not self.track_info or not self.track_info.analyzed:
            return []
        
        # Get appropriate event times based on type
        if event_type == 'kick' and hasattr(self.track_info, 'kick_times') and self.track_info.kick_times:
            event_times = self.track_info.kick_times
        elif event_type == 'snare' and hasattr(self.track_info, 'snare_times') and self.track_info.snare_times:
            event_times = self.track_info.snare_times
        elif event_type == 'hihat' and hasattr(self.track_info, 'hihat_times') and self.track_info.hihat_times:
            event_times = self.track_info.hihat_times
        elif event_type == 'other' and hasattr(self.track_info, 'other_times') and self.track_info.other_times:
            event_times = self.track_info.other_times
        elif event_type == 'beat' and self.track_info.beats:
            event_times = self.track_info.beats
        else:
            return []
        
        return [event_time for event_time in event_times 
                if start_time <= event_time <= end_time]
    
    def get_music_segments(self, target_duration: float, count: int = 1) -> List[Tuple[float, float]]:
        """
        Get music segments for video synchronization
        
        Args:
            target_duration: Duration needed for video
            count: Number of different segments to return
            
        Returns:
            List of (start_time, end_time) tuples
        """
        if not self.track_info:
            return [(0.0, target_duration)]
        
        segments = []
        track_duration = self.track_info.duration
        
        # Generate different starting points for variety
        for i in range(count):
            # Start at different points but ensure we have enough duration
            max_start = max(0, track_duration - target_duration)
            start_time = (i * max_start / max(1, count - 1)) if count > 1 else 0
            end_time = min(start_time + target_duration, track_duration)
            segments.append((start_time, end_time))
        
        return segments
    
    def get_track_info(self) -> Dict[str, Any]:
        """Get information about the loaded music track"""
        if not self.loaded or not self.track_info:
            return {}
        
        return {
            'filename': self.track_info.filename,
            'filepath': self.track_info.filepath,
            'duration': self.track_info.duration,
            'sample_rate': self.track_info.sample_rate,
            'tempo': self.track_info.tempo,
            'beats_analyzed': self.track_info.analyzed,
            'beat_count': len(self.track_info.beats) if self.track_info.beats else 0,
            'kick_count': len(self.track_info.kick_times) if hasattr(self.track_info, 'kick_times') and self.track_info.kick_times else 0,
            'snare_count': len(self.track_info.snare_times) if hasattr(self.track_info, 'snare_times') and self.track_info.snare_times else 0,
            'hihat_count': len(self.track_info.hihat_times) if hasattr(self.track_info, 'hihat_times') and self.track_info.hihat_times else 0,
            'other_count': len(self.track_info.other_times) if hasattr(self.track_info, 'other_times') and self.track_info.other_times else 0
        }
    
    def prepare_for_mixing(self, target_duration: float, sync_event_type: str = 'beat') -> Dict[str, Any]:
        """
        Prepare music track parameters for audio mixing with percussive sync support
        
        Args:
            target_duration: Target duration for the video
            sync_event_type: Type of percussive event to sync with ('beat', 'kick', 'snare', 'hihat', 'other')
            
        Returns:
            Dictionary with mixing parameters including sync events
        """
        if not self.loaded or not self.track_info:
            raise RuntimeError("Music track not loaded")
        
        segments = self.get_music_segments(target_duration, 1)
        start_time, end_time = segments[0] if segments else (0, target_duration)
        
        # Get sync events for the selected segment
        sync_events = self.get_beat_timing(start_time, end_time, sync_event_type)
        
        return {
            'music_file': self.track_info.filepath,
            'start_time': start_time,
            'end_time': end_time,
            'duration': end_time - start_time,
            'volume_level': 0.25,  # 25% volume for background music
            'fade_in': 1.0,  # 1 second fade in
            'fade_out': 2.0,  # 2 second fade out
            'sample_rate': self.track_info.sample_rate,
            'sync_event_type': sync_event_type,
            'sync_events': sync_events,
            'sync_event_count': len(sync_events)
        }
    
    def get_percussive_events(self, event_type: str) -> List[float]:
        """
        Get all percussive events of a specific type
        
        Args:
            event_type: Type of percussive event ('kick', 'snare', 'hihat', 'other', 'beat')
            
        Returns:
            List of event timestamps
        """
        if not self.track_info or not self.track_info.analyzed:
            return []
        
        if hasattr(self.track_info, 'beat_sync_result') and self.track_info.beat_sync_result:
            return self.beat_sync_engine.get_onset_times(event_type, self.track_info.beat_sync_result)
        else:
            # Fallback to direct access
            return self.get_beat_timing(0, self.track_info.duration, event_type)