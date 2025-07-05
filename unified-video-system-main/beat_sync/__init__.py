"""
Beat synchronization system for the Unified Video System.

This module provides beat detection and musical synchronization capabilities
for frame-perfect video-audio alignment.

Python 3.13 compatible implementation using LibrosaBeatDetector.
"""

from .beat_sync_engine import BeatSyncEngine
from .librosa_beat_detection import LibrosaBeatDetector

__all__ = ['BeatSyncEngine', 'LibrosaBeatDetector']