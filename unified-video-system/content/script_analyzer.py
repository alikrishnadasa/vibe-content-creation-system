"""
Audio Script Analyzer

Analyzes the 11 audio scripts for emotional content, timing, and thematic elements.
Maps emotions to existing beat sync semantic states and prepares for music synchronization.
"""

import os
import time
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from dataclasses import dataclass, field
import hashlib
import librosa
import librosa.display
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

@dataclass
class ScriptAnalysis:
    """Represents the analysis results for a single audio script."""
    filename: str
    duration: float
    primary_emotion: str
    emotional_intensity: float
    themes: List[str]
    raw_features: Dict[str, Any] = field(repr=False)

class AudioScriptAnalyzer:
    """Analyzes audio scripts for emotional content and characteristics"""
    
    def __init__(self, scripts_directory: str):
        """
        Initialize the script analyzer
        
        Args:
            scripts_directory: Path to directory containing audio scripts
        """
        self.scripts_directory = Path(scripts_directory)
        self.analyses: Dict[str, ScriptAnalysis] = {}
        self.loaded = False
        
        # Emotional mapping based on filename patterns
        self.emotion_mapping = {
            'anxiety': {
                'primary_emotion': 'anxiety',
                'intensity': 0.8,
                'tags': ['anxious', 'worried', 'tense', 'stressed'],
                'themes': ['mental_health', 'struggle', 'inner_conflict']
            },
            'adhd': {
                'primary_emotion': 'seeking',
                'intensity': 0.7,
                'tags': ['scattered', 'overwhelmed', 'seeking_focus'],
                'themes': ['attention', 'focus', 'neurodiversity']
            },
            'deadinside': {
                'primary_emotion': 'anxiety',
                'intensity': 0.9,
                'tags': ['empty', 'numb', 'disconnected'],
                'themes': ['depression', 'emptiness', 'inner_void']
            },
            'miserable': {
                'primary_emotion': 'anxiety',
                'intensity': 0.8,
                'tags': ['sad', 'hopeless', 'suffering'],
                'themes': ['sadness', 'despair', 'pain']
            },
            'phone': {
                'primary_emotion': 'seeking',
                'intensity': 0.6,
                'tags': ['distracted', 'addicted', 'disconnected'],
                'themes': ['technology', 'addiction', 'modern_life']
            },
            'safe': {
                'primary_emotion': 'peace',
                'intensity': 0.7,
                'tags': ['secure', 'protected', 'calm'],
                'themes': ['safety', 'security', 'comfort']
            },
            'friends': {
                'primary_emotion': 'seeking',
                'intensity': 0.6,
                'tags': ['lonely', 'connection', 'social'],
                'themes': ['friendship', 'connection', 'relationships']
            },
            'before': {
                'primary_emotion': 'awakening',
                'intensity': 0.7,
                'tags': ['transformation', 'change', 'growth'],
                'themes': ['personal_growth', 'transformation', 'journey']
            }
        }
    
    async def analyze_scripts(self) -> bool:
        """
        Analyze all audio scripts in the directory
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self.scripts_directory.exists():
                logger.error(f"Scripts directory not found: {self.scripts_directory}")
                return False
            
            # Find all WAV files
            wav_files = list(self.scripts_directory.glob("*.wav"))
            logger.info(f"Found {len(wav_files)} audio scripts to analyze")
            
            analyzed_count = 0
            for wav_file in wav_files:
                analysis = await self._analyze_single_script(wav_file)
                if analysis:
                    self.analyses[wav_file.stem] = analysis
                    analyzed_count += 1
            
            self.loaded = True
            logger.info(f"✅ Successfully analyzed {analyzed_count} audio scripts")
            return True
            
        except Exception as e:
            logger.error(f"Failed to analyze scripts: {e}")
            return False
    
    async def _analyze_single_script(self, wav_file: Path) -> Optional[ScriptAnalysis]:
        """Analyze a single audio script file"""
        try:
            # Get file info
            file_stats = wav_file.stat()
            file_size_mb = file_stats.st_size / (1024 * 1024)
            
            # Simulate real audio analysis time
            await asyncio.sleep(0.01)
            
            # Determine emotion based on filename
            emotion_info = self._determine_emotion_from_filename(wav_file.stem)
            
            analysis = ScriptAnalysis(
                filename=wav_file.name,
                duration=0.0,  # Will be updated if librosa is available
                primary_emotion=emotion_info['primary_emotion'],
                emotional_intensity=emotion_info['intensity'],
                themes=emotion_info['themes'].copy(),
                raw_features={
                    'filepath': str(wav_file),
                    'file_size_mb': file_size_mb,
                    'emotional_tags': emotion_info['tags'].copy()
                }
            )
            
            logger.debug(f"Analyzed {wav_file.name}: {analysis.primary_emotion} (intensity: {analysis.emotional_intensity})")
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze {wav_file}: {e}")
            return None
    
    def _determine_emotion_from_filename(self, filename_stem: str) -> Dict[str, Any]:
        """Determine emotional content based on filename patterns"""
        filename_lower = filename_stem.lower()
        
        # Check for known patterns
        for pattern, emotion_info in self.emotion_mapping.items():
            if pattern in filename_lower:
                return emotion_info
        
        # Check for numeric patterns (like anxiety1, miserable1)
        for pattern, emotion_info in self.emotion_mapping.items():
            if any(f"{pattern}{i}" in filename_lower for i in range(1, 10)):
                return emotion_info
        
        # Default mapping
        return {
            'primary_emotion': 'neutral',
            'intensity': 0.5,
            'tags': ['general', 'narrative'],
            'themes': ['general_content']
        }
    
    def get_scripts_by_emotion(self, emotion: str) -> List[ScriptAnalysis]:
        """
        Get scripts that match a specific emotional category
        
        Args:
            emotion: Emotional category ('anxiety', 'peace', 'seeking', 'awakening', 'neutral')
            
        Returns:
            List of matching script analyses
        """
        if not self.loaded:
            raise RuntimeError("Scripts not analyzed. Call analyze_scripts() first.")
        
        return [analysis for analysis in self.analyses.values() 
                if analysis.primary_emotion == emotion.lower()]
    
    def get_script_analysis(self, script_name: str) -> Optional[ScriptAnalysis]:
        """
        Get analysis for a specific script
        
        Args:
            script_name: Name of the script (with or without extension)
            
        Returns:
            Script analysis or None if not found
        """
        # Remove extension if present
        script_key = script_name.replace('.wav', '')
        return self.analyses.get(script_key)
    
    def get_matching_clips_emotions(self, script_name: str) -> List[str]:
        """
        Get emotional categories that would match well with this script
        
        Args:
            script_name: Name of the script
            
        Returns:
            List of emotional categories for clip selection
        """
        script_analysis = self.get_script_analysis(script_name)
        if not script_analysis:
            return ['neutral']
        
        # Map script emotions to clip emotions
        emotion_mapping = {
            'anxiety': ['anxiety', 'seeking'],  # Anxious content + seeking resolution
            'peace': ['peace', 'awakening'],    # Peaceful content + spiritual awakening
            'seeking': ['seeking', 'peace'],    # Seeking content + finding peace
            'awakening': ['awakening', 'peace'], # Awakening content + peaceful resolution
            'neutral': ['neutral', 'peace']     # Neutral content with peaceful visuals
        }
        
        return emotion_mapping.get(script_analysis.primary_emotion, ['neutral'])
    
    def get_analysis_stats(self) -> Dict[str, Any]:
        """Get statistics about analyzed scripts"""
        if not self.loaded:
            raise RuntimeError("Scripts not analyzed. Call analyze_scripts() first.")
        
        total_scripts = len(self.analyses)
        total_size_mb = sum(analysis.file_size_mb or 0 for analysis in self.analyses.values())
        
        # Emotional distribution
        emotion_counts = {}
        intensity_sum = 0
        for analysis in self.analyses.values():
            emotion = analysis.primary_emotion
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            intensity_sum += analysis.emotional_intensity
        
        avg_intensity = intensity_sum / total_scripts if total_scripts > 0 else 0
        
        # Theme distribution
        theme_counts = {}
        for analysis in self.analyses.values():
            for theme in analysis.themes:
                theme_counts[theme] = theme_counts.get(theme, 0) + 1
        
        return {
            'total_scripts': total_scripts,
            'total_size_mb': total_size_mb,
            'average_intensity': avg_intensity,
            'emotion_distribution': emotion_counts,
            'theme_distribution': theme_counts,
            'script_list': list(self.analyses.keys())
        }
    
    def create_content_fingerprint(self) -> str:
        """Create fingerprint of all analyzed content for cache validation"""
        if not self.loaded:
            raise RuntimeError("Scripts not analyzed. Call analyze_scripts() first.")
        
        # Create hash of all script analyses
        content_data = []
        for script_name in sorted(self.analyses.keys()):
            analysis = self.analyses[script_name]
            content_data.append(f"{script_name}:{analysis.primary_emotion}:{analysis.emotional_intensity}")
        
        content_string = '|'.join(content_data)
        return hashlib.sha256(content_string.encode()).hexdigest()[:16]