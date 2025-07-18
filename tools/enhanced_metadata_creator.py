#!/usr/bin/env python3
"""
Enhanced Metadata Creator with Visual Content Analysis

Supplements filename-based analysis with actual visual content analysis
using computer vision techniques for better semantic accuracy.
"""

import json
import logging
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import argparse
import sys
from collections import Counter, defaultdict
import re

# Try to import computer vision libraries
try:
    import cv2
    import numpy as np
    CV_AVAILABLE = True
except ImportError:
    CV_AVAILABLE = False
    print("Warning: OpenCV not available. Using basic visual analysis.")

# Try to import moviepy for video metadata
try:
    from moviepy import VideoFileClip
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False
    print("Warning: moviepy not available. Using basic metadata extraction.")

logger = logging.getLogger(__name__)

class EnhancedMetadataCreator:
    """Enhanced metadata creator with visual content analysis capabilities"""
    
    def __init__(self):
        """Initialize the enhanced metadata creator"""
        self.color_mappings = {
            'warm': ['orange', 'yellow', 'red', 'golden'],
            'cool': ['blue', 'green', 'purple', 'cyan'],
            'neutral': ['white', 'gray', 'black', 'beige'],
            'vibrant': ['bright', 'saturated', 'vivid'],
            'muted': ['soft', 'pastel', 'subdued', 'gentle']
        }
        
        self.motion_patterns = {
            'static': ['still', 'motionless', 'stable', 'fixed'],
            'gentle': ['slow', 'flowing', 'smooth', 'peaceful'],
            'dynamic': ['fast', 'energetic', 'active', 'vibrant'],
            'rhythmic': ['pulsing', 'rhythmic', 'patterned', 'cyclical']
        }
        
        self.composition_patterns = {
            'centered': ['balanced', 'symmetric', 'focused'],
            'rule_of_thirds': ['dynamic', 'artistic', 'engaging'],
            'close_up': ['intimate', 'detailed', 'personal'],
            'wide_shot': ['expansive', 'environmental', 'contextual']
        }
    
    def create_enhanced_metadata(self, 
                                clips_directory: Path, 
                                output_file: Path,
                                force_recreate: bool = False,
                                use_visual_analysis: bool = True) -> bool:
        """
        Create enhanced metadata with optional visual content analysis
        
        Args:
            clips_directory: Directory containing video clips
            output_file: Path to output metadata JSON file
            force_recreate: Whether to recreate existing metadata file
            use_visual_analysis: Whether to perform visual content analysis
            
        Returns:
            True if successful, False otherwise
        """
        if output_file.exists() and not force_recreate:
            logger.info(f"Enhanced metadata file already exists: {output_file}")
            logger.info("Use --force to recreate it")
            return True
        
        if not clips_directory.exists():
            logger.error(f"Clips directory not found: {clips_directory}")
            return False
        
        # Find all video files
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
        video_files = []
        
        for ext in video_extensions:
            video_files.extend(clips_directory.glob(f"*{ext}"))
        
        if not video_files:
            logger.error(f"No video files found in {clips_directory}")
            return False
        
        logger.info(f"Found {len(video_files)} video files for enhanced analysis")
        
        # Create enhanced metadata for each clip
        clips_metadata = []
        
        for i, video_file in enumerate(sorted(video_files), 1):
            logger.info(f"Processing {i}/{len(video_files)}: {video_file.name}")
            
            try:
                # Generate unique ID
                clip_id = hashlib.md5(video_file.name.encode()).hexdigest()[:12]
                
                # Extract basic video metadata
                video_metadata = self._extract_video_metadata(video_file)
                
                # Enhanced filename analysis
                filename_analysis = self._analyze_filename_enhanced(video_file.name)
                
                # Visual content analysis (if enabled and available)
                visual_analysis = {}
                if use_visual_analysis and CV_AVAILABLE:
                    visual_analysis = self._analyze_visual_content(video_file)
                
                # Combine analyses for enhanced tags
                enhanced_tags = self._combine_analyses(
                    filename_analysis, visual_analysis, video_metadata
                )
                
                # Enhanced shot analysis
                enhanced_shot_analysis = self._create_enhanced_shot_analysis(
                    filename_analysis, visual_analysis
                )
                
                # Semantic categorization
                semantic_categories = self._categorize_semantically(
                    enhanced_tags, enhanced_shot_analysis
                )
                
                # Create enhanced clip metadata
                clip_metadata = {
                    "id": clip_id,
                    "filename": video_file.name,
                    "tags": enhanced_tags,
                    "duration": video_metadata["duration"],
                    "resolution": video_metadata["resolution"],
                    "fps": video_metadata["fps"],
                    "file_size_mb": video_metadata["file_size_mb"],
                    "shot_analysis": enhanced_shot_analysis,
                    "semantic_categories": semantic_categories,
                    "visual_analysis": visual_analysis,
                    "enhancement_version": "2.0",
                    "created_at": datetime.now().isoformat()
                }
                
                clips_metadata.append(clip_metadata)
                
            except Exception as e:
                logger.error(f"Failed to process {video_file}: {e}")
                continue
        
        # Create final enhanced metadata structure
        metadata = {
            "metadata_info": {
                "created_at": datetime.now().isoformat(),
                "clips_directory": str(clips_directory),
                "total_clips": len(clips_metadata),
                "generator": "enhanced_metadata_creator.py",
                "version": "2.0",
                "features": {
                    "filename_analysis": True,
                    "visual_content_analysis": use_visual_analysis and CV_AVAILABLE,
                    "semantic_categorization": True,
                    "enhanced_shot_analysis": True
                }
            },
            "clips": clips_metadata
        }
        
        # Save enhanced metadata file
        try:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Successfully created enhanced metadata for {len(clips_metadata)} clips")
            logger.info(f"Enhanced metadata saved to: {output_file}")
            
            # Print comprehensive summary
            self._print_enhancement_summary(clips_metadata, use_visual_analysis and CV_AVAILABLE)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to save enhanced metadata file: {e}")
            return False
    
    def _extract_video_metadata(self, video_path: Path) -> Dict[str, Any]:
        """Extract metadata from a video file with enhanced information"""
        metadata = {
            "duration": 5.21,  # Default duration
            "resolution": "1080x1936",  # Default resolution
            "fps": 24.0,  # Default FPS
            "file_size_mb": round(video_path.stat().st_size / (1024 * 1024), 2)
        }
        
        if MOVIEPY_AVAILABLE:
            try:
                with VideoFileClip(str(video_path)) as video:
                    metadata.update({
                        "duration": round(video.duration, 2),
                        "resolution": f"{video.w}x{video.h}",
                        "fps": round(video.fps, 1) if video.fps else 24.0,
                        "has_audio": video.audio is not None,
                        "aspect_ratio": round(video.w / video.h, 2) if video.h > 0 else 1.0
                    })
            except Exception as e:
                logger.warning(f"Failed to extract enhanced metadata from {video_path}: {e}")
        
        return metadata
    
    def _analyze_filename_enhanced(self, filename: str) -> Dict[str, Any]:
        """Enhanced filename analysis with additional semantic extraction"""
        filename_lower = filename.lower()
        
        # Original comprehensive keyword analysis (from create_mjanime_metadata.py)
        # Enhanced with additional semantic categories
        
        tags = []
        shot_analysis = {}
        
        # EXPANDED CHARACTER AND SUBJECT ANALYSIS
        character_keywords = {
            'devotee': ['devotee', 'spiritual_person', 'religious_figure', 'practitioner', 'seeker'],
            'devotees': ['devotees', 'spiritual_community', 'group_worship', 'congregation', 'gathering'],
            'monks': ['monks', 'religious_order', 'monastic', 'spiritual_brotherhood', 'contemplatives'],
            'solitary': ['lone', 'solitary', 'individual', 'alone', 'isolated', 'single_person'],
            'community': ['group', 'community', 'together', 'collective', 'shared'],
            'teacher': ['teacher', 'guru', 'guide', 'master', 'instructor'],
            'student': ['student', 'disciple', 'learner', 'apprentice', 'novice']
        }
        
        # ENHANCED ACTION AND ACTIVITY ANALYSIS
        action_keywords = {
            'meditation': ['meditation', 'contemplation', 'mindfulness', 'dhyana', 'inner_focus', 'centering'],
            'worship': ['worship', 'chanting', 'prayer', 'devotion', 'kirtan', 'mantra'],
            'movement': ['walking', 'floating', 'dancing', 'moving', 'flowing', 'drifting'],
            'stillness': ['sitting', 'lying', 'resting', 'still', 'motionless', 'peaceful'],
            'service': ['service', 'seva', 'offering', 'giving', 'helping', 'serving'],
            'learning': ['reading', 'studying', 'learning', 'contemplating', 'reflecting'],
            'creating': ['painting', 'decorating', 'creating', 'making', 'crafting']
        }
        
        # ENHANCED LOCATION AND ENVIRONMENT ANALYSIS
        location_keywords = {
            'sacred_spaces': ['temple', 'shrine', 'altar', 'sanctuary', 'sacred_space'],
            'natural_settings': ['garden', 'forest', 'mountain', 'stream', 'lake', 'nature'],
            'architectural': ['courtyard', 'balcony', 'rooftop', 'terrace', 'building'],
            'elevated': ['cliff', 'mountain', 'high', 'elevated', 'above', 'peak'],
            'water_features': ['waterfall', 'stream', 'lake', 'water', 'flowing'],
            'intimate_spaces': ['cave', 'interior', 'private', 'enclosed', 'sheltered'],
            'open_spaces': ['field', 'meadow', 'open', 'expansive', 'wide'],
            'celestial': ['clouds', 'sky', 'heavenly', 'atmospheric', 'ethereal']
        }
        
        # ENHANCED MOOD AND ATMOSPHERE ANALYSIS
        mood_keywords = {
            'peaceful': ['peaceful', 'serene', 'tranquil', 'calm', 'harmonious', 'still'],
            'dramatic': ['dramatic', 'intense', 'powerful', 'striking', 'bold', 'dynamic'],
            'gentle': ['gentle', 'soft', 'tender', 'mild', 'soothing', 'delicate'],
            'vibrant': ['vibrant', 'lively', 'energetic', 'colorful', 'bright', 'alive'],
            'mystical': ['mystical', 'magical', 'enchanted', 'otherworldly', 'ethereal'],
            'contemplative': ['contemplative', 'reflective', 'thoughtful', 'introspective'],
            'joyful': ['joyful', 'happy', 'celebratory', 'festive', 'uplifting'],
            'solemn': ['solemn', 'serious', 'reverent', 'respectful', 'dignified']
        }
        
        # ENHANCED SPIRITUAL AND PHILOSOPHICAL CONCEPTS
        spiritual_keywords = {
            'devotional_practice': ['devotional', 'bhakti', 'worship', 'surrender', 'dedication'],
            'transcendence': ['transcendent', 'divine', 'eternal', 'infinite', 'beyond'],
            'consciousness': ['consciousness', 'awareness', 'mindfulness', 'presence'],
            'unity': ['unity', 'oneness', 'connection', 'harmony', 'integration'],
            'transformation': ['transformation', 'growth', 'evolution', 'change', 'becoming'],
            'wisdom': ['wisdom', 'insight', 'understanding', 'knowledge', 'realization'],
            'compassion': ['compassion', 'love', 'kindness', 'mercy', 'caring'],
            'liberation': ['liberation', 'freedom', 'release', 'moksha', 'enlightenment']
        }
        
        # Combine all keyword dictionaries
        all_keywords = {
            **character_keywords,
            **action_keywords,
            **location_keywords,
            **mood_keywords,
            **spiritual_keywords
        }
        
        # Analyze filename for enhanced content
        semantic_categories = defaultdict(list)
        
        for keyword, related_tags in all_keywords.items():
            if keyword.replace('_', ' ') in filename_lower or keyword.replace('_', '') in filename_lower.replace('_', ''):
                tags.extend(related_tags)
                
                # Categorize semantically
                if keyword in character_keywords:
                    semantic_categories['character_types'].extend(related_tags)
                elif keyword in action_keywords:
                    semantic_categories['activities'].extend(related_tags)
                elif keyword in location_keywords:
                    semantic_categories['environments'].extend(related_tags)
                elif keyword in mood_keywords:
                    semantic_categories['atmospheres'].extend(related_tags)
                elif keyword in spiritual_keywords:
                    semantic_categories['spiritual_concepts'].extend(related_tags)
        
        # ENHANCED LIGHTING ANALYSIS with more nuance
        lighting_analysis = self._analyze_lighting_enhanced(filename_lower)
        shot_analysis.update(lighting_analysis)
        tags.extend(lighting_analysis.get('lighting_tags', []))
        
        # ENHANCED CAMERA MOVEMENT ANALYSIS
        movement_analysis = self._analyze_movement_enhanced(filename_lower)
        shot_analysis.update(movement_analysis)
        tags.extend(movement_analysis.get('movement_tags', []))
        
        # ENHANCED SHOT TYPE ANALYSIS
        shot_type_analysis = self._analyze_shot_type_enhanced(filename_lower)
        shot_analysis.update(shot_type_analysis)
        tags.extend(shot_type_analysis.get('shot_tags', []))
        
        # EMOTIONAL TONE AND ENERGY ANALYSIS
        emotional_analysis = self._analyze_emotional_tone(filename_lower)
        tags.extend(emotional_analysis.get('emotional_tags', []))
        
        # Remove duplicates and organize
        tags = list(set(tags))
        
        # Prioritize tags by relevance and frequency
        primary_tags = [tag for tag in tags if any(keyword in filename_lower 
                      for keyword in ['devotee', 'temple', 'meditation', 'spiritual', 'peaceful'])]
        secondary_tags = [tag for tag in tags if tag not in primary_tags]
        
        organized_tags = primary_tags + secondary_tags
        final_tags = organized_tags[:30]  # Increased from 25 to 30
        
        return {
            "tags": final_tags,
            "shot_analysis": shot_analysis,
            "semantic_categories": dict(semantic_categories),
            "analysis_source": "enhanced_filename"
        }
    
    def _analyze_lighting_enhanced(self, filename_lower: str) -> Dict[str, Any]:
        """Enhanced lighting analysis with more granular categorization"""
        lighting_patterns = {
            'dramatic': {
                'keywords': ['night', 'dark', 'shadow', 'dramatic', 'blazing', 'intense'],
                'characteristics': ['high_contrast', 'mood_lighting', 'chiaroscuro', 'atmospheric'],
                'emotional_impact': ['intense', 'mysterious', 'powerful']
            },
            'bright': {
                'keywords': ['bright', 'sunrise', 'sunlit', 'glowing', 'crystal', 'luminous'],
                'characteristics': ['natural_illumination', 'well_lit', 'radiant', 'clear'],
                'emotional_impact': ['uplifting', 'clear', 'energetic']
            },
            'soft': {
                'keywords': ['filtered', 'soft', 'gentle', 'diffused', 'tender'],
                'characteristics': ['diffused_light', 'gentle_illumination', 'even_lighting'],
                'emotional_impact': ['peaceful', 'nurturing', 'calming']
            },
            'golden': {
                'keywords': ['golden', 'warm', 'sunset', 'amber', 'honey'],
                'characteristics': ['warm_tones', 'golden_hour', 'rich_lighting'],
                'emotional_impact': ['warm', 'nostalgic', 'romantic']
            },
            'ethereal': {
                'keywords': ['ethereal', 'misty', 'foggy', 'hazy', 'dreamlike'],
                'characteristics': ['atmospheric_lighting', 'mystical', 'otherworldly'],
                'emotional_impact': ['mystical', 'transcendent', 'dreamy']
            }
        }
        
        for lighting_type, config in lighting_patterns.items():
            if any(word in filename_lower for word in config['keywords']):
                return {
                    'lighting_type': lighting_type,
                    'lighting_characteristics': config['characteristics'],
                    'lighting_tags': config['characteristics'] + config['emotional_impact']
                }
        
        # Default natural lighting
        return {
            'lighting_type': 'natural',
            'lighting_characteristics': ['natural_lighting', 'ambient_light'],
            'lighting_tags': ['natural_lighting', 'ambient_light', 'balanced']
        }
    
    def _analyze_movement_enhanced(self, filename_lower: str) -> Dict[str, Any]:
        """Enhanced camera movement analysis"""
        movement_patterns = {
            'dynamic': {
                'keywords': ['floating', 'flying', 'rapidly', 'pans', 'dynamic', 'energetic'],
                'characteristics': ['fluid_motion', 'kinetic', 'active_camera'],
                'energy_level': 'high'
            },
            'gentle': {
                'keywords': ['gentle', 'flowing', 'smooth', 'drifting', 'graceful'],
                'characteristics': ['smooth_motion', 'flowing', 'graceful_movement'],
                'energy_level': 'medium'
            },
            'static': {
                'keywords': ['sitting', 'meditation', 'peaceful', 'lies', 'kneels', 'still'],
                'characteristics': ['stable_frame', 'contemplative_pace', 'steady'],
                'energy_level': 'low'
            },
            'rhythmic': {
                'keywords': ['rhythmic', 'pulsing', 'breathing', 'cyclical', 'repetitive'],
                'characteristics': ['rhythmic_movement', 'patterned', 'meditative_rhythm'],
                'energy_level': 'medium'
            }
        }
        
        for movement_type, config in movement_patterns.items():
            if any(word in filename_lower for word in config['keywords']):
                return {
                    'movement_type': movement_type,
                    'movement_characteristics': config['characteristics'],
                    'movement_tags': config['characteristics'],
                    'energy_level': config['energy_level']
                }
        
        # Default gentle movement
        return {
            'movement_type': 'gentle',
            'movement_characteristics': ['gentle_movement', 'smooth_motion'],
            'movement_tags': ['gentle_movement', 'smooth_motion'],
            'energy_level': 'medium'
        }
    
    def _analyze_shot_type_enhanced(self, filename_lower: str) -> Dict[str, Any]:
        """Enhanced shot type analysis with composition insights"""
        shot_patterns = {
            'wide_shot': {
                'keywords': ['group', 'crowd', 'procession', 'devotees', 'monks', 'landscape', 'wide'],
                'characteristics': ['wide_framing', 'environmental_context', 'establishing_shot'],
                'composition': 'environmental',
                'intimacy_level': 'distant'
            },
            'medium_shot': {
                'keywords': ['person', 'figure', 'standing', 'walking', 'medium'],
                'characteristics': ['balanced_composition', 'subject_focus', 'narrative_framing'],
                'composition': 'balanced',
                'intimacy_level': 'moderate'
            },
            'close_up': {
                'keywords': ['face', 'close', 'detail', 'eye', 'hands', 'intimate'],
                'characteristics': ['intimate_framing', 'detail_focus', 'emotional_connection'],
                'composition': 'intimate',
                'intimacy_level': 'high'
            },
            'extreme_wide': {
                'keywords': ['vast', 'panoramic', 'horizon', 'expansive', 'aerial'],
                'characteristics': ['panoramic_view', 'grand_scale', 'contextual_scope'],
                'composition': 'epic',
                'intimacy_level': 'very_distant'
            }
        }
        
        for shot_type, config in shot_patterns.items():
            if any(word in filename_lower for word in config['keywords']):
                return {
                    'shot_type': shot_type,
                    'shot_characteristics': config['characteristics'],
                    'shot_tags': config['characteristics'],
                    'composition_style': config['composition'],
                    'intimacy_level': config['intimacy_level']
                }
        
        # Default medium shot
        return {
            'shot_type': 'medium_shot',
            'shot_characteristics': ['balanced_composition', 'subject_focus'],
            'shot_tags': ['balanced_composition', 'subject_focus'],
            'composition_style': 'balanced',
            'intimacy_level': 'moderate'
        }
    
    def _analyze_emotional_tone(self, filename_lower: str) -> Dict[str, Any]:
        """Analyze emotional tone and energy from filename"""
        emotional_patterns = {
            'peaceful': {
                'keywords': ['peaceful', 'serene', 'calm', 'tranquil', 'meditation', 'quiet'],
                'energy': 'low',
                'valence': 'positive',
                'activation': 'calm'
            },
            'spiritual': {
                'keywords': ['spiritual', 'sacred', 'devotional', 'prayer', 'divine'],
                'energy': 'medium',
                'valence': 'positive',
                'activation': 'reverent'
            },
            'dramatic': {
                'keywords': ['dramatic', 'cliff', 'blazing', 'intense', 'powerful'],
                'energy': 'high',
                'valence': 'neutral',
                'activation': 'intense'
            },
            'gentle': {
                'keywords': ['gentle', 'soft', 'cozy', 'wrapped', 'tender'],
                'energy': 'low',
                'valence': 'positive',
                'activation': 'nurturing'
            },
            'vibrant': {
                'keywords': ['vibrant', 'lively', 'dancing', 'festival', 'celebratory'],
                'energy': 'high',
                'valence': 'positive',
                'activation': 'joyful'
            },
            'contemplative': {
                'keywords': ['alone', 'lone', 'solitary', 'individual', 'contemplative'],
                'energy': 'low',
                'valence': 'neutral',
                'activation': 'introspective'
            }
        }
        
        emotional_tags = []
        for emotion, config in emotional_patterns.items():
            if any(word in filename_lower for word in config['keywords']):
                emotional_tags.extend([
                    emotion,
                    f"{config['energy']}_energy",
                    f"{config['valence']}_valence",
                    config['activation']
                ])
        
        return {'emotional_tags': emotional_tags}
    
    def _analyze_visual_content(self, video_path: Path) -> Dict[str, Any]:
        """Analyze actual visual content using computer vision"""
        if not CV_AVAILABLE:
            return {"analysis_available": False, "reason": "OpenCV not available"}
        
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return {"analysis_available": False, "reason": "Could not open video file"}
            
            # Sample frames from different parts of the video
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            sample_indices = [0, frame_count // 4, frame_count // 2, 3 * frame_count // 4, frame_count - 1]
            
            frames = []
            color_analysis = []
            motion_analysis = []
            composition_analysis = []
            
            for idx in sample_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)
                    
                    # Analyze this frame
                    color_analysis.append(self._analyze_frame_colors(frame))
                    composition_analysis.append(self._analyze_frame_composition(frame))
            
            cap.release()
            
            if len(frames) < 2:
                return {"analysis_available": False, "reason": "Could not extract sufficient frames"}
            
            # Analyze motion between frames
            for i in range(len(frames) - 1):
                motion_analysis.append(self._analyze_frame_motion(frames[i], frames[i + 1]))
            
            # Aggregate analyses
            aggregated_color = self._aggregate_color_analysis(color_analysis)
            aggregated_motion = self._aggregate_motion_analysis(motion_analysis)
            aggregated_composition = self._aggregate_composition_analysis(composition_analysis)
            
            return {
                "analysis_available": True,
                "color_analysis": aggregated_color,
                "motion_analysis": aggregated_motion,
                "composition_analysis": aggregated_composition,
                "frames_analyzed": len(frames),
                "visual_tags": self._generate_visual_tags(aggregated_color, aggregated_motion, aggregated_composition)
            }
            
        except Exception as e:
            logger.warning(f"Visual analysis failed for {video_path}: {e}")
            return {"analysis_available": False, "reason": str(e)}
    
    def _analyze_frame_colors(self, frame: np.ndarray) -> Dict[str, Any]:
        """Analyze color characteristics of a frame"""
        # Convert BGR to HSV for better color analysis
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Calculate color statistics
        hue_mean = np.mean(hsv[:, :, 0])
        saturation_mean = np.mean(hsv[:, :, 1])
        value_mean = np.mean(hsv[:, :, 2])
        
        # Determine dominant color characteristics
        dominant_colors = []
        
        # Hue analysis (color wheel)
        if 0 <= hue_mean < 20 or 160 <= hue_mean <= 180:
            dominant_colors.append('red')
        elif 20 <= hue_mean < 40:
            dominant_colors.append('orange')
        elif 40 <= hue_mean < 80:
            dominant_colors.append('yellow')
        elif 80 <= hue_mean < 140:
            dominant_colors.append('green')
        elif 140 <= hue_mean < 160:
            dominant_colors.append('blue')
        
        # Saturation analysis
        if saturation_mean > 150:
            dominant_colors.append('vibrant')
        elif saturation_mean < 50:
            dominant_colors.append('muted')
        
        # Value analysis (brightness)
        if value_mean > 200:
            dominant_colors.append('bright')
        elif value_mean < 80:
            dominant_colors.append('dark')
        
        return {
            'hue_mean': float(hue_mean),
            'saturation_mean': float(saturation_mean),
            'value_mean': float(value_mean),
            'dominant_colors': dominant_colors
        }
    
    def _analyze_frame_motion(self, frame1: np.ndarray, frame2: np.ndarray) -> Dict[str, Any]:
        """Analyze motion between two frames"""
        # Convert to grayscale
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # Calculate optical flow
        flow = cv2.calcOpticalFlowPyrLK(gray1, gray2, None, None)
        
        # Calculate motion magnitude
        if flow[0] is not None:
            motion_magnitude = np.mean(np.sqrt(flow[0][:, :, 0]**2 + flow[0][:, :, 1]**2))
        else:
            motion_magnitude = 0.0
        
        # Categorize motion level
        if motion_magnitude > 10:
            motion_level = 'high'
        elif motion_magnitude > 3:
            motion_level = 'medium'
        else:
            motion_level = 'low'
        
        return {
            'motion_magnitude': float(motion_magnitude),
            'motion_level': motion_level
        }
    
    def _analyze_frame_composition(self, frame: np.ndarray) -> Dict[str, Any]:
        """Analyze composition characteristics of a frame"""
        height, width = frame.shape[:2]
        
        # Convert to grayscale for edge detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (height * width)
        
        # Analyze brightness distribution
        brightness_std = np.std(gray)
        
        # Analyze symmetry (simplified)
        left_half = gray[:, :width//2]
        right_half = cv2.flip(gray[:, width//2:], 1)
        if right_half.shape[1] == left_half.shape[1]:
            symmetry_score = cv2.matchTemplate(left_half, right_half, cv2.TM_CCOEFF_NORMED)[0, 0]
        else:
            symmetry_score = 0.0
        
        return {
            'edge_density': float(edge_density),
            'brightness_std': float(brightness_std),
            'symmetry_score': float(symmetry_score),
            'composition_complexity': 'high' if edge_density > 0.1 else 'medium' if edge_density > 0.05 else 'low'
        }
    
    def _aggregate_color_analysis(self, color_analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate color analysis across frames"""
        if not color_analyses:
            return {}
        
        # Average color characteristics
        avg_hue = np.mean([ca['hue_mean'] for ca in color_analyses])
        avg_saturation = np.mean([ca['saturation_mean'] for ca in color_analyses])
        avg_value = np.mean([ca['value_mean'] for ca in color_analyses])
        
        # Collect all dominant colors
        all_colors = []
        for ca in color_analyses:
            all_colors.extend(ca['dominant_colors'])
        
        color_counter = Counter(all_colors)
        dominant_colors = [color for color, count in color_counter.most_common(3)]
        
        return {
            'average_hue': float(avg_hue),
            'average_saturation': float(avg_saturation),
            'average_value': float(avg_value),
            'dominant_colors': dominant_colors,
            'color_consistency': len(set(all_colors)) / len(all_colors) if all_colors else 0
        }
    
    def _aggregate_motion_analysis(self, motion_analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate motion analysis across frame pairs"""
        if not motion_analyses:
            return {}
        
        avg_motion = np.mean([ma['motion_magnitude'] for ma in motion_analyses])
        motion_levels = [ma['motion_level'] for ma in motion_analyses]
        motion_counter = Counter(motion_levels)
        dominant_motion = motion_counter.most_common(1)[0][0] if motion_counter else 'low'
        
        return {
            'average_motion_magnitude': float(avg_motion),
            'dominant_motion_level': dominant_motion,
            'motion_consistency': motion_counter[dominant_motion] / len(motion_levels) if motion_levels else 0
        }
    
    def _aggregate_composition_analysis(self, composition_analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate composition analysis across frames"""
        if not composition_analyses:
            return {}
        
        avg_edge_density = np.mean([ca['edge_density'] for ca in composition_analyses])
        avg_brightness_std = np.mean([ca['brightness_std'] for ca in composition_analyses])
        avg_symmetry = np.mean([ca['symmetry_score'] for ca in composition_analyses])
        
        complexities = [ca['composition_complexity'] for ca in composition_analyses]
        complexity_counter = Counter(complexities)
        dominant_complexity = complexity_counter.most_common(1)[0][0] if complexity_counter else 'medium'
        
        return {
            'average_edge_density': float(avg_edge_density),
            'average_brightness_std': float(avg_brightness_std),
            'average_symmetry_score': float(avg_symmetry),
            'dominant_complexity': dominant_complexity
        }
    
    def _generate_visual_tags(self, color_analysis: Dict, motion_analysis: Dict, composition_analysis: Dict) -> List[str]:
        """Generate visual tags based on analysis results"""
        tags = []
        
        # Color-based tags
        if color_analysis:
            tags.extend(color_analysis.get('dominant_colors', []))
            
            if color_analysis.get('average_saturation', 0) > 150:
                tags.append('high_saturation')
            elif color_analysis.get('average_saturation', 0) < 50:
                tags.append('low_saturation')
            
            if color_analysis.get('average_value', 0) > 200:
                tags.append('bright_visuals')
            elif color_analysis.get('average_value', 0) < 80:
                tags.append('dark_visuals')
        
        # Motion-based tags
        if motion_analysis:
            motion_level = motion_analysis.get('dominant_motion_level', 'low')
            tags.append(f'{motion_level}_motion')
            
            if motion_analysis.get('motion_consistency', 0) > 0.7:
                tags.append('consistent_motion')
            else:
                tags.append('varied_motion')
        
        # Composition-based tags
        if composition_analysis:
            complexity = composition_analysis.get('dominant_complexity', 'medium')
            tags.append(f'{complexity}_complexity')
            
            if composition_analysis.get('average_symmetry_score', 0) > 0.7:
                tags.append('symmetric_composition')
            
            if composition_analysis.get('average_edge_density', 0) > 0.1:
                tags.append('detailed_visuals')
        
        return list(set(tags))
    
    def _combine_analyses(self, filename_analysis: Dict, visual_analysis: Dict, video_metadata: Dict) -> List[str]:
        """Combine filename and visual analyses for enhanced tags"""
        combined_tags = filename_analysis['tags'].copy()
        
        # Add visual analysis tags if available
        if visual_analysis.get('analysis_available'):
            visual_tags = visual_analysis.get('visual_tags', [])
            combined_tags.extend(visual_tags)
        
        # Add metadata-derived tags
        aspect_ratio = video_metadata.get('aspect_ratio', 1.0)
        if aspect_ratio > 1.5:
            combined_tags.append('widescreen')
        elif aspect_ratio < 0.8:
            combined_tags.append('portrait')
        else:
            combined_tags.append('square_format')
        
        # Add duration-based tags
        duration = video_metadata.get('duration', 0)
        if duration > 10:
            combined_tags.append('long_form')
        elif duration < 3:
            combined_tags.append('short_form')
        else:
            combined_tags.append('medium_form')
        
        # Remove duplicates and limit
        return list(set(combined_tags))[:35]  # Increased to 35 tags
    
    def _create_enhanced_shot_analysis(self, filename_analysis: Dict, visual_analysis: Dict) -> Dict[str, Any]:
        """Create enhanced shot analysis combining filename and visual data"""
        enhanced_analysis = filename_analysis['shot_analysis'].copy()
        
        # Add visual analysis enhancements
        if visual_analysis.get('analysis_available'):
            color_data = visual_analysis.get('color_analysis', {})
            motion_data = visual_analysis.get('motion_analysis', {})
            composition_data = visual_analysis.get('composition_analysis', {})
            
            # Enhance lighting analysis with actual color data
            if color_data:
                if color_data.get('average_value', 0) > 200:
                    enhanced_analysis['actual_brightness'] = 'bright'
                elif color_data.get('average_value', 0) < 80:
                    enhanced_analysis['actual_brightness'] = 'dark'
                else:
                    enhanced_analysis['actual_brightness'] = 'medium'
                
                enhanced_analysis['color_vibrancy'] = 'high' if color_data.get('average_saturation', 0) > 150 else 'low'
                enhanced_analysis['dominant_colors'] = color_data.get('dominant_colors', [])
            
            # Enhance movement analysis with actual motion data
            if motion_data:
                enhanced_analysis['actual_motion_level'] = motion_data.get('dominant_motion_level', 'low')
                enhanced_analysis['motion_consistency'] = motion_data.get('motion_consistency', 0)
            
            # Add composition analysis
            if composition_data:
                enhanced_analysis['visual_complexity'] = composition_data.get('dominant_complexity', 'medium')
                enhanced_analysis['symmetry_score'] = composition_data.get('average_symmetry_score', 0)
        
        return enhanced_analysis
    
    def _categorize_semantically(self, tags: List[str], shot_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Create semantic categorization for enhanced understanding"""
        categories = {
            'spiritual_elements': [],
            'visual_aesthetics': [],
            'emotional_resonance': [],
            'compositional_style': [],
            'narrative_context': []
        }
        
        # Categorize tags semantically
        spiritual_indicators = ['spiritual', 'sacred', 'divine', 'devotional', 'meditation', 'temple', 'worship']
        aesthetic_indicators = ['beautiful', 'artistic', 'dramatic', 'bright', 'colorful', 'serene']
        emotional_indicators = ['peaceful', 'intense', 'gentle', 'vibrant', 'contemplative', 'joyful']
        compositional_indicators = ['balanced', 'symmetric', 'dynamic', 'static', 'close_up', 'wide_shot']
        narrative_indicators = ['opening', 'development', 'climax', 'resolution', 'journey', 'transformation']
        
        for tag in tags:
            tag_lower = tag.lower()
            if any(indicator in tag_lower for indicator in spiritual_indicators):
                categories['spiritual_elements'].append(tag)
            if any(indicator in tag_lower for indicator in aesthetic_indicators):
                categories['visual_aesthetics'].append(tag)
            if any(indicator in tag_lower for indicator in emotional_indicators):
                categories['emotional_resonance'].append(tag)
            if any(indicator in tag_lower for indicator in compositional_indicators):
                categories['compositional_style'].append(tag)
            if any(indicator in tag_lower for indicator in narrative_indicators):
                categories['narrative_context'].append(tag)
        
        # Add shot analysis insights
        categories['technical_analysis'] = {
            'lighting_type': shot_analysis.get('lighting_type', 'natural'),
            'movement_type': shot_analysis.get('movement_type', 'gentle'),
            'shot_type': shot_analysis.get('shot_type', 'medium_shot'),
            'energy_level': shot_analysis.get('energy_level', 'medium')
        }
        
        return categories
    
    def _print_enhancement_summary(self, clips_metadata: List[Dict], visual_analysis_enabled: bool):
        """Print comprehensive enhancement summary"""
        print(f"\n{'='*70}")
        print(f"ENHANCED METADATA CREATION COMPLETE")
        print(f"{'='*70}")
        print(f"Total clips processed: {len(clips_metadata)}")
        print(f"Visual content analysis: {'Enabled' if visual_analysis_enabled else 'Disabled'}")
        
        # Analyze enhancement features
        enhanced_features = {
            'filename_analysis': 0,
            'visual_analysis': 0,
            'semantic_categorization': 0,
            'enhanced_shot_analysis': 0
        }
        
        total_tags = []
        semantic_categories = []
        
        for clip in clips_metadata:
            if clip.get('enhancement_version') == '2.0':
                enhanced_features['filename_analysis'] += 1
            
            if clip.get('visual_analysis', {}).get('analysis_available'):
                enhanced_features['visual_analysis'] += 1
            
            if clip.get('semantic_categories'):
                enhanced_features['semantic_categorization'] += 1
            
            if clip.get('shot_analysis', {}).get('actual_brightness'):
                enhanced_features['enhanced_shot_analysis'] += 1
            
            total_tags.extend(clip.get('tags', []))
            
            for category, items in clip.get('semantic_categories', {}).items():
                semantic_categories.extend(items)
        
        print(f"\nEnhancement Features:")
        for feature, count in enhanced_features.items():
            print(f"  {feature.replace('_', ' ').title()}: {count}/{len(clips_metadata)}")
        
        # Tag analysis
        tag_counts = Counter(total_tags)
        print(f"\nTop enhanced tags:")
        for tag, count in tag_counts.most_common(10):
            print(f"  {tag}: {count}")
        
        # Semantic analysis
        if semantic_categories:
            semantic_counts = Counter(semantic_categories)
            print(f"\nTop semantic categories:")
            for category, count in semantic_counts.most_common(8):
                print(f"  {category}: {count}")
        
        print(f"\n{'='*70}")


def main():
    """Main function for enhanced metadata creation"""
    parser = argparse.ArgumentParser(
        description="Create enhanced metadata with visual content analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create enhanced metadata with visual analysis
  python enhanced_metadata_creator.py /path/to/clips
  
  # Create enhanced metadata without visual analysis
  python enhanced_metadata_creator.py /path/to/clips --no-visual-analysis
  
  # Force recreate existing metadata
  python enhanced_metadata_creator.py /path/to/clips --force
  
  # Custom output location
  python enhanced_metadata_creator.py /path/to/clips --output enhanced_metadata.json
        """
    )
    
    parser.add_argument("clips_directory", 
                       help="Directory containing video clips")
    parser.add_argument("--output", "-o",
                       help="Output metadata file (default: ./enhanced_clips_metadata.json)")
    parser.add_argument("--force", action="store_true",
                       help="Force recreate existing metadata file")
    parser.add_argument("--no-visual-analysis", action="store_true",
                       help="Disable visual content analysis")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       default="INFO", help="Set logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Determine paths
    clips_directory = Path(args.clips_directory)
    if args.output:
        output_file = Path(args.output)
    else:
        output_file = Path("./enhanced_clips_metadata.json")
    
    # Create enhanced metadata
    creator = EnhancedMetadataCreator()
    success = creator.create_enhanced_metadata(
        clips_directory=clips_directory,
        output_file=output_file,
        force_recreate=args.force,
        use_visual_analysis=not args.no_visual_analysis
    )
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()