#!/usr/bin/env python3
"""
Create MJAnime Metadata File

Scans the MJAnime directory and creates a metadata JSON file for the clips.
This enables the MJAnime analyzer to work with the existing video collection.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any
import hashlib
from datetime import datetime
import argparse
import sys

# Try to import moviepy for video metadata extraction
try:
    from moviepy import VideoFileClip
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False
    print("Warning: moviepy not available. Using basic metadata extraction.")


def extract_video_metadata(video_path: Path) -> Dict[str, Any]:
    """Extract metadata from a video file."""
    metadata = {
        "duration": 5.21,  # Default duration
        "resolution": "1080x1936",  # Default resolution for MJAnime
        "fps": 24.0,  # Default FPS
        "file_size_mb": round(video_path.stat().st_size / (1024 * 1024), 2)
    }
    
    if MOVIEPY_AVAILABLE:
        try:
            with VideoFileClip(str(video_path)) as video:
                metadata.update({
                    "duration": round(video.duration, 2),
                    "resolution": f"{video.w}x{video.h}",
                    "fps": round(video.fps, 1) if video.fps else 24.0
                })
        except Exception as e:
            logging.warning(f"Failed to extract metadata from {video_path}: {e}")
    
    return metadata


def analyze_filename_for_content(filename: str) -> Dict[str, Any]:
    """Analyze filename to extract detailed content tags and emotional categories."""
    filename_lower = filename.lower()
    
    # Extract content from filename patterns
    tags = []
    shot_analysis = {}
    
    # CHARACTERS AND SUBJECTS
    character_keywords = {
        'devotee': ['devotee', 'spiritual_person', 'religious_figure', 'monk', 'practitioner'],
        'devotees': ['devotees', 'spiritual_community', 'group_worship', 'congregation', 'gathering'],
        'brahmachari': ['brahmachari', 'celibate_monk', 'student_monk', 'renunciate', 'ascetic'],
        'monks': ['monks', 'religious_order', 'monastic', 'spiritual_brotherhood', 'contemplatives'],
        'hare_krishna': ['hare_krishna', 'iskcon', 'krishna_devotee', 'vaishnava', 'bhakti'],
        'lone': ['solitary', 'individual', 'alone', 'isolated', 'single_person'],
        'young': ['young_person', 'youth', 'student', 'novice', 'apprentice'],
        'humble': ['humble_person', 'modest', 'simple', 'unpretentious', 'meek'],
        'saffron': ['saffron_robed', 'orange_garment', 'monk_attire', 'religious_clothing']
    }
    
    # ACTIONS AND ACTIVITIES
    action_keywords = {
        'meditation': ['meditation', 'contemplation', 'mindfulness', 'dhyana', 'inner_focus'],
        'meditates': ['meditating', 'contemplating', 'reflecting', 'centering', 'focusing'],
        'chants': ['chanting', 'mantra', 'kirtan', 'sacred_sound', 'vocal_prayer'],
        'walking': ['walking', 'strolling', 'pacing', 'pilgrimage', 'journey'],
        'barefoot': ['barefoot_walking', 'grounded', 'natural_connection', 'humble_approach'],
        'floating': ['floating', 'levitating', 'suspended', 'ethereal_movement', 'transcendent'],
        'sitting': ['sitting', 'seated_posture', 'stable_position', 'grounded_pose'],
        'cross-legged': ['cross_legged', 'lotus_position', 'sukhasana', 'meditation_pose'],
        'kneels': ['kneeling', 'genuflection', 'reverent_posture', 'humble_bow', 'prayer_position'],
        'offers': ['offering', 'giving', 'presenting', 'donation', 'seva'],
        'places': ['placing', 'positioning', 'arranging', 'setting', 'ritual_action'],
        'dancing': ['dancing', 'kirtan', 'ecstatic_movement', 'devotional_dance'],
        'sweeps': ['sweeping', 'cleaning', 'seva', 'service', 'purification'],
        'paints': ['painting', 'artistic_creation', 'devotional_art', 'creative_expression'],
        'decorates': ['decorating', 'adorning', 'beautifying', 'festival_preparation'],
        'wrapped': ['wrapped', 'covered', 'protected', 'embraced', 'sheltered'],
        'naps': ['resting', 'sleeping', 'reposing', 'peaceful_rest', 'tranquil_state'],
        'lies': ['lying_down', 'reclined', 'horizontal_rest', 'supine_position']
    }
    
    # LOCATIONS AND ENVIRONMENTS
    location_keywords = {
        'temple': ['temple', 'shrine', 'sanctuary', 'sacred_space', 'worship_hall'],
        'courtyard': ['courtyard', 'open_space', 'gathering_area', 'community_center'],
        'balcony': ['balcony', 'elevated_platform', 'overview_position', 'raised_area'],
        'rooftop': ['rooftop', 'roof_top', 'elevated_view', 'high_vantage'],
        'cliff': ['cliff', 'precipice', 'rocky_edge', 'dramatic_height', 'overlook'],
        'waterfall': ['waterfall', 'cascade', 'flowing_water', 'natural_fountain'],
        'mountain': ['mountain', 'hill', 'peak', 'elevated_terrain', 'summit'],
        'stream': ['stream', 'brook', 'flowing_water', 'creek', 'rivulet'],
        'garden': ['garden', 'cultivated_space', 'flower_bed', 'botanical_area'],
        'meadow': ['meadow', 'field', 'grassland', 'open_nature', 'pastoral'],
        'cave': ['cave', 'grotto', 'hidden_space', 'underground', 'secluded'],
        'underwater': ['underwater', 'submerged', 'aquatic', 'beneath_surface'],
        'clouds': ['clouds', 'sky', 'atmospheric', 'heavenly', 'celestial'],
        'village': ['village', 'settlement', 'community', 'rural_area', 'hamlet'],
        'lakeside': ['lakeside', 'waterside', 'shore', 'water_edge', 'bank'],
        'treehouse': ['treehouse', 'elevated_dwelling', 'forest_home', 'canopy_space'],
        'boat': ['boat', 'vessel', 'watercraft', 'floating_platform'],
        'interior': ['interior', 'indoor', 'enclosed_space', 'sheltered'],
        'island': ['island', 'isolated_land', 'surrounded_by_water', 'separate_realm'],
        'terrace': ['terrace', 'platform', 'leveled_area', 'outdoor_floor'],
        'banyan': ['banyan_tree', 'sacred_tree', 'ancient_tree', 'spiritual_landmark']
    }
    
    # OBJECTS AND ELEMENTS
    object_keywords = {
        'lamp': ['oil_lamp', 'ghee_lamp', 'diya', 'sacred_light', 'flame'],
        'books': ['books', 'scriptures', 'sacred_texts', 'bhagavad_gita', 'literature'],
        'altar': ['altar', 'shrine', 'sacred_platform', 'worship_center'],
        'cows': ['cows', 'sacred_animals', 'gau_mata', 'cattle', 'bovine'],
        'hammock': ['hammock', 'suspended_bed', 'hanging_rest', 'swing_bed'],
        'futon': ['futon', 'bedding', 'sleeping_mat', 'rest_surface'],
        'shawl': ['shawl', 'wrap', 'covering', 'garment', 'protective_cloth'],
        'plates': ['plates', 'dishes', 'serving_ware', 'food_vessels'],
        'flowers': ['flowers', 'blossoms', 'floral', 'botanical_beauty'],
        'tulasi': ['tulasi', 'holy_basil', 'sacred_plant', 'devotional_flora'],
        'marigold': ['marigold', 'festival_flower', 'decorative_bloom'],
        'cherry_blossom': ['cherry_blossom', 'sakura', 'spring_flower', 'delicate_beauty'],
        'scrolls': ['scrolls', 'manuscripts', 'written_texts', 'documents'],
        'stained_glass': ['stained_glass', 'colored_window', 'artistic_light', 'decorative_glass']
    }
    
    # MOODS AND ATMOSPHERES
    mood_keywords = {
        'peaceful': ['peaceful', 'serene', 'tranquil', 'calm', 'harmonious'],
        'serene': ['serene', 'placid', 'undisturbed', 'still', 'composed'],
        'quiet': ['quiet', 'silent', 'hushed', 'soft', 'gentle'],
        'dramatic': ['dramatic', 'intense', 'powerful', 'striking', 'bold'],
        'gentle': ['gentle', 'soft', 'tender', 'mild', 'soothing'],
        'vibrant': ['vibrant', 'lively', 'energetic', 'colorful', 'dynamic'],
        'cozy': ['cozy', 'comfortable', 'warm', 'intimate', 'homely'],
        'whimsical': ['whimsical', 'playful', 'fantastical', 'magical', 'imaginative'],
        'spirit-filled': ['spirit_filled', 'energized', 'alive', 'animated', 'inspired'],
        'magical': ['magical', 'enchanted', 'mystical', 'otherworldly', 'supernatural'],
        'blazing': ['blazing', 'bright', 'fiery', 'intense_light', 'radiant'],
        'lush': ['lush', 'abundant', 'rich', 'fertile', 'verdant'],
        'crystal-clear': ['crystal_clear', 'transparent', 'pure', 'pristine', 'clarity']
    }
    
    # TIME AND LIGHTING
    time_keywords = {
        'night': ['night', 'evening', 'dark', 'nocturnal', 'after_sunset'],
        'sunrise': ['sunrise', 'dawn', 'morning', 'daybreak', 'early_light'],
        'sunlit': ['sunlit', 'sunny', 'bright', 'illuminated', 'daylight'],
        'filtered': ['filtered_light', 'dappled', 'soft_light', 'gentle_illumination'],
        'glowing': ['glowing', 'luminous', 'radiant', 'emanating_light', 'bright']
    }
    
    # SPIRITUAL AND PHILOSOPHICAL CONCEPTS
    spiritual_keywords = {
        'devotional': ['devotional', 'bhakti', 'loving_service', 'surrender', 'dedication'],
        'sacred': ['sacred', 'holy', 'divine', 'blessed', 'consecrated'],
        'spiritual': ['spiritual', 'transcendent', 'divine', 'soul_focused', 'higher_consciousness'],
        'reverent': ['reverent', 'respectful', 'worshipful', 'devout', 'pious'],
        'contemplative': ['contemplative', 'reflective', 'introspective', 'thoughtful', 'meditative'],
        'procession': ['procession', 'parade', 'ceremonial_walk', 'ritual_movement', 'pilgrimage'],
        'kirtan': ['kirtan', 'devotional_singing', 'communal_chanting', 'spiritual_music'],
        'prayer': ['prayer', 'invocation', 'supplication', 'communion', 'spiritual_communication']
    }
    
    # Combine all keyword dictionaries
    all_keywords = {
        **character_keywords,
        **action_keywords, 
        **location_keywords,
        **object_keywords,
        **mood_keywords,
        **time_keywords,
        **spiritual_keywords
    }
    
    # Analyze filename for content
    for keyword, related_tags in all_keywords.items():
        if keyword.replace('_', ' ') in filename_lower or keyword.replace('_', '') in filename_lower.replace('_', ''):
            tags.extend(related_tags)
    
    # ENHANCED LIGHTING ANALYSIS
    if any(word in filename_lower for word in ['night', 'dark', 'shadow', 'dramatic', 'blazing']):
        shot_analysis['lighting'] = 'dramatic'
        tags.extend(['dramatic_lighting', 'high_contrast', 'mood_lighting'])
    elif any(word in filename_lower for word in ['bright', 'sunrise', 'sunlit', 'glowing', 'crystal']):
        shot_analysis['lighting'] = 'bright'
        tags.extend(['bright_lighting', 'natural_illumination', 'well_lit'])
    elif any(word in filename_lower for word in ['filtered', 'soft', 'gentle']):
        shot_analysis['lighting'] = 'soft'
        tags.extend(['soft_lighting', 'diffused_light', 'gentle_illumination'])
    else:
        shot_analysis['lighting'] = 'natural'
        tags.extend(['natural_lighting', 'ambient_light'])
    
    # ENHANCED CAMERA MOVEMENT ANALYSIS
    if any(word in filename_lower for word in ['floating', 'flying', 'rapidly', 'pans', 'dynamic']):
        shot_analysis['camera_movement'] = 'dynamic'
        tags.extend(['dynamic_movement', 'fluid_motion', 'kinetic'])
    elif any(word in filename_lower for word in ['sitting', 'meditation', 'peaceful', 'lies', 'kneels']):
        shot_analysis['camera_movement'] = 'static'
        tags.extend(['static_shot', 'stable_frame', 'contemplative_pace'])
    else:
        shot_analysis['camera_movement'] = 'gentle'
        tags.extend(['gentle_movement', 'smooth_motion', 'flowing'])
    
    # ENHANCED SHOT TYPE ANALYSIS
    if any(word in filename_lower for word in ['group', 'crowd', 'procession', 'devotees', 'monks']):
        shot_analysis['shot_type'] = 'wide_shot'
        tags.extend(['wide_framing', 'group_composition', 'establishing_shot'])
    elif any(word in filename_lower for word in ['face', 'close', 'detail', 'eye']):
        shot_analysis['shot_type'] = 'close_up'
        tags.extend(['intimate_framing', 'detail_focus', 'personal_view'])
    else:
        shot_analysis['shot_type'] = 'medium_shot'
        tags.extend(['medium_framing', 'balanced_composition', 'subject_focus'])
    
    # EMOTIONAL TONE ANALYSIS
    emotional_tones = []
    if any(word in filename_lower for word in ['peaceful', 'serene', 'calm', 'tranquil', 'meditation']):
        emotional_tones.extend(['peaceful', 'serene', 'calming', 'meditative'])
    if any(word in filename_lower for word in ['spiritual', 'sacred', 'devotional', 'prayer']):
        emotional_tones.extend(['spiritual', 'sacred', 'devotional', 'transcendent'])
    if any(word in filename_lower for word in ['dramatic', 'cliff', 'blazing', 'intense']):
        emotional_tones.extend(['dramatic', 'intense', 'powerful', 'striking'])
    if any(word in filename_lower for word in ['gentle', 'soft', 'cozy', 'wrapped']):
        emotional_tones.extend(['gentle', 'nurturing', 'comforting', 'protective'])
    if any(word in filename_lower for word in ['vibrant', 'lively', 'dancing', 'festival']):
        emotional_tones.extend(['vibrant', 'joyful', 'celebratory', 'energetic'])
    if any(word in filename_lower for word in ['alone', 'lone', 'solitary', 'individual']):
        emotional_tones.extend(['solitary', 'introspective', 'contemplative', 'individual'])
    if any(word in filename_lower for word in ['group', 'community', 'together', 'gathering']):
        emotional_tones.extend(['communal', 'social', 'collective', 'unified'])
    
    tags.extend(emotional_tones)
    
    # COMPOSITION AND FRAMING
    composition_tags = []
    if any(word in filename_lower for word in ['overlooking', 'atop', 'above', 'high']):
        composition_tags.extend(['elevated_perspective', 'overview', 'commanding_view'])
    if any(word in filename_lower for word in ['beneath', 'under', 'base']):
        composition_tags.extend(['low_angle', 'ground_level', 'upward_view'])
    if any(word in filename_lower for word in ['surrounded', 'among', 'within']):
        composition_tags.extend(['immersive', 'surrounded', 'environmental'])
    if any(word in filename_lower for word in ['floating', 'drifts', 'suspended']):
        composition_tags.extend(['suspended', 'weightless', 'ethereal_placement'])
    
    tags.extend(composition_tags)
    
    # CULTURAL AND RELIGIOUS CONTEXT
    cultural_tags = []
    if any(word in filename_lower for word in ['krishna', 'hare', 'bhagavad', 'gita']):
        cultural_tags.extend(['krishna_consciousness', 'vaishnava', 'vedic', 'hindu_tradition'])
    if any(word in filename_lower for word in ['devotee', 'monk', 'brahmachari']):
        cultural_tags.extend(['monastic_life', 'renunciation', 'spiritual_discipline'])
    if any(word in filename_lower for word in ['temple', 'altar', 'shrine']):
        cultural_tags.extend(['temple_worship', 'ritual_space', 'sacred_architecture'])
    if any(word in filename_lower for word in ['ghee', 'lamp', 'offering']):
        cultural_tags.extend(['ritual_offering', 'puja', 'worship_practice'])
    
    tags.extend(cultural_tags)
    
    # Remove duplicates and organize
    tags = list(set(tags))
    
    # Sort tags by relevance (put more specific tags first)
    primary_tags = [tag for tag in tags if any(keyword in filename_lower for keyword in ['devotee', 'temple', 'meditation', 'spiritual', 'peaceful'])]
    secondary_tags = [tag for tag in tags if tag not in primary_tags]
    
    organized_tags = primary_tags + secondary_tags
    
    # Limit to reasonable number but keep more detail
    final_tags = organized_tags[:25]  # Increased from 10 to 25
    
    return {
        "tags": final_tags,
        "shot_analysis": shot_analysis
    }


def create_mjclip_metadata(clips_directory: Path, 
                          output_file: Path,
                          force_recreate: bool = False) -> bool:
    """
    Create metadata file for MJAnime clips.
    
    Args:
        clips_directory: Directory containing MJAnime video clips
        output_file: Path to output metadata JSON file
        force_recreate: Whether to recreate existing metadata file
        
    Returns:
        True if successful, False otherwise
    """
    logger = logging.getLogger(__name__)
    
    if output_file.exists() and not force_recreate:
        logger.info(f"Metadata file already exists: {output_file}")
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
    
    logger.info(f"Found {len(video_files)} video files")
    
    # Create metadata for each clip
    clips_metadata = []
    
    for i, video_file in enumerate(sorted(video_files), 1):
        logger.info(f"Processing {i}/{len(video_files)}: {video_file.name}")
        
        try:
            # Generate unique ID
            clip_id = hashlib.md5(video_file.name.encode()).hexdigest()[:12]
            
            # Extract video metadata
            video_metadata = extract_video_metadata(video_file)
            
            # Analyze content from filename
            content_analysis = analyze_filename_for_content(video_file.name)
            
            # Create clip metadata
            clip_metadata = {
                "id": clip_id,
                "filename": video_file.name,
                "tags": content_analysis["tags"],
                "duration": video_metadata["duration"],
                "resolution": video_metadata["resolution"],
                "fps": video_metadata["fps"],
                "file_size_mb": video_metadata["file_size_mb"],
                "shot_analysis": content_analysis["shot_analysis"],
                "created_at": datetime.now().isoformat()
            }
            
            clips_metadata.append(clip_metadata)
            
        except Exception as e:
            logger.error(f"Failed to process {video_file}: {e}")
            continue
    
    # Create final metadata structure
    metadata = {
        "metadata_info": {
            "created_at": datetime.now().isoformat(),
            "clips_directory": str(clips_directory),
            "total_clips": len(clips_metadata),
            "generator": "create_mjclip_metadata.py",
            "version": "1.0.0"
        },
        "clips": clips_metadata
    }
    
    # Save metadata file
    try:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Successfully created metadata for {len(clips_metadata)} clips")
        logger.info(f"Metadata saved to: {output_file}")
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"MJANIME METADATA CREATION COMPLETE")
        print(f"{'='*60}")
        print(f"Clips processed: {len(clips_metadata)}")
        print(f"Metadata file: {output_file}")
        
        # Show tag distribution
        all_tags = []
        for clip in clips_metadata:
            all_tags.extend(clip["tags"])
        
        from collections import Counter
        tag_counts = Counter(all_tags)
        
        print(f"\nTop content tags:")
        for tag, count in tag_counts.most_common(10):
            print(f"  {tag}: {count}")
        
        # Show shot analysis distribution
        lighting_types = [clip["shot_analysis"].get("lighting", "unknown") 
                         for clip in clips_metadata]
        lighting_counts = Counter(lighting_types)
        
        print(f"\nLighting distribution:")
        for lighting, count in lighting_counts.items():
            print(f"  {lighting}: {count}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to save metadata file: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Create metadata file for MJAnime video clips",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create metadata for clips in MJAnime/fixed directory
  python create_mjclip_metadata.py /path/to/MJAnime/fixed
  
  # Force recreate existing metadata
  python create_mjclip_metadata.py /path/to/MJAnime/fixed --force
  
  # Custom output location
  python create_mjclip_metadata.py /path/to/MJAnime/fixed --output custom_metadata.json
        """
    )
    
    parser.add_argument("clips_directory", 
                       help="Directory containing MJAnime video clips")
    parser.add_argument("--output", "-o",
                       help="Output metadata file (default: ./mjclip_metadata.json)")
    parser.add_argument("--force", action="store_true",
                       help="Force recreate existing metadata file")
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
        output_file = Path("./mjclip_metadata.json")
    
    # Create metadata
    success = create_mjclip_metadata(
        clips_directory=clips_directory,
        output_file=output_file,
        force_recreate=args.force
    )
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()