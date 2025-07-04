#!/usr/bin/env python3
"""
Test Phase 1: Asset Loading Infrastructure

Test all Phase 1 components: MJAnime loader, script analyzer, music manager, and content database.
"""

import logging
import sys
import time
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import Phase 1 components
from content.mjanime_loader import MJAnimeVideoLoader as MJAnimeLoader
from content.script_analyzer import AudioScriptAnalyzer  
from content.music_manager import MusicManager
from content.content_database import ContentDatabase

def test_mjanime_loader():
    """Test MJAnime video loader"""
    logger.info("=== Testing MJAnime Loader ===")
    
    loader = MJAnimeLoader(
        clips_directory="../MJAnime",
        metadata_file="../MJAnime/metadata_final_clean_shots.json"
    )
    
    # Load clips
    success = loader.load_clips()
    if not success:
        logger.error("Failed to load MJAnime clips")
        return False
    
    # Get statistics
    stats = loader.get_clip_stats()
    logger.info(f"✅ Loaded {stats['total_clips']} clips")
    logger.info(f"✅ Total duration: {stats['total_duration_seconds']:.1f}s")
    logger.info(f"✅ Emotion distribution: {stats['emotion_distribution']}")
    
    # Test emotion filtering
    anxiety_clips = loader.get_clips_by_emotion('anxiety')
    peace_clips = loader.get_clips_by_emotion('peace')
    logger.info(f"✅ Anxiety clips: {len(anxiety_clips)}")
    logger.info(f"✅ Peace clips: {len(peace_clips)}")
    
    # Test random selection
    random_clips = loader.get_random_clips(5)
    logger.info(f"✅ Random clips selected: {len(random_clips)}")
    
    return True

def test_script_analyzer():
    """Test audio script analyzer"""
    logger.info("\\n=== Testing Script Analyzer ===")
    
    analyzer = AudioScriptAnalyzer("../11-scripts-for-tiktok")
    
    # Analyze scripts
    success = analyzer.analyze_scripts()
    if not success:
        logger.error("Failed to analyze scripts")
        return False
    
    # Get statistics  
    stats = analyzer.get_analysis_stats()
    logger.info(f"✅ Analyzed {stats['total_scripts']} scripts")
    logger.info(f"✅ Total size: {stats['total_size_mb']:.1f}MB")
    logger.info(f"✅ Emotion distribution: {stats['emotion_distribution']}")
    
    # Test emotion filtering
    anxiety_scripts = analyzer.get_scripts_by_emotion('anxiety')
    peace_scripts = analyzer.get_scripts_by_emotion('peace')
    logger.info(f"✅ Anxiety scripts: {len(anxiety_scripts)}")
    logger.info(f"✅ Peace scripts: {len(peace_scripts)}")
    
    # Test specific script analysis
    test_script = "anxiety1"
    analysis = analyzer.get_script_analysis(test_script)
    if analysis:
        logger.info(f"✅ {test_script} analysis: {analysis.primary_emotion} (intensity: {analysis.emotional_intensity})")
        matching_emotions = analyzer.get_matching_clips_emotions(test_script)
        logger.info(f"✅ Matching clip emotions: {matching_emotions}")
    
    return True

def test_music_manager():
    """Test music track manager"""
    logger.info("\\n=== Testing Music Manager ===")
    
    manager = MusicManager("../Beanie (Slowed).mp3")
    
    # Load music track
    success = manager.load_music_track()
    if not success:
        logger.error("Failed to load music track")
        return False
    
    # Analyze beats (mock analysis for now)
    success = manager.analyze_beats()
    if not success:
        logger.warning("Beat analysis failed")
    
    # Get track info
    track_info = manager.get_track_info()
    logger.info(f"✅ Music track: {track_info['filename']}")
    logger.info(f"✅ Duration: {track_info['duration']}s")
    logger.info(f"✅ Tempo: {track_info['tempo']} BPM")
    logger.info(f"✅ Beats analyzed: {track_info['beats_analyzed']}")
    
    # Test music segments
    segments = manager.get_music_segments(30.0, count=3)
    logger.info(f"✅ Music segments for 30s video: {len(segments)}")
    
    # Test mixing preparation
    mixing_params = manager.prepare_for_mixing(25.0)
    logger.info(f"✅ Mixing params prepared for 25s video")
    
    return True

def test_content_database():
    """Test integrated content database"""
    logger.info("\\n=== Testing Content Database ===")
    
    database = ContentDatabase(
        clips_directory="../MJAnime",
        metadata_file="../MJAnime/metadata_final_clean_shots.json",
        scripts_directory="../11-scripts-for-tiktok", 
        music_file="../Beanie (Slowed).mp3"
    )
    
    # Load all content
    success = database.load_all_content()
    if not success:
        logger.error("Failed to load content database")
        return False
    
    # Get database statistics
    stats = database.get_database_stats()
    logger.info(f"✅ Database loaded with {stats['clips']['total_clips']} clips, {stats['scripts']['total_scripts']} scripts")
    logger.info(f"✅ Total combinations possible: {stats['total_possible_combinations']}")
    
    # Test content matching
    test_scripts = ["anxiety1", "adhd", "safe1"]
    for script_name in test_scripts:
        matches = database.find_matching_clips(script_name, count=3)
        logger.info(f"✅ {script_name}: Found {len(matches)} matching clips")
        if matches:
            best_match = matches[0]
            logger.info(f"   Best match: {best_match.clip.filename} (score: {best_match.relevance_score:.2f})")
    
    # Test unique sequence generation  
    for script_name in test_scripts[:2]:  # Test first 2 scripts
        sequence = database.generate_unique_sequence(script_name, sequence_length=3)
        logger.info(f"✅ {script_name}: Generated sequence with {len(sequence)} clips")
    
    # Test search functionality
    search_results = database.search_content("meditation", content_type='all')
    logger.info(f"✅ Search for 'meditation': {len(search_results['clips'])} clips, {len(search_results['scripts'])} scripts")
    
    return True

def main():
    """Run all Phase 1 tests"""
    logger.info("Starting Phase 1: Asset Loading Infrastructure Tests")
    
    start_time = time.time()
    tests = [
        ("MJAnime Loader", test_mjanime_loader),
        ("Script Analyzer", test_script_analyzer), 
        ("Music Manager", test_music_manager),
        ("Content Database", test_content_database)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                logger.info(f"✅ {test_name}: PASSED")
                passed += 1
            else:
                logger.error(f"❌ {test_name}: FAILED")
        except Exception as e:
            logger.error(f"❌ {test_name}: ERROR - {e}")
    
    elapsed = time.time() - start_time
    
    logger.info(f"\\n=== Phase 1 Test Results ===")
    logger.info(f"Passed: {passed}/{total}")
    logger.info(f"Success Rate: {passed/total*100:.1f}%")
    logger.info(f"Total Time: {elapsed:.2f}s")
    
    if passed == total:
        logger.info("🎉 Phase 1: Asset Loading Infrastructure - COMPLETE")
        return True
    else:
        logger.error("❌ Phase 1: Some tests failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)