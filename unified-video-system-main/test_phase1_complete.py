#!/usr/bin/env python3
"""
Test Phase 1: Complete Asset Loading Infrastructure

Tests all Phase 1 components:
- MJAnime Video Loader
- Music Manager 
- Audio Script Analyzer
- Content Database Integration
"""

import logging
import asyncio
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from content.mjanime_loader import MJAnimeVideoLoader as MJAnimeLoader
from content.music_manager import MusicManager
from content.script_analyzer import AudioScriptAnalyzer

async def test_phase1_complete():
    """Test complete Phase 1 implementation"""
    logger.info("=== Phase 1: Asset Loading Infrastructure Test ===")
    
    success = True
    
    # Test 1: MJAnime Video Loader
    logger.info("\n--- Test 1: MJAnime Video Loader ---")
    try:
        loader = MJAnimeLoader(
            clips_directory="../MJAnime",
            metadata_file="../MJAnime/metadata_final_clean_shots.json"
        )
        
        load_success = await loader.load_clips()
        if load_success:
            stats = loader.get_clip_stats()
            logger.info(f"✅ Loaded {stats['total_clips']} clips")
            logger.info(f"   Total duration: {stats['total_duration_seconds']:.1f}s")
            logger.info(f"   Total size: {stats['total_size_mb']:.1f}MB")
            logger.info(f"   Emotion distribution: {stats['emotion_distribution']}")
            
            # Test emotional search
            anxiety_clips = loader.get_clips_by_emotion('anxiety')
            peace_clips = loader.get_clips_by_emotion('peace')
            logger.info(f"   Found {len(anxiety_clips)} anxiety clips, {len(peace_clips)} peace clips")
        else:
            logger.error("❌ Failed to load MJAnime clips")
            success = False
            
    except Exception as e:
        logger.error(f"❌ MJAnime loader test failed: {e}")
        success = False
    
    # Test 2: Music Manager
    logger.info("\n--- Test 2: Music Manager ---")
    try:
        music_manager = MusicManager("../Beanie (Slowed).mp3")
        
        load_success = await music_manager.load_music_track()
        if load_success:
            await music_manager.analyze_beats()
            track_info = music_manager.get_track_info()
            logger.info(f"✅ Music track loaded: {track_info['filename']}")
            logger.info(f"   Duration: {track_info['duration']:.1f}s")
            logger.info(f"   Tempo: {track_info['tempo']:.1f} BPM")
            logger.info(f"   Beats: {track_info['beat_count']}")
            
            # Test segment generation
            segments = music_manager.get_music_segments(15.0, 3)
            logger.info(f"   Generated {len(segments)} segments for 15s videos")
        else:
            logger.error("❌ Failed to load music track")
            success = False
            
    except Exception as e:
        logger.error(f"❌ Music manager test failed: {e}")
        success = False
    
    # Test 3: Audio Script Analyzer
    logger.info("\n--- Test 3: Audio Script Analyzer ---")
    try:
        script_analyzer = AudioScriptAnalyzer("../11-scripts-for-tiktok")
        
        analyze_success = await script_analyzer.analyze_scripts()
        if analyze_success:
            stats = script_analyzer.get_analysis_stats()
            logger.info(f"✅ Analyzed {stats['total_scripts']} scripts")
            logger.info(f"   Total size: {stats['total_size_mb']:.1f}MB")
            logger.info(f"   Average intensity: {stats['average_intensity']:.2f}")
            logger.info(f"   Emotion distribution: {stats['emotion_distribution']}")
            
            # Test specific script analysis
            anxiety_analysis = script_analyzer.get_script_analysis('anxiety1')
            if anxiety_analysis:
                logger.info(f"   Anxiety script: {anxiety_analysis.primary_emotion}, intensity {anxiety_analysis.emotional_intensity}")
            
            safe_analysis = script_analyzer.get_script_analysis('safe1')
            if safe_analysis:
                logger.info(f"   Safe script: {safe_analysis.primary_emotion}, intensity {safe_analysis.emotional_intensity}")
        else:
            logger.error("❌ Failed to analyze scripts")
            success = False
            
    except Exception as e:
        logger.error(f"❌ Script analyzer test failed: {e}")
        success = False
    
    # Test 4: Integration Test - Match Scripts to Clips
    logger.info("\n--- Test 4: Script-to-Clip Matching ---")
    try:
        if load_success and analyze_success:
            # Test anxiety script matching
            anxiety_analysis = script_analyzer.get_script_analysis('anxiety1')
            if anxiety_analysis:
                matching_emotions = script_analyzer.get_matching_clips_emotions('anxiety1')
                logger.info(f"Anxiety script matches emotions: {matching_emotions}")
                
                for emotion in matching_emotions:
                    matching_clips = loader.get_clips_by_emotion(emotion)
                    logger.info(f"   Found {len(matching_clips)} {emotion} clips for anxiety script")
            
            # Test safe script matching  
            safe_analysis = script_analyzer.get_script_analysis('safe1')
            if safe_analysis:
                matching_emotions = script_analyzer.get_matching_clips_emotions('safe1')
                logger.info(f"Safe script matches emotions: {matching_emotions}")
                
                for emotion in matching_emotions:
                    matching_clips = loader.get_clips_by_emotion(emotion)
                    logger.info(f"   Found {len(matching_clips)} {emotion} clips for safe script")
        else:
            logger.warning("⚠️  Skipping integration test due to previous failures")
            
    except Exception as e:
        logger.error(f"❌ Integration test failed: {e}")
        success = False
    
    # Summary
    logger.info("\n=== Phase 1 Test Summary ===")
    if success:
        logger.info("✅ Phase 1: Asset Loading Infrastructure - COMPLETE")
        logger.info("   • MJAnime clips loaded and indexed")
        logger.info("   • Universal music track analyzed")
        logger.info("   • Audio scripts analyzed for emotions")
        logger.info("   • Integration between components working")
        logger.info("\n🚀 Ready for Phase 2: Content Intelligence Engine")
    else:
        logger.error("❌ Phase 1: Asset Loading Infrastructure - FAILED")
        logger.error("   Fix the issues above before proceeding to Phase 2")
    
    return success

if __name__ == "__main__":
    success = asyncio.run(test_phase1_complete())
    if success:
        print("\n🎉 Phase 1 implementation test completed successfully!")
    else:
        print("\n💥 Phase 1 implementation test failed")
        sys.exit(1)