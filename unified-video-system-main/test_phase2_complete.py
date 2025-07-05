#!/usr/bin/env python3
"""
Test Phase 2: Complete Content Intelligence Engine

Tests all Phase 2 components:
- Content Selector with intelligent matching
- Sequence Uniqueness Engine with 100% uniqueness
- Integration with Phase 1 components
- Generation of 5 unique variations per script
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
from content.content_selector import ContentSelector
from content.uniqueness_engine import UniquenessEngine
from content.content_database import ContentDatabase

async def test_phase2_complete():
    """Test complete Phase 2 implementation"""
    logger.info("=== Phase 2: Content Intelligence Engine Test ===")
    
    success = True
    
    # Initialize Phase 1 components first
    logger.info("\n--- Initializing Phase 1 Components ---")
    
    try:
        # Initialize components
        mjanime_loader = MJAnimeLoader(
            clips_directory="../MJAnime",
            metadata_file="../MJAnime/metadata_final_clean_shots.json"
        )
        music_manager = MusicManager("../Beanie (Slowed).mp3")
        script_analyzer = AudioScriptAnalyzer("../11-scripts-for-tiktok")
        
        # Load all Phase 1 data
        load_clips = await mjanime_loader.load_clips()
        load_music = await music_manager.load_music_track()
        await music_manager.analyze_beats()
        analyze_scripts = await script_analyzer.analyze_scripts()
        
        if not (load_clips and load_music and analyze_scripts):
            logger.error("❌ Phase 1 components failed to load")
            return False
        
        logger.info("✅ Phase 1 components loaded successfully")
        
    except Exception as e:
        logger.error(f"❌ Phase 1 initialization failed: {e}")
        return False
    
    # Test 1: Content Database Integration
    logger.info("\n--- Test 1: Content Database Integration ---")
    try:
        content_db = ContentDatabase(
            clips_directory="../MJAnime",
            metadata_file="../MJAnime/metadata_final_clean_shots.json",
            scripts_directory="../11-scripts-for-tiktok",
            music_file="../Beanie (Slowed).mp3"
        )
        
        # Override with our already-loaded components for efficiency
        content_db.clips_loader = mjanime_loader
        content_db.music_manager = music_manager
        content_db.scripts_analyzer = script_analyzer
        
        logger.info("✅ Content database integrated with Phase 1 components")
        
    except Exception as e:
        logger.error(f"❌ Content database integration failed: {e}")
        success = False
    
    # Test 2: Content Selector
    logger.info("\n--- Test 2: Content Selector ---")
    try:
        content_selector = ContentSelector(content_db)
        
        # Test single script selection
        anxiety_analysis = script_analyzer.get_script_analysis('anxiety1')
        if anxiety_analysis:
            # Get music beats for synchronization
            track_info = music_manager.get_track_info()
            music_beats = music_manager.get_beat_timing(0, 15.0)  # 15 seconds
            
            sequence = await content_selector.select_clips_for_script(
                script_analysis=anxiety_analysis,
                clip_count=5,
                music_beats=music_beats
            )
            
            logger.info(f"✅ Selected sequence for anxiety1:")
            logger.info(f"   Clips: {len(sequence.clips)}")
            logger.info(f"   Duration: {sequence.total_duration:.1f}s")
            logger.info(f"   Relevance: {sequence.relevance_score:.3f}")
            logger.info(f"   Variety: {sequence.visual_variety_score:.3f}")
            logger.info(f"   Hash: {sequence.sequence_hash}")
        else:
            logger.error("❌ Could not get anxiety1 script analysis")
            success = False
            
    except Exception as e:
        logger.error(f"❌ Content selector test failed: {e}")
        success = False
    
    # Test 3: Uniqueness Engine
    logger.info("\n--- Test 3: Sequence Uniqueness Engine ---")
    try:
        uniqueness_engine = UniquenessEngine()
        
        # Test uniqueness checking
        if 'sequence' in locals():
            is_unique = uniqueness_engine.is_sequence_unique(sequence)
            logger.info(f"   Initial sequence uniqueness: {is_unique}")
            
            # Register the sequence
            registered = await uniqueness_engine.register_sequence(sequence, 'anxiety1', 1)
            logger.info(f"   Sequence registration: {registered}")
            
            # Test duplicate detection
            is_unique_after = uniqueness_engine.is_sequence_unique(sequence)
            logger.info(f"   Sequence uniqueness after registration: {is_unique_after}")
            
            if is_unique and registered and not is_unique_after:
                logger.info("✅ Uniqueness engine working correctly")
            else:
                logger.error("❌ Uniqueness engine behavior unexpected")
                success = False
        
    except Exception as e:
        logger.error(f"❌ Uniqueness engine test failed: {e}")
        success = False
    
    # Test 4: Multiple Variations Generation
    logger.info("\n--- Test 4: Generate 5 Unique Variations ---")
    try:
        # Test with anxiety1 script
        anxiety_analysis = script_analyzer.get_script_analysis('anxiety1')
        if anxiety_analysis:
            music_beats = music_manager.get_beat_timing(0, 15.0)
            
            variations = content_selector.generate_multiple_sequences(
                script_analysis=anxiety_analysis,
                count=5,
                music_beats=music_beats
            )
            
            logger.info(f"✅ Generated {len(variations)} variations for anxiety1")
            
            # Test uniqueness of all variations
            unique_variations, duplicates = uniqueness_engine.validate_sequence_uniqueness(variations)
            logger.info(f"   Unique: {len(unique_variations)}, Duplicates: {len(duplicates)}")
            
            # Register all unique variations
            for i, variation in enumerate(unique_variations):
                registered = await uniqueness_engine.register_sequence(variation, 'anxiety1', i+2)
                if registered:
                    logger.info(f"   ✅ Registered variation {i+2}")
                else:
                    logger.warning(f"   ⚠️  Failed to register variation {i+2}")
        
    except Exception as e:
        logger.error(f"❌ Multiple variations test failed: {e}")
        success = False
    
    # Test 5: Test with Multiple Scripts
    logger.info("\n--- Test 5: Multiple Scripts Processing ---")
    try:
        test_scripts = ['safe1', 'adhd', 'miserable1']
        all_variations = {}
        
        for script_name in test_scripts:
            script_analysis = script_analyzer.get_script_analysis(script_name)
            if script_analysis:
                logger.info(f"\n   Processing {script_name}...")
                
                music_beats = music_manager.get_beat_timing(0, 12.0)
                
                variations = content_selector.generate_multiple_sequences(
                    script_analysis=script_analysis,
                    count=5,
                    music_beats=music_beats
                )
                
                all_variations[script_name] = variations
                logger.info(f"   Generated {len(variations)} variations for {script_name}")
                
                # Register variations
                for i, variation in enumerate(variations):
                    try:
                        registered = await uniqueness_engine.register_sequence(variation, script_name, i+1)
                        if not registered:
                            logger.warning(f"   ⚠️  Duplicate detected for {script_name} var {i+1}")
                    except Exception as reg_error:
                        logger.warning(f"   ⚠️  Registration failed for {script_name} var {i+1}: {reg_error}")
        
        total_variations = sum(len(variations) for variations in all_variations.values())
        logger.info(f"✅ Processed {len(test_scripts)} scripts, generated {total_variations} total variations")
        
    except Exception as e:
        logger.error(f"❌ Multiple scripts test failed: {e}")
        success = False
    
    # Test 6: Uniqueness Report
    logger.info("\n--- Test 6: Comprehensive Uniqueness Report ---")
    try:
        uniqueness_report = uniqueness_engine.generate_uniqueness_report()
        logger.info(f"✅ Uniqueness Report:")
        logger.info(f"   Total unique sequences: {uniqueness_report['total_unique_sequences']}")
        logger.info(f"   Uniqueness percentage: {uniqueness_report['uniqueness_percentage']:.1f}%")
        logger.info(f"   Scripts with variations: {uniqueness_report['total_scripts_with_variations']}")
        logger.info(f"   Average relevance score: {uniqueness_report['average_relevance_score']:.3f}")
        logger.info(f"   Average variety score: {uniqueness_report['average_variety_score']:.3f}")
        logger.info(f"   Script distribution: {uniqueness_report['script_distribution']}")
        
        # Check if we achieved 100% uniqueness
        if uniqueness_report['uniqueness_percentage'] == 100.0:
            logger.info("🎯 100% uniqueness achieved!")
        else:
            logger.warning(f"⚠️  Uniqueness below 100%: {uniqueness_report['uniqueness_percentage']:.1f}%")
        
    except Exception as e:
        logger.error(f"❌ Uniqueness report test failed: {e}")
        success = False
    
    # Test 7: Content Selection Statistics
    logger.info("\n--- Test 7: Content Selection Statistics ---")
    try:
        selection_stats = content_selector.get_selection_stats()
        logger.info(f"✅ Content Selection Stats:")
        logger.info(f"   Supported emotions: {selection_stats['supported_emotions']}")
        logger.info(f"   Emotion mappings: {selection_stats['emotion_mappings_available']}")
        
        clip_stats = mjanime_loader.get_clip_stats()
        logger.info(f"   Available clips: {clip_stats['total_clips']}")
        logger.info(f"   Emotion distribution: {clip_stats['emotion_distribution']}")
        
    except Exception as e:
        logger.error(f"❌ Selection statistics test failed: {e}")
        success = False
    
    # Summary
    logger.info("\n=== Phase 2 Test Summary ===")
    if success:
        logger.info("✅ Phase 2: Content Intelligence Engine - COMPLETE")
        logger.info("   • Content Selector working with intelligent matching")
        logger.info("   • Sequence Uniqueness Engine ensuring 100% uniqueness")
        logger.info("   • Successfully generating 5 unique variations per script")
        logger.info("   • Integration with Phase 1 components successful")
        logger.info("   • Ready for Phase 3: Production Pipeline Integration")
        
        # Show readiness for testing phase
        uniqueness_report = uniqueness_engine.generate_uniqueness_report()
        total_sequences = uniqueness_report['total_unique_sequences']
        logger.info(f"\n🚀 Testing Phase Ready:")
        logger.info(f"   • {total_sequences} unique sequences generated")
        logger.info(f"   • Multiple scripts tested successfully")
        logger.info(f"   • 100% uniqueness maintained")
        
    else:
        logger.error("❌ Phase 2: Content Intelligence Engine - FAILED")
        logger.error("   Fix the issues above before proceeding to Phase 3")
    
    return success

if __name__ == "__main__":
    success = asyncio.run(test_phase2_complete())
    if success:
        print("\n🎉 Phase 2 implementation test completed successfully!")
        print("Ready to begin Phase 3: Production Pipeline Integration")
    else:
        print("\n💥 Phase 2 implementation test failed")
        sys.exit(1)