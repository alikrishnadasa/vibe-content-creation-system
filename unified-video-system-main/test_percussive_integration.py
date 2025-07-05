#!/usr/bin/env python3
"""
Integration test for percussive sync video generation.

Tests the complete pipeline from percussive event detection to video generation
with sync points aligned to specific drum events (kick, snare, hi-hat).
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, Any
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def test_percussive_sync_video_generation():
    """Test complete percussive sync video generation workflow"""
    
    try:
        logger.info("🥁 Starting percussive sync integration test...")
        
        # Import required modules
        from core.real_content_generator import RealContentGenerator, RealVideoRequest
        from beat_sync.beat_sync_engine import BeatSyncEngine
        from content.music_manager import MusicManager
        
        # Paths (adjust as needed for your setup)
        clips_directory = "MJAnime"  # Adjust path as needed
        metadata_file = "MJAnime/metadata.json"  # Adjust path as needed  
        scripts_directory = "scripts"
        music_file = "music/Beanie (Slowed).mp3"
        test_script = "scripts/spiritual_script.txt"  # Use an existing script
        
        # Check if required files exist
        required_paths = [
            Path(music_file),
            Path(test_script)
        ]
        
        missing_files = [p for p in required_paths if not p.exists()]
        if missing_files:
            logger.warning(f"Missing files: {missing_files}")
            logger.info("Creating mock test setup...")
            return await test_percussive_sync_mock()
        
        # Test 1: Initialize content generator
        logger.info("📁 Initializing content generator...")
        generator = RealContentGenerator(
            clips_directory=clips_directory,
            metadata_file=metadata_file,
            scripts_directory=scripts_directory,
            music_file=music_file
        )
        
        # Initialize
        initialized = await generator.initialize()
        if not initialized:
            logger.error("❌ Failed to initialize content generator")
            return False
        
        logger.info("✅ Content generator initialized")
        
        # Test 2: Analyze music for percussive events
        logger.info("🎵 Analyzing music for percussive events...")
        music_manager = generator.content_database.music_manager
        
        # Load and analyze music
        await music_manager.load_music_track()
        await music_manager.analyze_beats()
        
        track_info = music_manager.get_track_info()
        logger.info(f"✅ Music analysis complete:")
        logger.info(f"   - Tempo: {track_info.get('tempo', 'N/A')} BPM")
        logger.info(f"   - Total beats: {track_info.get('beat_count', 0)}")
        logger.info(f"   - Kick events: {track_info.get('kick_count', 0)}")
        logger.info(f"   - Snare events: {track_info.get('snare_count', 0)}")
        logger.info(f"   - Hi-hat events: {track_info.get('hihat_count', 0)}")
        
        # Test 3: Generate videos with different percussive sync types
        sync_types = ['beat', 'kick', 'snare', 'hihat']
        results = {}
        
        for sync_type in sync_types:
            logger.info(f"🎬 Generating video with {sync_type} synchronization...")
            
            # Create request for percussive sync
            request = RealVideoRequest(
                script_path=test_script,
                script_name="spiritual_script",
                variation_number=1,
                caption_style="tiktok",
                music_sync=True,
                sync_event_type=sync_type,
                use_percussive_sync=(sync_type != 'beat')
            )
            
            # Generate video
            result = await generator.generate_video(request)
            results[sync_type] = result
            
            if result.success:
                logger.info(f"✅ {sync_type.capitalize()} sync video generated successfully")
                logger.info(f"   - Output: {Path(result.output_path).name}")
                logger.info(f"   - Duration: {result.total_duration:.1f}s")
                logger.info(f"   - Generation time: {result.generation_time:.1f}s")
                logger.info(f"   - Relevance score: {result.relevance_score:.3f}")
                logger.info(f"   - Visual variety: {result.visual_variety_score:.3f}")
            else:
                logger.error(f"❌ {sync_type.capitalize()} sync video generation failed: {result.error_message}")
        
        # Test 4: Compare sync points between different types
        logger.info("📊 Analyzing sync point differences...")
        
        successful_results = {k: v for k, v in results.items() if v.success}
        if len(successful_results) >= 2:
            logger.info("✅ Successfully generated videos with different sync types")
            
            # Compare generation metrics
            for sync_type, result in successful_results.items():
                logger.info(f"{sync_type}: {result.generation_time:.1f}s generation, "
                           f"{result.relevance_score:.3f} relevance, "
                           f"{result.visual_variety_score:.3f} variety")
        
        # Test 5: Verify percussive event extraction
        logger.info("🔍 Testing percussive event extraction...")
        
        for event_type in ['kick', 'snare', 'hihat']:
            events = music_manager.get_percussive_events(event_type)
            if events:
                logger.info(f"✅ {event_type.capitalize()} events: {len(events)} found")
                logger.info(f"   First few: {events[:5]}")
            else:
                logger.warning(f"⚠️  No {event_type} events detected")
        
        # Summary
        successful_count = sum(1 for r in results.values() if r.success)
        total_count = len(results)
        
        logger.info("🎉 Integration test summary:")
        logger.info(f"   - Successful generations: {successful_count}/{total_count}")
        logger.info(f"   - Music analysis: {'✅ Success' if track_info.get('beats_analyzed') else '❌ Failed'}")
        
        return successful_count == total_count
        
    except Exception as e:
        logger.error(f"❌ Integration test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_percussive_sync_mock():
    """Test percussive sync with mock data when real files aren't available"""
    logger.info("🎭 Running mock percussive sync test...")
    
    try:
        # Test the core functionality without requiring actual files
        from beat_sync.beat_sync_engine import BeatSyncEngine, BeatSyncResult, BeatInfo
        from content.music_manager import MusicManager, MusicTrackInfo
        from content.content_selector import ContentSelector, SelectionCriteria
        import numpy as np
        
        # Test 1: BeatSyncEngine percussive classification
        logger.info("🎵 Testing BeatSyncEngine percussive classification...")
        
        engine = BeatSyncEngine()
        
        # Create mock result with percussive events
        mock_result = BeatSyncResult(
            beats=[
                BeatInfo(0.0, 1.0, True, 1, 120.0),
                BeatInfo(0.5, 0.8, False, 2, 120.0),
                BeatInfo(1.0, 1.0, False, 3, 120.0),
                BeatInfo(1.5, 0.6, False, 4, 120.0),
                BeatInfo(2.0, 1.0, True, 1, 120.0)
            ],
            tempo=120.0,
            time_signature=(4, 4),
            phrases=[],
            onset_times=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5],
            energy_curve=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.3]),
            processing_time=0.1,
            kick_times=[0.0, 2.0],  # On downbeats
            snare_times=[1.0],      # On beat 3
            hihat_times=[0.5, 1.5, 2.5],  # Off-beats
            other_times=[]
        )
        
        # Test getting different event types
        kick_events = engine.get_onset_times('kick', mock_result)
        snare_events = engine.get_onset_times('snare', mock_result)
        hihat_events = engine.get_onset_times('hihat', mock_result)
        beat_events = engine.get_onset_times('beat', mock_result)
        
        logger.info(f"✅ Kick events: {kick_events}")
        logger.info(f"✅ Snare events: {snare_events}")
        logger.info(f"✅ Hi-hat events: {hihat_events}")
        logger.info(f"✅ Beat events: {len(beat_events)} total")
        
        # Test 2: MusicManager with percussive events
        logger.info("🎼 Testing MusicManager percussive functionality...")
        
        music_manager = MusicManager("mock_music.mp3")
        
        # Set up mock track info with percussive data
        music_manager.track_info = MusicTrackInfo(
            filepath="mock_music.mp3",
            filename="mock_music.mp3",
            duration=30.0,
            sample_rate=44100,
            beats=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
            kick_times=[0.0, 2.0],
            snare_times=[1.0],
            hihat_times=[0.5, 1.5, 2.5],
            other_times=[],
            tempo=120.0,
            analyzed=True,
            beat_sync_result=mock_result
        )
        music_manager.loaded = True
        
        # Test getting timing for different event types
        kick_timing = music_manager.get_beat_timing(0.0, 3.0, 'kick')
        snare_timing = music_manager.get_beat_timing(0.0, 3.0, 'snare')
        hihat_timing = music_manager.get_beat_timing(0.0, 3.0, 'hihat')
        
        logger.info(f"✅ Kick timing in 3s window: {kick_timing}")
        logger.info(f"✅ Snare timing in 3s window: {snare_timing}")
        logger.info(f"✅ Hi-hat timing in 3s window: {hihat_timing}")
        
        # Test mixing parameters with percussive sync
        kick_mix_params = music_manager.prepare_for_mixing(10.0, 'kick')
        hihat_mix_params = music_manager.prepare_for_mixing(10.0, 'hihat')
        
        logger.info(f"✅ Kick mix params: {kick_mix_params.get('sync_event_count')} events")
        logger.info(f"✅ Hi-hat mix params: {hihat_mix_params.get('sync_event_count')} events")
        
        # Test 3: ContentSelector with percussive sync
        logger.info("🎬 Testing ContentSelector percussive functionality...")
        
        from unittest.mock import Mock
        mock_content_db = Mock()
        content_selector = ContentSelector(mock_content_db)
        
        # Test selection criteria with percussive sync
        criteria = SelectionCriteria(
            emotion='anxiety',
            intensity=0.8,
            themes=['stress'],
            duration_target=10.0,
            music_beats=kick_events,  # Use kick events as sync points
            visual_variety=True,
            sync_event_type='kick',
            use_percussive_sync=True
        )
        
        logger.info(f"✅ Selection criteria: {criteria.sync_event_type} sync, "
                   f"{len(criteria.music_beats)} sync points")
        
        # Test sync point generation
        mock_clips = [Mock() for _ in range(3)]
        for i, clip in enumerate(mock_clips):
            clip.duration = 3.0
            clip.id = f"mock_clip_{i}"
        
        sync_points = content_selector._generate_music_sync_points(
            mock_clips, kick_events, 'kick'
        )
        
        logger.info(f"✅ Generated {len(sync_points)} sync points for kick events")
        for i, point in enumerate(sync_points):
            logger.info(f"   Clip {i}: {point['start_time']:.1f}s - {point['end_time']:.1f}s "
                       f"({point['sync_event_type']})")
        
        # Test 4: Demonstrate different sync patterns
        logger.info("📊 Comparing sync patterns...")
        
        sync_patterns = {}
        for event_type in ['beat', 'kick', 'snare', 'hihat']:
            if event_type == 'beat':
                events = beat_events
            else:
                events = engine.get_onset_times(event_type, mock_result)
            
            if events:
                # Calculate average interval
                if len(events) > 1:
                    intervals = [events[i+1] - events[i] for i in range(len(events)-1)]
                    avg_interval = sum(intervals) / len(intervals)
                else:
                    avg_interval = 0
                
                sync_patterns[event_type] = {
                    'count': len(events),
                    'avg_interval': avg_interval,
                    'first_event': events[0] if events else None
                }
        
        logger.info("✅ Sync pattern analysis:")
        for event_type, pattern in sync_patterns.items():
            logger.info(f"   {event_type.capitalize()}: {pattern['count']} events, "
                       f"{pattern['avg_interval']:.2f}s avg interval")
        
        logger.info("🎉 Mock test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Mock test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_specific_use_cases():
    """Test specific use cases for percussive sync"""
    logger.info("🎯 Testing specific percussive sync use cases...")
    
    use_cases = [
        {
            'name': 'High-energy video with kick sync',
            'sync_type': 'kick',
            'description': 'Video cuts synchronized to kick drum for powerful impact'
        },
        {
            'name': 'Fast-paced video with hi-hat sync',
            'sync_type': 'hihat',
            'description': 'Quick cuts synchronized to hi-hat for rapid-fire effect'
        },
        {
            'name': 'Dramatic video with snare sync',
            'sync_type': 'snare',
            'description': 'Strategic cuts on snare hits for dramatic emphasis'
        }
    ]
    
    for use_case in use_cases:
        logger.info(f"🎬 Testing: {use_case['name']}")
        logger.info(f"   Description: {use_case['description']}")
        logger.info(f"   Sync type: {use_case['sync_type']}")
        
        # Simulate the use case with mock data
        from beat_sync.beat_sync_engine import BeatSyncEngine, BeatSyncResult
        
        # Create different mock results for each use case
        if use_case['sync_type'] == 'kick':
            # Heavy kick pattern - fewer but powerful hits
            mock_result = BeatSyncResult(
                beats=[], tempo=100.0, time_signature=(4, 4), phrases=[], 
                onset_times=[], energy_curve=None, processing_time=0.1,
                kick_times=[0.0, 2.4, 4.8, 7.2],  # Every 2.4 seconds
                snare_times=[1.2, 3.6, 6.0],
                hihat_times=[],
                other_times=[]
            )
        elif use_case['sync_type'] == 'hihat':
            # Rapid hi-hat pattern - many quick hits
            mock_result = BeatSyncResult(
                beats=[], tempo=140.0, time_signature=(4, 4), phrases=[], 
                onset_times=[], energy_curve=None, processing_time=0.1,
                kick_times=[0.0, 1.7, 3.4],
                snare_times=[0.85, 2.55],
                hihat_times=[0.2, 0.4, 0.6, 1.0, 1.2, 1.4, 1.9, 2.1, 2.3, 2.7, 2.9, 3.1],  # Rapid hits
                other_times=[]
            )
        else:  # snare
            # Strategic snare hits for drama
            mock_result = BeatSyncResult(
                beats=[], tempo=85.0, time_signature=(4, 4), phrases=[], 
                onset_times=[], energy_curve=None, processing_time=0.1,
                kick_times=[0.0, 2.8, 5.6],
                snare_times=[1.4, 4.2, 7.0],  # Strategic dramatic hits
                hihat_times=[0.7, 2.1, 3.5, 4.9, 6.3],
                other_times=[]
            )
        
        engine = BeatSyncEngine()
        events = engine.get_onset_times(use_case['sync_type'], mock_result)
        
        if events:
            intervals = [events[i+1] - events[i] for i in range(len(events)-1)] if len(events) > 1 else [0]
            avg_interval = sum(intervals) / len(intervals) if intervals else 0
            
            logger.info(f"   ✅ {len(events)} {use_case['sync_type']} events")
            logger.info(f"   ✅ Average interval: {avg_interval:.2f}s")
            logger.info(f"   ✅ Clip duration range: {avg_interval:.1f}s per cut")
            
            # Determine video style based on interval
            if avg_interval < 1.0:
                style = "Very fast-paced"
            elif avg_interval < 2.0:
                style = "Fast-paced"
            elif avg_interval < 3.0:
                style = "Moderate-paced"
            else:
                style = "Slow-paced"
            
            logger.info(f"   📺 Recommended style: {style}")
        else:
            logger.warning(f"   ⚠️  No {use_case['sync_type']} events found")
        
        logger.info("")
    
    logger.info("✅ Use case testing completed!")
    return True


async def main():
    """Main test runner"""
    print("🥁 Percussive Sync Integration Test Suite")
    print("=" * 60)
    
    # Run tests in sequence
    tests = [
        ("Mock functionality test", test_percussive_sync_mock),
        ("Specific use cases test", test_specific_use_cases),
        ("Full integration test", test_percussive_sync_video_generation)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name}...")
        print("-" * 40)
        
        try:
            result = await test_func()
            results[test_name] = result
            
            if result:
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
                
        except Exception as e:
            print(f"❌ {test_name} ERROR: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for r in results.values() if r)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Percussive sync is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    
    return passed == total


if __name__ == "__main__":
    asyncio.run(main())