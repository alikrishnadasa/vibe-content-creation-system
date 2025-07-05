#!/usr/bin/env python3
"""
Test script for percussive sync implementation.

Tests the onset classification and percussive event detection functionality
added to the BeatSyncEngine for syncing video cuts to specific drum events.
"""

import asyncio
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
import unittest
from unittest.mock import Mock, patch, MagicMock

# Import the modules we're testing
from beat_sync.beat_sync_engine import BeatSyncEngine, BeatSyncResult
from content.music_manager import MusicManager
from content.content_selector import ContentSelector, SelectionCriteria
from core.real_content_generator import RealContentGenerator, RealVideoRequest

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestPercussiveSync(unittest.TestCase):
    """Test cases for percussive sync functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.beat_sync_engine = BeatSyncEngine()
        self.music_file_path = Path("music/Beanie (Slowed).mp3")
        
        # Create mock data for testing
        self.mock_onset_times = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
        self.mock_audio_data = np.random.randn(44100 * 5)  # 5 seconds of mock audio
        
    def test_get_onset_times_method(self):
        """Test the get_onset_times method with different event types"""
        # Create mock BeatSyncResult
        mock_result = BeatSyncResult(
            beats=[],
            tempo=120.0,
            time_signature=(4, 4),
            phrases=[],
            onset_times=[0.5, 1.0, 1.5, 2.0],
            energy_curve=np.array([0.1, 0.2, 0.3, 0.4]),
            processing_time=0.1,
            kick_times=[0.5, 2.0],
            snare_times=[1.0],
            hihat_times=[1.5],
            other_times=[]
        )
        
        # Test different event types
        kick_times = self.beat_sync_engine.get_onset_times('kick', mock_result)
        self.assertEqual(kick_times, [0.5, 2.0])
        
        snare_times = self.beat_sync_engine.get_onset_times('snare', mock_result)
        self.assertEqual(snare_times, [1.0])
        
        hihat_times = self.beat_sync_engine.get_onset_times('hihat', mock_result)
        self.assertEqual(hihat_times, [1.5])
        
        # Test invalid event type
        with self.assertRaises(ValueError):
            self.beat_sync_engine.get_onset_times('invalid', mock_result)
    
    @patch('librosa.load')
    @patch('librosa.feature.spectral_centroid')
    @patch('librosa.feature.spectral_rolloff')
    @patch('librosa.feature.zero_crossing_rate')
    @patch('librosa.stft')
    @patch('librosa.fft_frequencies')
    async def test_classify_percussive_events(self, mock_fft_freqs, mock_stft, 
                                            mock_zcr, mock_rolloff, mock_centroid, mock_load):
        """Test percussive event classification"""
        # Mock librosa functions
        mock_load.return_value = (self.mock_audio_data, 44100)
        mock_centroid.return_value = np.array([[100.0, 2500.0, 200.0, 4000.0]])  # Different centroids for classification
        mock_rolloff.return_value = np.array([[800.0, 3000.0, 1500.0, 5000.0]])
        mock_zcr.return_value = np.array([[0.05, 0.15, 0.08, 0.2]])
        
        # Mock STFT and frequency data
        mock_stft.return_value = np.random.randn(1025, 10) + 1j * np.random.randn(1025, 10)
        mock_fft_freqs.return_value = np.linspace(0, 22050, 1025)
        
        # Test classification
        kick_times, snare_times, hihat_times, other_times = await self.beat_sync_engine._classify_percussive_events(
            Path("test_audio.mp3"), self.mock_audio_data, self.mock_onset_times[:4]
        )
        
        # Verify that events were classified (exact classification depends on mock data)
        total_classified = len(kick_times) + len(snare_times) + len(hihat_times) + len(other_times)
        self.assertEqual(total_classified, 4)  # Should classify all 4 mock onsets
        
        # Verify all returned times are from the original onset times
        all_classified_times = kick_times + snare_times + hihat_times + other_times
        for time in all_classified_times:
            self.assertIn(time, self.mock_onset_times[:4])


class TestMusicManagerPercussiveSync(unittest.TestCase):
    """Test MusicManager percussive sync functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.music_manager = MusicManager("music/Beanie (Slowed).mp3")
        
        # Mock track info with percussive events
        self.music_manager.track_info = Mock()
        self.music_manager.track_info.analyzed = True
        self.music_manager.track_info.duration = 180.0
        self.music_manager.track_info.beats = [0.0, 0.7, 1.4, 2.1, 2.8, 3.5]
        self.music_manager.track_info.kick_times = [0.0, 2.1]
        self.music_manager.track_info.snare_times = [0.7, 2.8]
        self.music_manager.track_info.hihat_times = [1.4, 3.5]
        self.music_manager.track_info.other_times = []
        self.music_manager.loaded = True
        
    def test_get_beat_timing_with_event_types(self):
        """Test getting beat timing for different percussive event types"""
        # Test kick events
        kick_events = self.music_manager.get_beat_timing(0.0, 3.0, 'kick')
        self.assertEqual(kick_events, [0.0, 2.1])
        
        # Test snare events
        snare_events = self.music_manager.get_beat_timing(0.0, 3.0, 'snare')
        self.assertEqual(snare_events, [0.7, 2.8])
        
        # Test hihat events
        hihat_events = self.music_manager.get_beat_timing(0.0, 3.0, 'hihat')
        self.assertEqual(hihat_events, [1.4])  # 3.5 is outside range
        
        # Test regular beats
        beat_events = self.music_manager.get_beat_timing(0.0, 3.0, 'beat')
        self.assertEqual(beat_events, [0.0, 0.7, 1.4, 2.1, 2.8])
        
    def test_prepare_for_mixing_with_percussive_sync(self):
        """Test preparing mixing parameters with percussive sync"""
        mixing_params = self.music_manager.prepare_for_mixing(10.0, 'kick')
        
        self.assertEqual(mixing_params['sync_event_type'], 'kick')
        self.assertIn('sync_events', mixing_params)
        self.assertIn('sync_event_count', mixing_params)
        
    def test_get_percussive_events(self):
        """Test getting all percussive events of a specific type"""
        # Mock beat sync result
        mock_beat_sync_result = Mock()
        self.music_manager.track_info.beat_sync_result = mock_beat_sync_result
        
        with patch.object(self.music_manager.beat_sync_engine, 'get_onset_times') as mock_get_onset:
            mock_get_onset.return_value = [0.0, 2.1]
            
            kick_events = self.music_manager.get_percussive_events('kick')
            mock_get_onset.assert_called_once_with('kick', mock_beat_sync_result)


class TestContentSelectorPercussiveSync(unittest.TestCase):
    """Test ContentSelector percussive sync functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_content_database = Mock()
        self.content_selector = ContentSelector(self.mock_content_database)
        
    def test_selection_criteria_with_percussive_sync(self):
        """Test SelectionCriteria dataclass with new percussive sync fields"""
        criteria = SelectionCriteria(
            emotion='anxiety',
            intensity=0.8,
            themes=['stress', 'worry'],
            duration_target=30.0,
            music_beats=[0.0, 0.7, 1.4, 2.1],
            sync_event_type='kick',
            use_percussive_sync=True
        )
        
        self.assertEqual(criteria.sync_event_type, 'kick')
        self.assertTrue(criteria.use_percussive_sync)
        
    def test_generate_music_sync_points_with_event_type(self):
        """Test music sync point generation with percussive event type"""
        # Mock clips
        mock_clips = [Mock(), Mock(), Mock()]
        for i, clip in enumerate(mock_clips):
            clip.duration = 3.0
            clip.id = f"clip_{i}"
        
        # Mock music beats (kick events)
        kick_events = [0.0, 2.1, 4.2, 6.3]
        
        sync_points = self.content_selector._generate_music_sync_points(
            mock_clips, kick_events, 'kick'
        )
        
        # Verify sync points include event type
        for point in sync_points:
            self.assertEqual(point['sync_event_type'], 'kick')
            self.assertIn('start_time', point)
            self.assertIn('end_time', point)


class TestRealContentGeneratorPercussiveSync(unittest.TestCase):
    """Test RealContentGenerator percussive sync functionality"""
    
    def test_real_video_request_with_percussive_sync(self):
        """Test RealVideoRequest dataclass with new percussive sync fields"""
        request = RealVideoRequest(
            script_path="scripts/test.txt",
            script_name="test",
            variation_number=1,
            sync_event_type='hihat',
            use_percussive_sync=True
        )
        
        self.assertEqual(request.sync_event_type, 'hihat')
        self.assertTrue(request.use_percussive_sync)


class IntegrationTest(unittest.TestCase):
    """Integration tests for percussive sync workflow"""
    
    @patch('pathlib.Path.exists')
    async def test_end_to_end_percussive_sync_workflow(self, mock_exists):
        """Test the complete percussive sync workflow"""
        mock_exists.return_value = True
        
        # This would be a more complex integration test that tests the full workflow
        # For now, we'll test that the components work together
        
        # Create BeatSyncEngine and test classification
        engine = BeatSyncEngine()
        
        # Mock the librosa dependencies for testing
        with patch('librosa.load') as mock_load, \
             patch('librosa.feature.spectral_centroid') as mock_centroid, \
             patch('librosa.feature.spectral_rolloff') as mock_rolloff, \
             patch('librosa.feature.zero_crossing_rate') as mock_zcr, \
             patch('librosa.stft') as mock_stft, \
             patch('librosa.fft_frequencies') as mock_fft_freqs:
            
            # Setup mocks
            mock_load.return_value = (np.random.randn(44100 * 3), 44100)
            mock_centroid.return_value = np.array([[100.0, 2500.0]])
            mock_rolloff.return_value = np.array([[800.0, 3000.0]])
            mock_zcr.return_value = np.array([[0.05, 0.15]])
            mock_stft.return_value = np.random.randn(1025, 10) + 1j * np.random.randn(1025, 10)
            mock_fft_freqs.return_value = np.linspace(0, 22050, 1025)
            
            # Test that classification works
            kick_times, snare_times, hihat_times, other_times = await engine._classify_percussive_events(
                Path("test.mp3"), np.random.randn(44100 * 3), [0.5, 1.5]
            )
            
            # Verify we get some classification
            total_events = len(kick_times) + len(snare_times) + len(hihat_times) + len(other_times)
            self.assertEqual(total_events, 2)


async def run_tests():
    """Run all the tests"""
    logger.info("🧪 Running percussive sync tests...")
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTest(unittest.makeSuite(TestPercussiveSync))
    suite.addTest(unittest.makeSuite(TestMusicManagerPercussiveSync))
    suite.addTest(unittest.makeSuite(TestContentSelectorPercussiveSync))
    suite.addTest(unittest.makeSuite(TestRealContentGeneratorPercussiveSync))
    suite.addTest(unittest.makeSuite(IntegrationTest))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    if result.wasSuccessful():
        logger.info("✅ All percussive sync tests passed!")
        return True
    else:
        logger.error(f"❌ {len(result.failures)} test(s) failed, {len(result.errors)} error(s)")
        return False


def run_simple_functionality_test():
    """Run a simple test to verify the new functionality works"""
    logger.info("🎵 Testing percussive sync functionality...")
    
    try:
        # Test BeatSyncEngine get_onset_times method
        engine = BeatSyncEngine()
        
        # Create mock result with percussive events
        from beat_sync.beat_sync_engine import BeatSyncResult, BeatInfo
        mock_result = BeatSyncResult(
            beats=[BeatInfo(0.0, 1.0, True, 1, 120.0)],
            tempo=120.0,
            time_signature=(4, 4),
            phrases=[],
            onset_times=[0.0, 0.5, 1.0, 1.5],
            energy_curve=np.array([0.1, 0.2, 0.3, 0.4]),
            processing_time=0.1,
            kick_times=[0.0, 1.0],
            snare_times=[0.5],
            hihat_times=[1.5],
            other_times=[]
        )
        
        # Test different event types
        kick_times = engine.get_onset_times('kick', mock_result)
        snare_times = engine.get_onset_times('snare', mock_result)
        hihat_times = engine.get_onset_times('hihat', mock_result)
        
        logger.info(f"✅ Kick events: {kick_times}")
        logger.info(f"✅ Snare events: {snare_times}")
        logger.info(f"✅ Hi-hat events: {hihat_times}")
        
        # Test MusicManager
        music_manager = MusicManager("music/Beanie (Slowed).mp3")
        
        # Mock the track info
        from content.music_manager import MusicTrackInfo
        music_manager.track_info = MusicTrackInfo(
            filepath="music/Beanie (Slowed).mp3",
            filename="Beanie (Slowed).mp3",
            duration=180.0,
            sample_rate=44100,
            beats=[0.0, 0.7, 1.4, 2.1],
            analyzed=True
        )
        # Set percussive event data
        music_manager.track_info.kick_times = [0.0, 2.1]
        music_manager.track_info.snare_times = [0.7]
        music_manager.track_info.hihat_times = [1.4]
        music_manager.track_info.other_times = []
        music_manager.loaded = True
        
        # Test getting different event types
        kick_events = music_manager.get_beat_timing(0.0, 3.0, 'kick')
        logger.info(f"✅ Music manager kick events: {kick_events}")
        
        # Test mixing parameters with percussive sync
        mixing_params = music_manager.prepare_for_mixing(10.0, 'kick')
        logger.info(f"✅ Mixing params include sync_event_type: {mixing_params.get('sync_event_type')}")
        
        # Test ContentSelector
        from content.content_selector import SelectionCriteria
        criteria = SelectionCriteria(
            emotion='anxiety',
            intensity=0.8,
            themes=['test'],
            duration_target=30.0,
            music_beats=[0.0, 0.7, 1.4],
            sync_event_type='hihat',
            use_percussive_sync=True
        )
        logger.info(f"✅ Selection criteria with percussive sync: {criteria.sync_event_type}")
        
        # Test RealVideoRequest
        from core.real_content_generator import RealVideoRequest
        request = RealVideoRequest(
            script_path="test.txt",
            script_name="test",
            variation_number=1,
            sync_event_type='snare',
            use_percussive_sync=True
        )
        logger.info(f"✅ Video request with percussive sync: {request.sync_event_type}")
        
        logger.info("🎉 All functionality tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Functionality test failed: {e}")
        return False


if __name__ == "__main__":
    print("🥁 Percussive Sync Test Suite")
    print("=" * 50)
    
    # Run simple functionality test first
    if run_simple_functionality_test():
        print("\n" + "=" * 50)
        print("Running unit tests...")
        
        # Run async tests
        asyncio.run(run_tests())
    else:
        print("❌ Basic functionality test failed - skipping unit tests")