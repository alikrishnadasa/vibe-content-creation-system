import pytest
import numpy as np
import tempfile
import os
from unittest.mock import patch, MagicMock
from moviepy.editor import VideoFileClip, ColorClip

from src.processing.video_processor import VideoProcessor, VideoSegment


class TestVideoProcessor:
    
    @pytest.fixture
    def video_processor(self):
        return VideoProcessor(clip_duration=2.0, overlap=0.5)
    
    @pytest.fixture
    def sample_video_path(self):
        """Create a simple test video file."""
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
            # Create a simple 5-second red video
            clip = ColorClip(size=(64, 64), color=(255, 0, 0), duration=5)
            clip.write_videofile(tmp_file.name, fps=10, verbose=False, logger=None)
            clip.close()
            yield tmp_file.name
            
        # Cleanup
        if os.path.exists(tmp_file.name):
            os.unlink(tmp_file.name)
    
    def test_video_processor_initialization(self, video_processor):
        """Test VideoProcessor initialization."""
        assert video_processor.clip_duration == 2.0
        assert video_processor.overlap == 0.5
        assert video_processor.config is not None
    
    def test_video_processor_default_config(self):
        """Test VideoProcessor with default configuration."""
        processor = VideoProcessor()
        assert processor.clip_duration == processor.config.clip_duration.default
        assert processor.overlap == processor.config.clip_duration.overlap
    
    def test_segment_video(self, video_processor, sample_video_path):
        """Test video segmentation."""
        segments = video_processor.segment_video(sample_video_path)
        
        assert len(segments) > 0
        assert all(isinstance(segment, VideoSegment) for segment in segments)
        
        # Check segment properties
        for segment in segments:
            assert segment.start_time >= 0
            assert segment.end_time > segment.start_time
            assert segment.duration > 0
            assert segment.frames is not None
            assert len(segment.frames) > 0
    
    def test_segment_video_with_overlap(self, video_processor, sample_video_path):
        """Test video segmentation with overlap."""
        segments = video_processor.segment_video(sample_video_path)
        
        # Check that segments have proper overlap
        if len(segments) > 1:
            gap = segments[1].start_time - segments[0].end_time
            expected_gap = -(video_processor.overlap)  # Negative because of overlap
            assert abs(gap - expected_gap) < 0.1  # Allow small tolerance
    
    def test_process_video_file(self, video_processor, sample_video_path):
        """Test complete video file processing."""
        segments = video_processor.process_video_file(sample_video_path)
        
        assert len(segments) > 0
        assert all(isinstance(segment, VideoSegment) for segment in segments)
        
        # Verify segments cover the video duration
        total_duration = segments[-1].end_time
        assert total_duration > 0
    
    def test_process_nonexistent_file(self, video_processor):
        """Test processing non-existent file."""
        with pytest.raises(FileNotFoundError):
            video_processor.process_video_file("/nonexistent/path.mp4")
    
    def test_invalid_video_format(self, video_processor):
        """Test processing invalid video format."""
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp_file:
            tmp_file.write(b"not a video file")
            tmp_file.flush()
            
            with pytest.raises(ValueError):
                video_processor.process_video_file(tmp_file.name)
            
            os.unlink(tmp_file.name)
    
    def test_resize_frames(self, video_processor):
        """Test frame resizing."""
        # Create dummy frames
        frames = np.random.randint(0, 255, (3, 100, 100, 3), dtype=np.uint8)
        
        resized = video_processor.resize_frames(frames, target_size=(64, 64))
        
        assert resized.shape == (3, 64, 64, 3)
        assert resized.dtype == np.uint8
    
    def test_resize_empty_frames(self, video_processor):
        """Test resizing empty frames array."""
        frames = np.array([])
        resized = video_processor.resize_frames(frames)
        
        assert len(resized) == 0
    
    def test_normalize_frames(self, video_processor):
        """Test frame normalization."""
        frames = np.random.randint(0, 255, (3, 64, 64, 3), dtype=np.uint8)
        
        normalized = video_processor.normalize_frames(frames)
        
        assert normalized.shape == frames.shape
        assert normalized.dtype == np.float32
        assert normalized.min() >= -3.0  # Approximately after normalization
        assert normalized.max() <= 3.0
    
    def test_normalize_empty_frames(self, video_processor):
        """Test normalizing empty frames array."""
        frames = np.array([])
        normalized = video_processor.normalize_frames(frames)
        
        assert len(normalized) == 0
    
    def test_get_processing_stats(self, video_processor):
        """Test getting processing statistics."""
        stats = video_processor.get_processing_stats()
        
        assert isinstance(stats, dict)
        assert "clip_duration" in stats
        assert "overlap" in stats
        assert "temp_dir" in stats
        assert "cache_dir" in stats
    
    def test_extract_frames(self, video_processor):
        """Test frame extraction from video clip."""
        # Create a simple video clip
        clip = ColorClip(size=(32, 32), color=(0, 255, 0), duration=1.0)
        
        frames = video_processor._extract_frames(clip, max_frames=10)
        
        assert len(frames) > 0
        assert frames.shape[1:] == (32, 32, 3)  # Height, width, channels
        
        clip.close()
    
    def test_extract_frames_long_video(self, video_processor):
        """Test frame extraction with sampling for long videos."""
        clip = ColorClip(size=(32, 32), color=(0, 0, 255), duration=5.0)
        
        frames = video_processor._extract_frames(clip, max_frames=10)
        
        assert len(frames) <= 10
        assert frames.shape[1:] == (32, 32, 3)
        
        clip.close()
    
    def test_is_valid_video_format(self, video_processor):
        """Test video format validation."""
        assert video_processor._is_valid_video_format("test.mp4")
        assert video_processor._is_valid_video_format("test.avi")
        assert video_processor._is_valid_video_format("test.mov")
        assert video_processor._is_valid_video_format("test.mkv")
        assert video_processor._is_valid_video_format("test.webm")
        assert video_processor._is_valid_video_format("test.flv")
        
        assert not video_processor._is_valid_video_format("test.txt")
        assert not video_processor._is_valid_video_format("test.jpg")
        assert not video_processor._is_valid_video_format("test.wav")
    
    def test_video_segment_properties(self):
        """Test VideoSegment dataclass properties."""
        frames = np.random.randint(0, 255, (5, 64, 64, 3), dtype=np.uint8)
        
        segment = VideoSegment(
            start_time=0.0,
            end_time=2.0,
            frames=frames,
            frame_rate=30.0,
            duration=2.0
        )
        
        assert segment.start_time == 0.0
        assert segment.end_time == 2.0
        assert segment.duration == 2.0
        assert segment.frame_rate == 30.0
        assert np.array_equal(segment.frames, frames)
    
    @patch('src.processing.video_processor.VideoFileClip')
    def test_segment_video_error_handling(self, mock_video_clip, video_processor):
        """Test error handling in video segmentation."""
        mock_video_clip.side_effect = Exception("Video processing error")
        
        with pytest.raises(Exception):
            video_processor.segment_video("/fake/path.mp4")
    
    def test_different_clip_durations(self, sample_video_path):
        """Test video processing with different clip durations."""
        # Test with 1 second clips
        processor1 = VideoProcessor(clip_duration=1.0, overlap=0.0)
        segments1 = processor1.process_video_file(sample_video_path)
        
        # Test with 3 second clips  
        processor3 = VideoProcessor(clip_duration=3.0, overlap=0.0)
        segments3 = processor3.process_video_file(sample_video_path)
        
        # More segments with smaller duration
        assert len(segments1) >= len(segments3)
        
        # Check durations are approximately correct
        for segment in segments1:
            assert segment.duration <= 1.1  # Allow small tolerance
        
        for segment in segments3:
            assert segment.duration <= 3.1  # Allow small tolerance
    
    def test_edge_case_very_short_video(self, video_processor):
        """Test processing very short video."""
        # Create a very short video (0.1 seconds)
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
            clip = ColorClip(size=(32, 32), color=(255, 255, 0), duration=0.1)
            clip.write_videofile(tmp_file.name, fps=10, verbose=False, logger=None)
            clip.close()
            
            segments = video_processor.process_video_file(tmp_file.name)
            
            assert len(segments) >= 1
            assert segments[0].duration > 0
            
            os.unlink(tmp_file.name)