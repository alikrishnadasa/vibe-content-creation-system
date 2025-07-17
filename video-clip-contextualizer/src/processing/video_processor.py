import cv2
import numpy as np
from typing import List, Tuple, Optional
from pathlib import Path
import logging
from dataclasses import dataclass
import torch
from moviepy import VideoFileClip
import tempfile
import os

from ..config import get_config


@dataclass
class VideoSegment:
    start_time: float
    end_time: float
    frames: np.ndarray
    frame_rate: float
    duration: float


class VideoProcessor:
    def __init__(self, clip_duration: float = None, overlap: float = None):
        self.config = get_config()
        self.clip_duration = clip_duration or self.config.clip_duration.default
        self.overlap = overlap or self.config.clip_duration.overlap
        self.logger = logging.getLogger(__name__)
        
        # Ensure temp directories exist
        os.makedirs(self.config.storage.temp_dir, exist_ok=True)
        os.makedirs(self.config.storage.cache_dir, exist_ok=True)
    
    def segment_video(self, video_path: str) -> List[VideoSegment]:
        """
        Segment video into clips based on configured duration and overlap.
        
        Args:
            video_path: Path to the input video file
            
        Returns:
            List of VideoSegment objects
        """
        try:
            # Load video
            video = VideoFileClip(video_path)
            total_duration = video.duration
            segments = []
            
            # Calculate segment positions
            current_time = 0.0
            segment_id = 0
            
            while current_time < total_duration:
                end_time = min(current_time + self.clip_duration, total_duration)
                
                # Extract segment
                segment_clip = video.subclip(current_time, end_time)
                
                # Convert to frames
                frames = self._extract_frames(segment_clip)
                
                segment = VideoSegment(
                    start_time=current_time,
                    end_time=end_time,
                    frames=frames,
                    frame_rate=segment_clip.fps,
                    duration=end_time - current_time
                )
                
                segments.append(segment)
                
                # Move to next segment with overlap
                current_time += self.clip_duration - self.overlap
                segment_id += 1
                
                # Prevent infinite loop
                if current_time >= total_duration:
                    break
            
            video.close()
            return segments
            
        except Exception as e:
            self.logger.error(f"Error segmenting video {video_path}: {str(e)}")
            raise
    
    def _extract_frames(self, video_clip, max_frames: int = 30) -> np.ndarray:
        """
        Extract frames from video clip for processing.
        
        Args:
            video_clip: MoviePy VideoClip object
            max_frames: Maximum number of frames to extract
            
        Returns:
            Array of frames [num_frames, height, width, channels]
        """
        duration = video_clip.duration
        fps = video_clip.fps
        
        # Calculate frame sampling
        total_frames = int(duration * fps)
        if total_frames <= max_frames:
            frame_indices = list(range(total_frames))
        else:
            # Sample frames evenly
            frame_indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
        
        frames = []
        for frame_idx in frame_indices:
            time_point = frame_idx / fps
            if time_point < duration:
                frame = video_clip.get_frame(time_point)
                frames.append(frame)
        
        return np.array(frames) if frames else np.array([])
    
    def process_video_file(self, video_path: str) -> List[VideoSegment]:
        """
        Process a video file and return segmented clips.
        
        Args:
            video_path: Path to video file
            
        Returns:
            List of processed video segments
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Validate video format
        if not self._is_valid_video_format(video_path):
            raise ValueError(f"Unsupported video format: {video_path}")
        
        self.logger.info(f"Processing video: {video_path}")
        self.logger.info(f"Clip duration: {self.clip_duration}s, Overlap: {self.overlap}s")
        
        segments = self.segment_video(video_path)
        
        self.logger.info(f"Created {len(segments)} video segments")
        return segments
    
    def _is_valid_video_format(self, video_path: str) -> bool:
        """Check if video format is supported."""
        supported_formats = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv'}
        return Path(video_path).suffix.lower() in supported_formats
    
    def resize_frames(self, frames: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
        """
        Resize frames to target dimensions for model input.
        
        Args:
            frames: Array of frames
            target_size: Target (height, width)
            
        Returns:
            Resized frames
        """
        if len(frames) == 0:
            return frames
        
        resized_frames = []
        for frame in frames:
            resized = cv2.resize(frame, (target_size[1], target_size[0]))
            resized_frames.append(resized)
        
        return np.array(resized_frames)
    
    def normalize_frames(self, frames: np.ndarray) -> np.ndarray:
        """
        Normalize frames for model input.
        
        Args:
            frames: Array of frames
            
        Returns:
            Normalized frames
        """
        if len(frames) == 0:
            return frames
        
        # Convert to float and normalize to [0, 1]
        normalized = frames.astype(np.float32) / 255.0
        
        # Standard ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        
        normalized = (normalized - mean) / std
        
        return normalized
    
    def get_processing_stats(self) -> dict:
        """Get processing statistics."""
        return {
            "clip_duration": self.clip_duration,
            "overlap": self.overlap,
            "temp_dir": self.config.storage.temp_dir,
            "cache_dir": self.config.storage.cache_dir
        }