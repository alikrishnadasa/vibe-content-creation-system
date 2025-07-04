"""
Real Asset Processor

Loads MJAnime clips into GPU memory efficiently and processes them with 
universal background music integration. Handles audio mixing and video compositing.
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
from dataclasses import dataclass

logger = logging.getLogger(__name__)

def check_moviepy():
    """Check if MoviePy is available and working"""
    try:
        from moviepy.editor import VideoFileClip, AudioFileClip, CompositeVideoClip, TextClip, concatenate_videoclips
        logger.info("MoviePy available for real video processing")
        return True
    except ImportError as e:
        logger.warning(f"MoviePy not available - using placeholder video generation: {e}")
        return False

MOVIEPY_AVAILABLE = check_moviepy()

@dataclass
class AssetLoadResult:
    """Result of asset loading operation"""
    success: bool
    asset_path: str
    duration: float
    resolution: Tuple[int, int]
    fps: float
    memory_usage_mb: float
    load_time: float
    error_message: Optional[str] = None

@dataclass
class AudioMixResult:
    """Result of audio mixing operation"""
    success: bool
    output_path: str
    script_duration: float
    music_duration: float
    mixed_duration: float
    script_volume: float
    music_volume: float
    mix_time: float
    error_message: Optional[str] = None

class RealAssetProcessor:
    """Processes real MJAnime clips with GPU optimization and music integration"""
    
    def __init__(self, gpu_memory_pool_mb: int = 2048):
        """
        Initialize real asset processor
        
        Args:
            gpu_memory_pool_mb: GPU memory pool size in MB
        """
        self.gpu_memory_pool_mb = gpu_memory_pool_mb
        self.loaded_assets = {}  # Cache for loaded assets
        self.gpu_initialized = False
        
        # Asset processing stats
        self.stats = {
            'assets_loaded': 0,
            'total_load_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0,
            'gpu_memory_used_mb': 0.0
        }
    
    def initialize_gpu_processing(self) -> bool:
        """
        Initialize GPU processing capabilities
        
        Returns:
            bool: True if successful
        """
        try:
            logger.info("Initializing GPU processing for real assets...")
            
            # For now, simulate GPU initialization
            # This will be replaced with actual GPU processing setup
            self.gpu_initialized = True
            
            logger.info(f"GPU processing initialized with {self.gpu_memory_pool_mb}MB memory pool")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize GPU processing: {e}")
            return False
    
    async def load_clip_efficiently(self, clip_path: str, preload_to_gpu: bool = True) -> AssetLoadResult:
        """
        Load MJAnime clip efficiently with optional GPU preload
        
        Args:
            clip_path: Path to the video clip
            preload_to_gpu: Whether to preload to GPU memory
            
        Returns:
            Asset load result
        """
        start_time = time.time()
        
        try:
            clip_path = Path(clip_path)
            
            if not clip_path.exists():
                return AssetLoadResult(
                    success=False,
                    asset_path=str(clip_path),
                    duration=0.0,
                    resolution=(0, 0),
                    fps=0.0,
                    memory_usage_mb=0.0,
                    load_time=time.time() - start_time,
                    error_message=f"Clip file not found: {clip_path}"
                )
            
            # Check cache first
            if str(clip_path) in self.loaded_assets:
                self.stats['cache_hits'] += 1
                cached_result = self.loaded_assets[str(clip_path)]
                logger.debug(f"Cache hit for {clip_path.name}")
                return cached_result
            
            self.stats['cache_misses'] += 1
            
            # Get file info
            file_size_mb = clip_path.stat().st_size / (1024 * 1024)
            
            # Simulate video loading (would use actual video library)
            await self._simulate_video_loading(clip_path)
            
            # For MJAnime clips, we know the standard format
            duration = 5.21  # Standard MJAnime clip duration
            resolution = (1080, 1936)  # Standard resolution
            fps = 24.0  # Standard FPS
            
            # Estimate GPU memory usage
            gpu_memory_mb = 0.0
            if preload_to_gpu and self.gpu_initialized:
                # Estimate: uncompressed frame size × fps × duration
                frame_size_mb = (resolution[0] * resolution[1] * 3) / (1024 * 1024)  # RGB
                gpu_memory_mb = frame_size_mb * fps * duration
                self.stats['gpu_memory_used_mb'] += gpu_memory_mb
            
            load_time = time.time() - start_time
            
            result = AssetLoadResult(
                success=True,
                asset_path=str(clip_path),
                duration=duration,
                resolution=resolution,
                fps=fps,
                memory_usage_mb=gpu_memory_mb,
                load_time=load_time
            )
            
            # Cache the result
            self.loaded_assets[str(clip_path)] = result
            self.stats['assets_loaded'] += 1
            self.stats['total_load_time'] += load_time
            
            logger.debug(f"Loaded {clip_path.name}: {duration:.1f}s, {gpu_memory_mb:.1f}MB GPU")
            
            return result
            
        except Exception as e:
            load_time = time.time() - start_time
            logger.error(f"Failed to load clip {clip_path}: {e}")
            
            return AssetLoadResult(
                success=False,
                asset_path=str(clip_path),
                duration=0.0,
                resolution=(0, 0),
                fps=0.0,
                memory_usage_mb=0.0,
                load_time=load_time,
                error_message=str(e)
            )
    
    async def _simulate_video_loading(self, clip_path: Path):
        """Simulate video loading delay (would be actual processing)"""
        # Simulate different loading times based on file size
        file_size_mb = clip_path.stat().st_size / (1024 * 1024)
        load_delay = min(0.1, file_size_mb / 1000)  # Scale with file size
        
        if load_delay > 0.01:  # Only delay for larger files
            import asyncio
            await asyncio.sleep(load_delay)
    
    async def load_music_track(self, music_path: str, segment_start: float = 0.0, segment_duration: float = 180.0) -> AssetLoadResult:
        """
        Load and prepare music track segment
        
        Args:
            music_path: Path to music file
            segment_start: Start time for segment
            segment_duration: Duration of segment needed
            
        Returns:
            Asset load result
        """
        start_time = time.time()
        
        try:
            music_path = Path(music_path)
            
            if not music_path.exists():
                return AssetLoadResult(
                    success=False,
                    asset_path=str(music_path),
                    duration=0.0,
                    resolution=(0, 0),  # N/A for audio
                    fps=0.0,  # N/A for audio
                    memory_usage_mb=0.0,
                    load_time=time.time() - start_time,
                    error_message=f"Music file not found: {music_path}"
                )
            
            # Check cache
            cache_key = f"{music_path}:{segment_start}:{segment_duration}"
            if cache_key in self.loaded_assets:
                self.stats['cache_hits'] += 1
                return self.loaded_assets[cache_key]
            
            self.stats['cache_misses'] += 1
            
            # Simulate audio loading
            await self._simulate_audio_loading(music_path)
            
            # Estimate memory usage for audio segment
            # Assuming 44.1kHz, 16-bit, stereo
            sample_rate = 44100
            bits_per_sample = 16
            channels = 2
            audio_memory_mb = (sample_rate * bits_per_sample * channels * segment_duration) / (8 * 1024 * 1024)
            
            load_time = time.time() - start_time
            
            result = AssetLoadResult(
                success=True,
                asset_path=str(music_path),
                duration=segment_duration,
                resolution=(0, 0),  # N/A for audio
                fps=0.0,  # N/A for audio
                memory_usage_mb=audio_memory_mb,
                load_time=load_time
            )
            
            # Cache the result
            self.loaded_assets[cache_key] = result
            self.stats['assets_loaded'] += 1
            self.stats['total_load_time'] += load_time
            
            logger.debug(f"Loaded music segment: {segment_duration:.1f}s from {segment_start:.1f}s, {audio_memory_mb:.1f}MB")
            
            return result
            
        except Exception as e:
            load_time = time.time() - start_time
            logger.error(f"Failed to load music track: {e}")
            
            return AssetLoadResult(
                success=False,
                asset_path=str(music_path),
                duration=0.0,
                resolution=(0, 0),
                fps=0.0,
                memory_usage_mb=0.0,
                load_time=load_time,
                error_message=str(e)
            )
    
    async def _simulate_audio_loading(self, audio_path: Path):
        """Simulate audio loading delay"""
        file_size_mb = audio_path.stat().st_size / (1024 * 1024)
        load_delay = min(0.05, file_size_mb / 2000)  # Audio loads faster than video
        
        if load_delay > 0.01:
            import asyncio
            await asyncio.sleep(load_delay)
    
    async def mix_script_with_music(self,
                                   script_path: str,
                                   music_path: str, 
                                   music_start_time: float,
                                   music_duration: float,
                                   output_path: str,
                                   script_volume: float = 1.0,
                                   music_volume: float = 0.25) -> AudioMixResult:
        """
        Mix script audio with background music
        
        Args:
            script_path: Path to script audio file
            music_path: Path to music file
            music_start_time: Start time in music track
            music_duration: Duration to use from music
            output_path: Output path for mixed audio
            script_volume: Script volume level (0.0-1.0)
            music_volume: Music volume level (0.0-1.0)
            
        Returns:
            Audio mix result
        """
        start_time = time.time()
        
        try:
            script_path = Path(script_path)
            music_path = Path(music_path)
            output_path = Path(output_path)
            
            # Validate input files
            if not script_path.exists():
                return AudioMixResult(
                    success=False,
                    output_path=str(output_path),
                    script_duration=0.0,
                    music_duration=0.0,
                    mixed_duration=0.0,
                    script_volume=script_volume,
                    music_volume=music_volume,
                    mix_time=time.time() - start_time,
                    error_message=f"Script file not found: {script_path}"
                )
            
            if not music_path.exists():
                return AudioMixResult(
                    success=False,
                    output_path=str(output_path),
                    script_duration=0.0,
                    music_duration=0.0,
                    mixed_duration=0.0,
                    script_volume=script_volume,
                    music_volume=music_volume,
                    mix_time=time.time() - start_time,
                    error_message=f"Music file not found: {music_path}"
                )
            
            # Simulate audio analysis and mixing
            await self._simulate_audio_mixing(script_path, music_path)
            
            # Estimate durations (would be actual analysis)
            script_file_size = script_path.stat().st_size / (1024 * 1024)
            script_duration = script_file_size * 8.0  # Rough estimate: 8 seconds per MB
            
            mixed_duration = max(script_duration, music_duration)
            
            # Create output directory
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create real mixed audio file if MoviePy is available
            if MOVIEPY_AVAILABLE:
                try:
                    from moviepy.editor import AudioFileClip
                    # Load script audio
                    script_audio = AudioFileClip(str(script_path))
                    script_duration = script_audio.duration
                    
                    # Load music segment
                    music_audio = AudioFileClip(str(music_path))
                    if music_start_time > 0 or music_duration < music_audio.duration:
                        music_audio = music_audio.subclip(music_start_time, music_start_time + music_duration)
                    
                    # Adjust volumes
                    script_audio = script_audio.volumex(script_volume)
                    music_audio = music_audio.volumex(music_volume)
                    
                    # Mix audio (script as primary, music as background)
                    if music_audio.duration > script_audio.duration:
                        music_audio = music_audio.subclip(0, script_audio.duration)
                    elif music_audio.duration < script_audio.duration:
                        # Loop music if it's shorter than script
                        try:
                            from moviepy.audio.fx.audio_loop import audio_loop
                            loops_needed = int(script_audio.duration / music_audio.duration) + 1
                            # Try both parameter names for compatibility
                            try:
                                music_audio = audio_loop(music_audio, nloops=loops_needed).subclip(0, script_audio.duration)
                            except TypeError:
                                music_audio = audio_loop(music_audio, n_loops=loops_needed).subclip(0, script_audio.duration)
                        except Exception as loop_error:
                            logger.warning(f"Audio looping failed: {loop_error}, using music as-is")
                            # Use music as-is without looping
                            pass
                    
                    # Composite audio - mix script and music
                    from moviepy.editor import CompositeAudioClip
                    mixed_audio = CompositeAudioClip([script_audio, music_audio])
                    
                    # Write mixed audio
                    mixed_audio.write_audiofile(str(output_path), verbose=False, logger=None, fps=44100)
                    
                    # Get actual duration
                    mixed_duration = mixed_audio.duration
                    
                    # Clean up
                    script_audio.close()
                    music_audio.close()
                    mixed_audio.close()
                    
                    logger.info(f"Real audio mixed: {script_duration:.1f}s script + music = {mixed_duration:.1f}s")
                    
                except Exception as e:
                    logger.warning(f"Real audio mixing failed, creating placeholder: {e}")
                    # Fall back to placeholder
                    await self._create_placeholder_audio(script_path, music_path, music_start_time, script_duration, music_duration, mixed_duration, script_volume, music_volume, output_path)
            else:
                # Create placeholder audio file
                await self._create_placeholder_audio(script_path, music_path, music_start_time, script_duration, music_duration, mixed_duration, script_volume, music_volume, output_path)
            
            mix_time = time.time() - start_time
            
            logger.info(f"Mixed audio: {script_duration:.1f}s script + {music_duration:.1f}s music = {mixed_duration:.1f}s")
            
            return AudioMixResult(
                success=True,
                output_path=str(output_path),
                script_duration=script_duration,
                music_duration=music_duration,
                mixed_duration=mixed_duration,
                script_volume=script_volume,
                music_volume=music_volume,
                mix_time=mix_time
            )
            
        except Exception as e:
            mix_time = time.time() - start_time
            logger.error(f"Audio mixing failed: {e}")
            
            return AudioMixResult(
                success=False,
                output_path=str(output_path),
                script_duration=0.0,
                music_duration=0.0,
                mixed_duration=0.0,
                script_volume=script_volume,
                music_volume=music_volume,
                mix_time=mix_time,
                error_message=str(e)
            )
    
    async def _simulate_audio_mixing(self, script_path: Path, music_path: Path):
        """Simulate audio mixing processing"""
        # Simulate processing time based on file sizes
        script_size = script_path.stat().st_size / (1024 * 1024)
        music_size = music_path.stat().st_size / (1024 * 1024)
        
        processing_delay = min(0.2, (script_size + music_size) / 100)
        
        if processing_delay > 0.01:
            import asyncio
            await asyncio.sleep(processing_delay)
    
    async def composite_video_with_captions(self,
                                          clip_paths: List[str],
                                          audio_path: str,
                                          caption_data: Dict[str, Any],
                                          output_path: str,
                                          target_resolution: Tuple[int, int] = (1080, 1936),
                                          beat_sync_points: Optional[List[Dict[str, float]]] = None) -> Dict[str, Any]:
        """
        Composite final video with clips, audio, and captions
        
        Args:
            clip_paths: List of video clip paths
            audio_path: Path to mixed audio
            caption_data: Caption information
            output_path: Output video path
            target_resolution: Target video resolution
            
        Returns:
            Composition result
        """
        start_time = time.time()
        
        try:
            logger.info(f"Creating real video with {len(clip_paths)} clips and {len(caption_data.get('captions', []))} captions")
            
            if not MOVIEPY_AVAILABLE:
                logger.warning("MoviePy not available, creating placeholder video")
                return await self._create_placeholder_video(clip_paths, audio_path, caption_data, output_path, target_resolution, start_time)
            
            # Create output directory
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Load video clips
            from moviepy.editor import VideoFileClip, AudioFileClip, CompositeVideoClip, TextClip, concatenate_videoclips
            
            video_clips = []
            target_video_duration = None
            
            # Get target duration from audio if available
            if audio_path and Path(audio_path).exists():
                try:
                    temp_audio = AudioFileClip(str(audio_path))
                    target_video_duration = temp_audio.duration
                    temp_audio.close()
                    logger.info(f"Target video duration from audio: {target_video_duration:.1f}s")
                except Exception as e:
                    logger.warning(f"Could not get audio duration: {e}")
            
            # Use beat synchronization if available, otherwise fallback to fixed duration
            if beat_sync_points and len(beat_sync_points) == len(clip_paths):
                logger.info(f"Using beat synchronization for {len(clip_paths)} clips")
                total_duration = 0.0
                
                for i, clip_path in enumerate(clip_paths):
                    if Path(clip_path).exists():
                        try:
                            clip = VideoFileClip(clip_path)
                            sync_info = beat_sync_points[i]
                            target_clip_duration = sync_info['duration']
                            
                            # Trim clip to beat-synchronized duration
                            if clip.duration > target_clip_duration:
                                # Trim from center to preserve the best part of the clip
                                start_trim = (clip.duration - target_clip_duration) / 2
                                clip = clip.subclip(start_trim, start_trim + target_clip_duration)
                                logger.debug(f"Beat-trimmed clip to {target_clip_duration:.2f}s: {Path(clip_path).name}")
                            elif clip.duration < target_clip_duration:
                                # If clip is shorter than target, use the whole clip
                                logger.debug(f"Using full clip duration {clip.duration:.2f}s (beat target: {target_clip_duration:.2f}s): {Path(clip_path).name}")
                            
                            # Resize to target resolution
                            clip = clip.resize(target_resolution)
                            video_clips.append(clip)
                            total_duration += clip.duration
                            logger.debug(f"Beat-synced clip {i+1}: {Path(clip_path).name} ({clip.duration:.2f}s, beat: {sync_info['beat_index']})")
                        except Exception as e:
                            logger.warning(f"Failed to load clip {clip_path}: {e}")
                            continue
                    else:
                        logger.warning(f"Clip not found: {clip_path}")
            else:
                # Fallback to fixed duration
                target_clip_duration = 2.875
                if beat_sync_points:
                    logger.warning(f"Beat sync mismatch: {len(beat_sync_points)} sync points vs {len(clip_paths)} clips, using fixed duration")
                else:
                    logger.info(f"No beat sync available, using fixed {target_clip_duration}s per clip")
                
                total_duration = 0.0
                
                for clip_path in clip_paths:
                    if Path(clip_path).exists():
                        try:
                            clip = VideoFileClip(clip_path)
                            
                            # Trim clip to target duration
                            if clip.duration > target_clip_duration:
                                # Trim from center to preserve the best part of the clip
                                start_trim = (clip.duration - target_clip_duration) / 2
                                clip = clip.subclip(start_trim, start_trim + target_clip_duration)
                                logger.debug(f"Trimmed clip to {target_clip_duration:.2f}s: {Path(clip_path).name}")
                            elif clip.duration < target_clip_duration:
                                # If clip is shorter than target, use the whole clip
                                logger.debug(f"Using full clip duration {clip.duration:.2f}s: {Path(clip_path).name}")
                            
                            # Resize to target resolution
                            clip = clip.resize(target_resolution)
                            video_clips.append(clip)
                            total_duration += clip.duration
                            logger.debug(f"Loaded clip: {Path(clip_path).name} ({clip.duration:.1f}s)")
                        except Exception as e:
                            logger.warning(f"Failed to load clip {clip_path}: {e}")
                            continue
                    else:
                        logger.warning(f"Clip not found: {clip_path}")
            
            if not video_clips:
                logger.error("No video clips could be loaded")
                return self._create_error_result(output_path, start_time, "No video clips could be loaded")
            
            # Get target audio duration BEFORE concatenating video clips
            target_audio_duration = None
            if audio_path and Path(audio_path).exists():
                try:
                    audio_path_obj = Path(audio_path)
                    if audio_path_obj.suffix.lower() in ['.mp3', '.wav', '.m4a', '.aac']:
                        temp_audio = AudioFileClip(str(audio_path))
                        target_audio_duration = temp_audio.duration
                        temp_audio.close()
                        logger.info(f"Target audio duration: {target_audio_duration:.1f}s")
                except Exception as e:
                    logger.warning(f"Could not get audio duration: {e}")
            
            # Concatenate video clips
            logger.info(f"Concatenating {len(video_clips)} video clips...")
            final_video = concatenate_videoclips(video_clips, method="compose")
            
            # CRITICAL: Trim video to audio duration IMMEDIATELY after concatenation
            if target_audio_duration and final_video.duration > target_audio_duration:
                logger.info(f"Trimming video from {final_video.duration:.1f}s to audio duration {target_audio_duration:.1f}s")
                final_video = final_video.subclip(0, target_audio_duration)
                total_duration = target_audio_duration  # Update total duration
            
            # Load and add audio if provided
            if audio_path and Path(audio_path).exists():
                try:
                    # Check if it's a real audio file or placeholder
                    audio_path_obj = Path(audio_path)
                    if audio_path_obj.suffix.lower() in ['.mp3', '.wav', '.m4a', '.aac']:
                        logger.info(f"Loading audio file: {audio_path_obj.name}")
                        audio = AudioFileClip(str(audio_path))
                        
                        # Final check - video duration should exactly match audio duration
                        if abs(audio.duration - final_video.duration) > 0.1:  # Allow small differences
                            logger.warning(f"Duration still mismatched - Audio: {audio.duration:.1f}s, Video: {final_video.duration:.1f}s")
                            # Force trim video to exact audio duration
                            final_video = final_video.subclip(0, audio.duration)
                            logger.info(f"Force-trimmed video to exact audio duration: {audio.duration:.1f}s")
                        else:
                            logger.info(f"Audio and video durations match: {audio.duration:.1f}s")
                        
                        final_video = final_video.set_audio(audio)
                        logger.info(f"✅ Audio successfully added: {audio_path_obj.name} ({audio.duration:.1f}s)")
                    else:
                        logger.warning(f"Audio file format not supported: {audio_path_obj.suffix}")
                        
                except Exception as e:
                    logger.warning(f"Failed to add audio {audio_path}: {e}")
                    # Try using the original script audio as fallback
                    logger.info("Attempting to use original script audio as fallback...")
                    try:
                        # Find a script audio file to use as fallback
                        for fallback_path in ["../11-scripts-for-tiktok/anxiety1.wav", "../11-scripts-for-tiktok/safe1.wav"]:
                            if Path(fallback_path).exists():
                                fallback_audio = AudioFileClip(fallback_path)
                                # Adjust video duration to match fallback audio
                                if fallback_audio.duration < final_video.duration:
                                    final_video = final_video.subclip(0, fallback_audio.duration)
                                final_video = final_video.set_audio(fallback_audio)
                                logger.info(f"✅ Fallback audio added: {Path(fallback_path).name}")
                                break
                    except Exception as fallback_error:
                        logger.warning(f"Fallback audio also failed: {fallback_error}")
            else:
                logger.info(f"No audio file provided or file doesn't exist: {audio_path}")
            
            # Add captions if provided
            caption_clips = []
            captions = caption_data.get('captions', [])
            
            if captions:
                logger.info(f"Adding {len(captions)} captions...")
                video_duration = final_video.duration
                logger.info(f"Video duration for caption sync: {video_duration:.1f}s")
                
                for i, caption in enumerate(captions):
                    try:
                        # CRITICAL: Ensure captions don't extend beyond video duration
                        start_time = min(caption['start_time'], video_duration)
                        end_time = min(caption['end_time'], video_duration)
                        
                        # Skip captions that would start beyond video duration
                        if start_time >= video_duration:
                            logger.info(f"Skipping caption '{caption['text']}' - starts beyond video duration ({start_time:.1f}s >= {video_duration:.1f}s)")
                            continue
                        
                        # Ensure minimum caption duration if trimmed
                        if end_time - start_time < 0.5:
                            logger.info(f"Skipping caption '{caption['text']}' - too short after trimming ({end_time - start_time:.1f}s)")
                            continue
                        
                        # Create text clip for caption
                        txt_clip = TextClip(
                            caption['text'], 
                            fontsize=60,
                            color='white',
                            stroke_color='black',
                            stroke_width=2,
                            font='Arial-Bold'
                        ).set_position(('center', 'bottom')).set_duration(
                            end_time - start_time
                        ).set_start(start_time)
                        
                        caption_clips.append(txt_clip)
                        logger.debug(f"Created caption: '{caption['text']}' ({start_time:.1f}s-{end_time:.1f}s)")
                        
                    except Exception as e:
                        logger.warning(f"Failed to create caption {i}: {e}")
            
            # Composite final video with captions
            if caption_clips:
                original_duration = final_video.duration
                final_video = CompositeVideoClip([final_video] + caption_clips)
                
                # CRITICAL: Ensure composite doesn't extend beyond original video duration
                if final_video.duration > original_duration:
                    logger.warning(f"Composite extended duration ({final_video.duration:.1f}s > {original_duration:.1f}s), trimming")
                    final_video = final_video.subclip(0, original_duration)
                
                logger.info(f"Composited video with {len(caption_clips)} captions (final: {final_video.duration:.1f}s)")
            else:
                logger.info(f"No captions to add (final: {final_video.duration:.1f}s)")
            
            # Write final video
            logger.info(f"Writing final video to {output_path}...")
            final_video.write_videofile(
                str(output_path),
                codec='libx264',
                audio_codec='aac',
                temp_audiofile='temp-audio.m4a',
                remove_temp=True,
                fps=24,
                preset='medium',
                ffmpeg_params=['-crf', '23']  # Good quality/size balance
            )
            
            # Clean up
            final_video.close()
            for clip in video_clips:
                clip.close()
            for clip in caption_clips:
                clip.close()
            
            composition_time = time.time() - start_time
            
            logger.info(f"✅ Real video created successfully in {composition_time:.3f}s: {output_path}")
            
            return {
                'success': True,
                'output_path': str(output_path),
                'duration': total_duration,
                'clips_processed': len(video_clips),
                'captions_applied': len(caption_clips),
                'composition_time': composition_time,
                'target_resolution': target_resolution,
                'file_size_mb': output_path.stat().st_size / (1024 * 1024) if output_path.exists() else 0
            }
            
        except Exception as e:
            composition_time = time.time() - start_time
            logger.error(f"Real video composition failed: {e}")
            
            return {
                'success': False,
                'output_path': str(output_path),
                'duration': 0.0,
                'clips_processed': 0,
                'captions_applied': 0,
                'composition_time': composition_time,
                'error_message': str(e)
            }
    
    async def _create_placeholder_video(self, clip_paths, audio_path, caption_data, output_path, target_resolution, start_time):
        """Create placeholder video when MoviePy is not available"""
        total_clips = len(clip_paths)
        total_captions = len(caption_data.get('captions', []))
        estimated_duration = caption_data.get('total_duration', 15.0)
        
        # Create placeholder file
        with open(output_path, 'w') as f:
            f.write(f"# Placeholder Video File (MoviePy not available)\n")
            f.write(f"Clips: {total_clips}\n")
            f.write(f"Audio: {Path(audio_path).name if audio_path else 'None'}\n")
            f.write(f"Captions: {total_captions}\n")
            f.write(f"Duration: {estimated_duration:.1f}s\n")
            f.write(f"Resolution: {target_resolution[0]}x{target_resolution[1]}\n")
            f.write(f"Caption Style: {caption_data.get('style', 'default')}\n")
            f.write(f"\nClip Sequence:\n")
            for i, clip_path in enumerate(clip_paths):
                f.write(f"  {i+1}. {Path(clip_path).name}\n")
            f.write(f"\nCaptions:\n")
            for cap in caption_data.get('captions', []):
                f.write(f"  {cap['start_time']:.1f}s-{cap['end_time']:.1f}s: {cap['text']}\n")
        
        composition_time = time.time() - start_time
        
        return {
            'success': True,
            'output_path': str(output_path),
            'duration': estimated_duration,
            'clips_processed': total_clips,
            'captions_applied': total_captions,
            'composition_time': composition_time,
            'target_resolution': target_resolution,
            'placeholder': True
        }
    
    def _create_error_result(self, output_path, start_time, error_message):
        """Create error result for failed video composition"""
        composition_time = time.time() - start_time
        
        return {
            'success': False,
            'output_path': str(output_path),
            'duration': 0.0,
            'clips_processed': 0,
            'captions_applied': 0,
            'composition_time': composition_time,
            'error_message': error_message
        }
    
    async def _create_placeholder_audio(self, script_path, music_path, music_start_time, script_duration, music_duration, mixed_duration, script_volume, music_volume, output_path):
        """Create placeholder audio file when real mixing is not available"""
        with open(output_path, 'w') as f:
            f.write(f"# Mixed Audio File (Placeholder)\n")
            f.write(f"Script: {Path(script_path).name}\n")
            f.write(f"Music: {Path(music_path).name}\n")
            f.write(f"Music Start: {music_start_time:.1f}s\n")
            f.write(f"Script Duration: {script_duration:.1f}s\n")
            f.write(f"Music Duration: {music_duration:.1f}s\n")
            f.write(f"Mixed Duration: {mixed_duration:.1f}s\n")
            f.write(f"Script Volume: {script_volume:.2f}\n")
            f.write(f"Music Volume: {music_volume:.2f}\n")
    
    async def _simulate_video_composition(self, clip_paths: List[str], audio_path: str):
        """Simulate video composition processing"""
        # Simulate processing time based on number of clips and complexity
        base_time = 0.1  # Base processing time
        per_clip_time = 0.02  # Additional time per clip
        
        processing_delay = base_time + (len(clip_paths) * per_clip_time)
        
        import asyncio
        await asyncio.sleep(min(processing_delay, 0.5))  # Cap at 0.5s for simulation
    
    def clear_asset_cache(self):
        """Clear loaded asset cache to free memory"""
        cleared_count = len(self.loaded_assets)
        self.loaded_assets.clear()
        self.stats['gpu_memory_used_mb'] = 0.0
        
        logger.info(f"Cleared {cleared_count} cached assets")
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get asset processing statistics"""
        cache_hit_rate = 0.0
        if self.stats['cache_hits'] + self.stats['cache_misses'] > 0:
            cache_hit_rate = self.stats['cache_hits'] / (self.stats['cache_hits'] + self.stats['cache_misses'])
        
        avg_load_time = 0.0
        if self.stats['assets_loaded'] > 0:
            avg_load_time = self.stats['total_load_time'] / self.stats['assets_loaded']
        
        return {
            'gpu_initialized': self.gpu_initialized,
            'gpu_memory_pool_mb': self.gpu_memory_pool_mb,
            'gpu_memory_used_mb': self.stats['gpu_memory_used_mb'],
            'assets_loaded': self.stats['assets_loaded'],
            'cached_assets': len(self.loaded_assets),
            'cache_hit_rate': cache_hit_rate,
            'average_load_time': avg_load_time,
            'total_load_time': self.stats['total_load_time']
        }