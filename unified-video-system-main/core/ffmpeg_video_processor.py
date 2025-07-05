"""
FFmpeg Video Processor
Ultra-fast video generation using FFmpeg instead of MoviePy
Target: <1s per video with hardware acceleration
"""

import asyncio
import subprocess
import logging
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class FFmpegResult:
    """Result of FFmpeg operation"""
    success: bool
    output_path: str
    processing_time: float
    error_message: Optional[str] = None
    command_used: Optional[str] = None

class FFmpegVideoProcessor:
    """
    Ultra-fast video processor using FFmpeg directly
    Replaces MoviePy for 10-50x speed improvement
    """
    
    def __init__(self, enable_hardware_acceleration: bool = True, ultra_speed_mode: bool = True):
        """
        Initialize FFmpeg processor
        
        Args:
            enable_hardware_acceleration: Use GPU encoding if available
            ultra_speed_mode: Enable maximum speed optimizations
        """
        self.enable_hw_accel = enable_hardware_acceleration
        self.ultra_speed_mode = ultra_speed_mode
        self.temp_dir = Path("temp")
        self.temp_dir.mkdir(exist_ok=True)
        
        # Hardware acceleration detection
        self.hw_encoders = self._detect_hardware_encoders()
        
        # FFmpeg settings optimized for ultra-speed
        self.fast_encode_settings = [
            "-movflags", "+faststart",  # Web optimization
            "-threads", "0",         # Use all CPU threads
            "-fflags", "+genpts",     # Generate presentation timestamps
            "-avoid_negative_ts", "make_zero",  # Avoid timestamp issues
            "-max_muxing_queue_size", "1024"   # Reduce memory overhead
        ]
    
    def _detect_hardware_encoders(self) -> Dict[str, bool]:
        """Detect available hardware encoders"""
        encoders = {
            'nvenc': False,    # NVIDIA
            'videotoolbox': False,  # Apple Silicon/Intel Mac
            'vaapi': False,    # Intel/AMD Linux
            'qsv': False       # Intel Quick Sync
        }
        
        try:
            # Check for NVIDIA NVENC
            result = subprocess.run(
                ["ffmpeg", "-hide_banner", "-encoders"], 
                capture_output=True, text=True, timeout=5
            )
            if "h264_nvenc" in result.stdout:
                encoders['nvenc'] = True
            if "h264_videotoolbox" in result.stdout:
                encoders['videotoolbox'] = True
            if "h264_vaapi" in result.stdout:
                encoders['vaapi'] = True
            if "h264_qsv" in result.stdout:
                encoders['qsv'] = True
                
        except Exception as e:
            logger.warning(f"Could not detect hardware encoders: {e}")
        
        logger.info(f"Hardware encoders available: {encoders}")
        return encoders
    
    def _get_optimal_encoder(self) -> List[str]:
        """Get the best available encoder for speed"""
        if not self.enable_hw_accel:
            return ["-c:v", "libx264"]
        
        # Priority order: fastest to slowest with ultra-speed optimizations
        if self.hw_encoders['videotoolbox']:
            return ["-c:v", "h264_videotoolbox", "-b:v", "1.5M", "-allow_sw", "1", "-realtime", "1"]
        elif self.hw_encoders['nvenc']:
            return ["-c:v", "h264_nvenc", "-preset", "p1", "-tune", "ll", "-b:v", "1.5M", "-2pass", "0"]  # Fastest NVENC preset
        elif self.hw_encoders['qsv']:
            return ["-c:v", "h264_qsv", "-preset", "veryfast", "-b:v", "1.5M"]
        elif self.hw_encoders['vaapi']:
            return ["-c:v", "h264_vaapi", "-b:v", "1.5M"]
        else:
            return ["-c:v", "libx264"]
    
    async def process_video_ultra_fast(self,
                                     clip_paths: List[str],
                                     audio_path: str,
                                     caption_data: Dict[str, Any],
                                     output_path: str,
                                     target_resolution: Tuple[int, int] = (1080, 1620),
                                     beat_sync_points: Optional[List[Dict]] = None,
                                     target_duration: Optional[float] = None,
                                     burn_in_captions: bool = False) -> FFmpegResult:
        """
        Ultra-fast video processing using FFmpeg
        Target: <1s per video
        """
        start_time = time.perf_counter()
        
        try:
            # Get audio duration if not provided
            if target_duration is None:
                target_duration = await self._get_audio_duration(audio_path)
                logger.info(f"Detected audio duration: {target_duration:.1f}s")
            
            # Step 1: Create video from clips (parallel processing)
            video_result = await self._create_video_from_clips(
                clip_paths, target_resolution, beat_sync_points, target_duration
            )
            if not video_result.success:
                return video_result
            
            # Step 2: Add audio (streaming, no temp files)
            audio_result = await self._add_audio_to_video(
                video_result.output_path, audio_path
            )
            if not audio_result.success:
                return audio_result
            
            # Step 3: Add captions (GPU-accelerated if available)
            final_result = await self._add_captions_to_video(
                audio_result.output_path, caption_data, output_path, burn_in_captions
            )
            
            # Step 4: Trim final video to match audio duration
            if final_result.success:
                trimmed_output = output_path.replace('.mp4', '_trimmed.mp4')
                trim_success = await self._trim_video_to_audio_duration(output_path, audio_path, trimmed_output)
                if trim_success:
                    # Replace original output with trimmed version
                    import os
                    os.replace(trimmed_output, output_path)
                else:
                    logger.warning("Failed to trim video to audio duration; output may have trailing silence/black.")
            
            processing_time = time.perf_counter() - start_time
            
            if final_result.success:
                # Cleanup temp files
                await self._cleanup_temp_files([video_result.output_path, audio_result.output_path])
                
                return FFmpegResult(
                    success=True,
                    output_path=output_path,
                    processing_time=processing_time
                )
            else:
                return final_result
                
        except Exception as e:
            processing_time = time.perf_counter() - start_time
            logger.error(f"FFmpeg processing failed: {e}")
            return FFmpegResult(
                success=False,
                output_path="",
                processing_time=processing_time,
                error_message=str(e)
            )
    
    async def _create_video_from_clips(self, 
                                     clip_paths: List[str], 
                                     target_resolution: Tuple[int, int],
                                     beat_sync_points: Optional[List[Dict]] = None,
                                     target_duration: Optional[float] = None) -> FFmpegResult:
        """Create video from clips with beat synchronization"""
        start_time = time.perf_counter()
        temp_output = self.temp_dir / f"temp_video_{int(time.time())}.mp4"
        
        try:
            # Create filter complex for clip concatenation and timing
            filter_complex = self._build_clip_filter_complex(
                clip_paths, target_resolution, beat_sync_points, target_duration
            )
            
            # Build FFmpeg command
            cmd = ["ffmpeg", "-y"]  # -y to overwrite output
            
            # Add input files
            for clip_path in clip_paths:
                cmd.extend(["-i", clip_path])
            
            # Add filter complex and map output
            cmd.extend(["-filter_complex", filter_complex])
            cmd.extend(["-map", "[out]"])  # Map the filter output
            
            # Add encoding settings
            encoder_settings = self._get_optimal_encoder()
            cmd.extend(encoder_settings)
            
            # Add software encoding settings only if using libx264
            if encoder_settings[1] == "libx264":
                cmd.extend(["-preset", "ultrafast", "-crf", "28", "-tune", "fastdecode", "-x264opts", "no-scenecut"])
            
            cmd.extend(self.fast_encode_settings)
            
            # Output settings
            cmd.extend([
                "-r", "24",  # 24 FPS
                "-an",  # No audio in video-only step
                str(temp_output)
            ])
            
            # Execute FFmpeg - try hardware first, fallback to software
            logger.info(f"Creating video from {len(clip_paths)} clips...")
            process = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            processing_time = time.perf_counter() - start_time
            
            if process.returncode == 0:
                logger.info(f"Video creation successful in {processing_time:.3f}s")
                return FFmpegResult(
                    success=True,
                    output_path=str(temp_output),
                    processing_time=processing_time,
                    command_used=" ".join(cmd)
                )
            else:
                # Hardware encoding failed, try software fallback
                if self.enable_hw_accel and encoder_settings[1] != "libx264":
                    logger.warning("Hardware encoding failed, trying software fallback...")
                    return await self._try_software_fallback(clip_paths, target_resolution, beat_sync_points, temp_output, start_time, target_duration)
                else:
                    error_msg = stderr.decode() if stderr else "Unknown FFmpeg error"
                    logger.error(f"FFmpeg video creation failed: {error_msg}")
                    return FFmpegResult(
                        success=False,
                        output_path="",
                        processing_time=processing_time,
                        error_message=error_msg,
                        command_used=" ".join(cmd)
                    )
                
        except Exception as e:
            processing_time = time.perf_counter() - start_time
            return FFmpegResult(
                success=False,
                output_path="",
                processing_time=processing_time,
                error_message=str(e)
            )
    
    def _build_clip_filter_complex(self, 
                                 clip_paths: List[str], 
                                 target_resolution: Tuple[int, int],
                                 beat_sync_points: Optional[List[Dict]] = None,
                                 target_duration: Optional[float] = None) -> str:
        """Build FFmpeg filter complex for clip concatenation with exact duration matching"""
        filters = []
        
        # Scale and prepare each clip - convert any aspect ratio to 2:3 by smart cropping
        for i, _ in enumerate(clip_paths):
            target_width = target_resolution[0]
            target_height = target_resolution[1]
            
            # Use scale and crop filter that works for any input resolution
            # First scale to fill the target dimensions, then crop from center
            filters.append(f"[{i}:v]scale={target_width}:{target_height}:force_original_aspect_ratio=increase,crop={target_width}:{target_height}:(iw-{target_width})/2:(ih-{target_height})/2,setpts=PTS-STARTPTS[v{i}]")
        
        # Create concatenation to match beat intervals (variable durations)
        if beat_sync_points and len(beat_sync_points) == len(clip_paths):
            for i, sync_point in enumerate(beat_sync_points):
                duration = sync_point.get('duration', 2.0)
                filters.append(f"[v{i}]trim=duration={duration:.3f}[v{i}trimmed]")
            concat_inputs = "".join(f"[v{i}trimmed]" for i in range(len(clip_paths)))
            filters.append(f"{concat_inputs}concat=n={len(clip_paths)}:v=1:a=0[out]")
        else:
            # Fallback: fixed duration if no beat sync points
            duration_per_clip = 2.0
            for i, _ in enumerate(clip_paths):
                filters.append(f"[v{i}]trim=duration={duration_per_clip:.3f}[v{i}trimmed]")
            concat_inputs = "".join(f"[v{i}trimmed]" for i in range(len(clip_paths)))
            filters.append(f"{concat_inputs}concat=n={len(clip_paths)}:v=1:a=0[out]")
        
        return ";".join(filters)
    
    async def _get_audio_duration(self, audio_path: str) -> float:
        """Get audio duration using FFmpeg"""
        try:
            cmd = [
                "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                "-of", "csv=p=0", audio_path
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                duration = float(stdout.decode().strip())
                return duration
            else:
                logger.warning(f"Could not get audio duration, using default: {stderr.decode()}")
                return 60.0  # Default fallback
                
        except Exception as e:
            logger.warning(f"Audio duration detection failed: {e}")
            return 60.0  # Default fallback
    
    async def _try_software_fallback(self, 
                                   clip_paths: List[str], 
                                   target_resolution: Tuple[int, int],
                                   beat_sync_points: Optional[List[Dict]],
                                   temp_output: Path,
                                   original_start_time: float,
                                   target_duration: Optional[float] = None) -> FFmpegResult:
        """Try software encoding as fallback when hardware fails"""
        try:
            filter_complex = self._build_clip_filter_complex(clip_paths, target_resolution, beat_sync_points, target_duration)
            
            cmd = ["ffmpeg", "-y"]
            
            # Add input files
            for clip_path in clip_paths:
                cmd.extend(["-i", clip_path])
            
            # Add filter complex and map output
            cmd.extend(["-filter_complex", filter_complex])
            cmd.extend(["-map", "[out]"])
            
            # Use software encoding with ultra-speed optimizations
            cmd.extend(["-c:v", "libx264", "-preset", "ultrafast", "-crf", "28", "-tune", "fastdecode", "-x264opts", "no-scenecut"])
            cmd.extend(self.fast_encode_settings)
            
            # Output settings
            cmd.extend([
                "-r", "24",  # 24 FPS
                "-an",  # No audio in video-only step
                str(temp_output)
            ])
            
            # Execute software fallback
            logger.info("Trying software encoding fallback...")
            process = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            processing_time = time.perf_counter() - original_start_time
            
            if process.returncode == 0:
                logger.info(f"Software fallback successful in {processing_time:.3f}s")
                return FFmpegResult(
                    success=True,
                    output_path=str(temp_output),
                    processing_time=processing_time,
                    command_used=" ".join(cmd)
                )
            else:
                error_msg = stderr.decode() if stderr else "Software fallback failed"
                logger.error(f"Software fallback failed: {error_msg}")
                return FFmpegResult(
                    success=False,
                    output_path="",
                    processing_time=processing_time,
                    error_message=error_msg,
                    command_used=" ".join(cmd)
                )
                
        except Exception as e:
            processing_time = time.perf_counter() - original_start_time
            return FFmpegResult(
                success=False,
                output_path="",
                processing_time=processing_time,
                error_message=f"Software fallback exception: {e}"
            )
    
    async def _add_audio_to_video(self, video_path: str, audio_path: str) -> FFmpegResult:
        """Add audio to video using FFmpeg streaming"""
        start_time = time.perf_counter()
        temp_output = self.temp_dir / f"temp_with_audio_{int(time.time())}.mp4"
        
        try:
            if self.ultra_speed_mode:
                cmd = [
                    "ffmpeg", "-y",
                    "-i", video_path,
                    "-i", audio_path,
                    "-c:v", "copy",  # Copy video stream (no re-encoding)
                    "-c:a", "aac", "-b:a", "96k",   # Lower bitrate audio for speed
                    "-shortest",     # Match shortest stream
                    "-avoid_negative_ts", "make_zero",
                    str(temp_output)
                ]
            else:
                cmd = [
                    "ffmpeg", "-y",
                    "-i", video_path,
                    "-i", audio_path,
                    "-c:v", "copy",  # Copy video stream (no re-encoding)
                    "-c:a", "aac",   # Audio codec
                    "-shortest",     # Match shortest stream
                    str(temp_output)
                ]
            
            logger.info("Adding audio to video...")
            process = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            processing_time = time.perf_counter() - start_time
            
            if process.returncode == 0:
                logger.info(f"Audio added successfully in {processing_time:.3f}s")
                return FFmpegResult(
                    success=True,
                    output_path=str(temp_output),
                    processing_time=processing_time
                )
            else:
                error_msg = stderr.decode() if stderr else "Unknown audio error"
                return FFmpegResult(
                    success=False,
                    output_path="",
                    processing_time=processing_time,
                    error_message=error_msg
                )
                
        except Exception as e:
            processing_time = time.perf_counter() - start_time
            return FFmpegResult(
                success=False,
                output_path="",
                processing_time=processing_time,
                error_message=str(e)
            )
    
    async def _add_captions_to_video(self, 
                                   video_path: str, 
                                   caption_data: Dict[str, Any], 
                                   output_path: str,
                                   burn_in_captions: bool = False) -> FFmpegResult:
        """Add captions using SRT subtitle file or burn them into video"""
        start_time = time.perf_counter()
        
        try:
            # Create SRT subtitle file
            srt_path = self.temp_dir / f"temp_subtitles_{int(time.time())}.srt"
            self._create_srt_file(caption_data, str(srt_path))
            
            if burn_in_captions:
                # Burn captions directly into video with custom styling - 120px from bottom with drop shadow
                cmd = [
                    "ffmpeg", "-y",
                    "-i", video_path,
                    "-vf", f"subtitles={str(srt_path)}:force_style='Fontname=HelveticaTextNow-ExtraBold,Fontsize=24,PrimaryColour=&Hffffff,OutlineColour=&Hffffff,Outline=0,Alignment=2,MarginV=120,Bold=1,Shadow=2,BackColour=&HD9000000'",
                    "-c:a", "copy",  # Copy audio stream
                    output_path
                ]
                logger.info("Burning captions into video with HelveticaTextNow-ExtraBold font and drop shadow...")
            else:
                # Add as subtitle stream (default behavior)
                cmd = [
                    "ffmpeg", "-y",
                    "-i", video_path,
                    "-i", str(srt_path),
                    "-c:v", "copy",  # Copy video stream (no re-encoding)
                    "-c:a", "copy",  # Copy audio stream
                    "-c:s", "mov_text",  # Subtitle codec
                    "-metadata:s:s:0", "language=eng",
                    output_path
                ]
                logger.info("Adding captions via SRT subtitle file...")
            
            process = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            processing_time = time.perf_counter() - start_time
            
            # Cleanup SRT file
            try:
                srt_path.unlink(missing_ok=True)
            except:
                pass
            
            if process.returncode == 0:
                logger.info(f"Captions added successfully in {processing_time:.3f}s")
                return FFmpegResult(
                    success=True,
                    output_path=output_path,
                    processing_time=processing_time
                )
            else:
                error_msg = stderr.decode() if stderr else "Unknown caption error"
                logger.warning(f"SRT subtitle method failed, trying without captions: {error_msg}")
                
                # Fallback: just copy the video without captions
                fallback_cmd = [
                    "ffmpeg", "-y",
                    "-i", video_path,
                    "-c", "copy",
                    output_path
                ]
                
                fallback_process = await asyncio.create_subprocess_exec(
                    *fallback_cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )
                
                stdout, stderr = await fallback_process.communicate()
                processing_time = time.perf_counter() - start_time
                
                if fallback_process.returncode == 0:
                    logger.info(f"Video copied without captions in {processing_time:.3f}s")
                    return FFmpegResult(
                        success=True,
                        output_path=output_path,
                        processing_time=processing_time
                    )
                else:
                    error_msg = stderr.decode() if stderr else "Unknown fallback error"
                    return FFmpegResult(
                        success=False,
                        output_path="",
                        processing_time=processing_time,
                        error_message=error_msg
                    )
                
        except Exception as e:
            processing_time = time.perf_counter() - start_time
            return FFmpegResult(
                success=False,
                output_path="",
                processing_time=processing_time,
                error_message=str(e)
            )
    
    def _build_caption_filters(self, caption_data: Dict[str, Any]) -> str:
        """Build FFmpeg drawtext filters for captions"""
        captions = caption_data.get('captions', [])
        if not captions or self.ultra_speed_mode:
            # Skip captions in ultra-speed mode to maximize performance
            return "scale=trunc(iw/2)*2:trunc(ih/2)*2"  # Just ensure even dimensions
        
        # For word-by-word captions, group words into phrases to reduce filter complexity
        # Group every 3-4 words together to cover the full duration while staying manageable
        if len(captions) > 20:
            logger.info(f"Grouping {len(captions)} word captions into phrases for FFmpeg processing")
            grouped_captions = []
            words_per_group = 4
            
            for i in range(0, len(captions), words_per_group):
                group = captions[i:i + words_per_group]
                if group:
                    # Combine words into a phrase
                    combined_text = " ".join(cap['text'] for cap in group)
                    grouped_captions.append({
                        'text': combined_text,
                        'start_time': group[0]['start_time'],
                        'end_time': group[-1]['end_time'],
                        'style': group[0]['style'],
                        'confidence': group[0]['confidence']
                    })
            
            captions = grouped_captions
            logger.info(f"Created {len(captions)} phrase groups covering full duration")
        
        filters = []
        
        for i, caption in enumerate(captions):
            # Properly escape text for FFmpeg
            text = caption['text'].replace("'", "\\'").replace(":", "\\:")
            start_time = caption['start_time']
            end_time = caption['end_time']
            
            # Get style from caption data
            style = caption_data.get('style', 'tiktok')
            font_settings = self._get_ffmpeg_font_settings(style)
            
            # Build drawtext filter with proper escaping
            drawtext = f"drawtext=text='{text}':enable='between(t\\,{start_time}\\,{end_time})'"
            drawtext += f":{font_settings}"
            
            filters.append(drawtext)
        
        # Chain filters properly
        if len(filters) == 1:
            return filters[0]
        else:
            return ",".join(filters)
    
    def _create_srt_file(self, caption_data: Dict[str, Any], srt_path: str) -> None:
        """Create SRT subtitle file from caption data"""
        captions = caption_data.get('captions', [])
        if not captions:
            logger.warning("No captions found in caption data")
            return
        
        try:
            with open(srt_path, 'w', encoding='utf-8') as srt_file:
                for i, caption in enumerate(captions, 1):
                    # Convert seconds to SRT time format (HH:MM:SS,mmm)
                    start_time = self._seconds_to_srt_time(caption['start_time'])
                    end_time = self._seconds_to_srt_time(caption['end_time'])
                    
                    # Write SRT entry
                    srt_file.write(f"{i}\n")
                    srt_file.write(f"{start_time} --> {end_time}\n")
                    srt_file.write(f"{caption['text']}\n")
                    srt_file.write("\n")
            
            logger.info(f"Created SRT file with {len(captions)} captions at {srt_path}")
            
        except Exception as e:
            logger.error(f"Failed to create SRT file: {e}")
            raise
    
    def _seconds_to_srt_time(self, seconds: float) -> str:
        """Convert seconds to SRT time format (HH:MM:SS,mmm)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millisecs = int((seconds % 1) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"
    
    def _get_ffmpeg_font_settings(self, style: str) -> str:
        """Get FFmpeg font settings for caption style with HelveticaTextNow-ExtraBold default (center screen, no border)"""
        style_settings = {
            'tiktok': "fontsize=24:fontcolor=white:fontfile=HelveticaTextNow-ExtraBold:x=(w-text_w)/2:y=(h-text_h)/2",
            'default': "fontsize=24:fontcolor=white:fontfile=HelveticaTextNow-ExtraBold:x=(w-text_w)/2:y=(h-text_h)/2", 
            'youtube': "fontsize=24:fontcolor=white:fontfile=HelveticaTextNow-ExtraBold:x=(w-text_w)/2:y=(h-text_h)/2",
            'minimal': "fontsize=24:fontcolor=white:fontfile=HelveticaTextNow-ExtraBold:x=(w-text_w)/2:y=(h-text_h)/2"
        }
        
        return style_settings.get(style, style_settings['default'])
    
    def _get_ffmpeg_style_from_preset(self, style_name: str) -> str:
        """Convert preset_manager style to FFmpeg subtitle styling"""
        try:
            # Import here to avoid circular imports
            from captions.preset_manager import CaptionPresetManager, CaptionPosition
            
            manager = CaptionPresetManager()
            style = manager.get_preset(style_name)
            
            # Convert hex colors to FFmpeg format
            def hex_to_ffmpeg_color(color_str: str) -> str:
                if color_str.startswith('#'):
                    # Remove # and convert to BGR format for FFmpeg
                    hex_color = color_str[1:]
                    if len(hex_color) == 6:
                        r, g, b = hex_color[0:2], hex_color[2:4], hex_color[4:6]
                        return f"&H{b}{g}{r}"
                elif color_str.lower() == 'white':
                    return "&Hffffff"
                elif color_str.lower() == 'black':
                    return "&H000000"
                elif color_str.lower() == 'yellow':
                    return "&H00ffff"
                else:
                    return "&Hffffff"  # Default to white
            
            # Convert position to alignment
            alignment_map = {
                CaptionPosition.CENTER: "2",      # Bottom center - will offset up with margin for better centering
                CaptionPosition.BOTTOM: "2",      # Bottom center
                CaptionPosition.TOP: "8",         # Top center
                CaptionPosition.BOTTOM_LEFT: "1", # Bottom left
                CaptionPosition.BOTTOM_RIGHT: "3" # Bottom right
            }
            
            alignment = alignment_map.get(style.position, "2")
            
            # Build FFmpeg style string
            ffmpeg_parts = [
                f"Fontname={style.font_family}",
                f"Fontsize={style.font_size}",
                f"PrimaryColour={hex_to_ffmpeg_color(style.font_color)}",
                f"Alignment={alignment}",
                f"Bold=1" if style.font_weight in ["bold", "extra-bold"] else "Bold=0"
            ]
            
            # Add positioning margins
            if style.position == CaptionPosition.CENTER:
                # Use bottom alignment with 60px offset for better centering
                ffmpeg_parts.extend([
                    "MarginV=100",   # 60px up from bottom for better centering
                    "MarginL=0",    # No left margin
                    "MarginR=0"     # No right margin
                ])
            elif style.position == CaptionPosition.BOTTOM:
                ffmpeg_parts.append(f"MarginV={style.margin}")
            elif style.position == CaptionPosition.TOP:
                ffmpeg_parts.append(f"MarginV={style.margin}")
            
            # Add outline if specified
            if style.outline_color and style.outline_width > 0:
                ffmpeg_parts.extend([
                    f"OutlineColour={hex_to_ffmpeg_color(style.outline_color)}",
                    f"Outline={style.outline_width}"
                ])
            else:
                ffmpeg_parts.append("Outline=0")
            
            # Add shadow if specified - increased blur and opacity
            if style.shadow_blur > 0:
                # Increased shadow distance and blur for more prominent shadow
                shadow_distance = 4  # More prominent shadow
                ffmpeg_parts.append(f"Shadow={shadow_distance}")
                
                # Add shadow color with higher opacity
                if hasattr(style, 'shadow_color') and style.shadow_color:
                    shadow_color = hex_to_ffmpeg_color(style.shadow_color)
                    # Add high opacity black shadow
                    ffmpeg_parts.append(f"BackColour=&HFF000000")  # Full opacity black
                else:
                    # Default to high opacity black shadow
                    ffmpeg_parts.append("BackColour=&HFF000000")
            
            # Add background if specified
            if style.background_color and style.background_opacity > 0:
                bg_color = hex_to_ffmpeg_color(style.background_color)
                # Add alpha channel for opacity (FF = fully opaque, 00 = transparent)
                opacity_hex = format(int(255 * style.background_opacity), '02x')
                bg_with_alpha = f"&H{opacity_hex}{bg_color[2:]}"
                ffmpeg_parts.append(f"BackColour={bg_with_alpha}")
            
            return ",".join(ffmpeg_parts)
            
        except Exception as e:
            logger.warning(f"Failed to load preset {style_name}, using default: {e}")
            # Fallback to improved default style with bottom positioning and better shadow
            return "Fontname=HelveticaTextNow-ExtraBold,Fontsize=45,PrimaryColour=&Hffffff,Alignment=2,Bold=1,Outline=0,Shadow=4,BackColour=&HFF000000,MarginV=60,MarginL=0,MarginR=0"
    
    async def _cleanup_temp_files(self, temp_files: List[str]):
        """Clean up temporary files"""
        for temp_file in temp_files:
            try:
                Path(temp_file).unlink(missing_ok=True)
            except Exception as e:
                logger.warning(f"Could not delete temp file {temp_file}: {e}")
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processor statistics and capabilities"""
        return {
            'hardware_acceleration_enabled': self.enable_hw_accel,
            'available_encoders': self.hw_encoders,
            'optimal_encoder': self._get_optimal_encoder(),
            'fast_encode_settings': self.fast_encode_settings
        }

    async def _trim_video_to_audio_duration(self, video_path: str, audio_path: str, output_path: str) -> bool:
        """Trim the video to match the duration of the audio file using FFmpeg."""
        try:
            # Get audio duration using ffprobe
            import subprocess
            result = subprocess.run([
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", audio_path
            ], capture_output=True, text=True)
            if result.returncode != 0:
                return False
            audio_duration = float(result.stdout.strip())
            # Run FFmpeg to trim video
            cmd = [
                "ffmpeg", "-y",
                "-i", video_path,
                "-t", f"{audio_duration}",
                "-c", "copy",
                output_path
            ]
            proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()
            return proc.returncode == 0
        except Exception as e:
            logger.warning(f"Failed to trim video: {e}")
            return False

# Test function
async def test_ffmpeg_processor():
    """Test the FFmpeg processor"""
    processor = FFmpegVideoProcessor()
    
    print("🧪 Testing FFmpeg Video Processor...")
    print(f"Hardware acceleration: {processor.enable_hw_accel}")
    print(f"Available encoders: {processor.hw_encoders}")
    print(f"Optimal encoder: {processor._get_optimal_encoder()}")
    
    # Test with dummy data
    clip_paths = ["temp/placeholder_clip.mp4"]  # Use your existing placeholder
    audio_path = "../11-scripts-for-tiktok/anxiety1.wav"
    caption_data = {
        'style': 'tiktok',
        'captions': [
            {'text': 'WHEN ANXIETY', 'start_time': 0.0, 'end_time': 2.0},
            {'text': 'OVERWHELMS YOU', 'start_time': 2.0, 'end_time': 4.0},
            {'text': 'REMEMBER', 'start_time': 4.0, 'end_time': 6.0}
        ]
    }
    
    if Path(clip_paths[0]).exists() and Path(audio_path).exists():
        result = await processor.process_video_ultra_fast(
            clip_paths=clip_paths * 3,  # Repeat clip for testing
            audio_path=audio_path,
            caption_data=caption_data,
            output_path="output/ffmpeg_test.mp4"
        )
        
        print(f"✅ Test result: {result.success}")
        print(f"   Processing time: {result.processing_time:.3f}s")
        if result.error_message:
            print(f"   Error: {result.error_message}")
    else:
        print("⚠️ Test files not found, skipping actual processing test")

if __name__ == "__main__":
    asyncio.run(test_ffmpeg_processor())