"""
Real Content Video Generator

Orchestrates full pipeline from script to video with music integration.
Replaces synthetic content generation with intelligent real clip selection.
"""

import logging
import time
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from content.content_database import ContentDatabase
from content.content_selector import ContentSelector, SelectedSequence
from content.uniqueness_engine import UniquenessEngine
from core.optimized_video_processor import OptimizedVideoProcessor
from core.performance_optimizer import PerformanceOptimizer
from captions.unified_caption_engine import UnifiedCaptionEngine
from captions.script_caption_cache import ScriptCaptionCache
from beat_sync.beat_sync_engine import BeatSyncEngine

logger = logging.getLogger(__name__)

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    logger.warning("librosa not available - will use estimated durations")

@dataclass
class RealVideoRequest:
    """Request for real content video generation"""
    script_path: str
    script_name: str
    variation_number: int
    caption_style: str = "tiktok"
    music_sync: bool = True
    target_duration: Optional[float] = None
    min_clip_duration: float = 2.5
    output_path: Optional[str] = None
    burn_in_captions: bool = False  # Debug option - burn captions into video

@dataclass
class RealVideoResult:
    """Result of real content video generation"""
    success: bool
    output_path: str
    generation_time: float
    sequence_hash: str
    clips_used: List[str]
    total_duration: float
    relevance_score: float
    visual_variety_score: float
    error_message: Optional[str] = None

class RealContentGenerator:
    """Orchestrates real content video generation pipeline"""
    
    def __init__(self, 
                 clips_directory: str,
                 metadata_file: str,
                 scripts_directory: str,
                 music_file: str,
                 output_directory: str = "output"):
        """
        Initialize real content generator
        
        Args:
            clips_directory: Path to MJAnime clips
            metadata_file: Path to clips metadata
            scripts_directory: Path to audio scripts
            music_file: Path to universal music track
            output_directory: Output directory for videos
        """
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(exist_ok=True)
        
        # Initialize content management
        self.content_database = ContentDatabase(
            clips_directory, metadata_file, scripts_directory, music_file
        )
        self.content_selector = ContentSelector(self.content_database)
        self.uniqueness_engine = UniquenessEngine()
        
        # Initialize processing engines
        self.performance_optimizer = None
        self.video_processor = None
        self.caption_engine = UnifiedCaptionEngine()  # Initialize the caption engine
        self.caption_cache = ScriptCaptionCache()  # Initialize caption cache
        self.beat_sync_engine = None
        
        self.loaded = False
    
    def get_audio_duration(self, audio_path: str) -> float:
        """
        Get the actual duration of an audio file
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Duration in seconds
        """
        try:
            if LIBROSA_AVAILABLE:
                # Use librosa to get accurate duration
                duration = librosa.get_duration(path=audio_path)
                logger.info(f"Audio duration detected: {duration:.1f}s for {Path(audio_path).name}")
                return duration
            else:
                # Fallback: estimate based on file size (very rough)
                file_size = Path(audio_path).stat().st_size
                # Rough estimate: ~1MB per minute for compressed audio
                estimated_duration = (file_size / 1024 / 1024) * 60
                logger.warning(f"Estimated audio duration: {estimated_duration:.1f}s for {Path(audio_path).name}")
                return estimated_duration
        except Exception as e:
            logger.error(f"Failed to get audio duration for {audio_path}: {e}")
            # Default fallback
            return 15.0
        
    async def initialize(self) -> bool:
        """
        Initialize all components
        
        Returns:
            bool: True if successful
        """
        try:
            logger.info("Initializing real content generator...")
            
            # Load content database
            try:
                if not await self.content_database.load_all_content():
                    logger.error("Failed to load content database")
                    return False
                logger.info("✅ Content database loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load content database: {e}")
                return False
            
            # Initialize processing components (these will be set up when needed)
            # Performance optimizer and other components are initialized by the quantum pipeline
            
            self.loaded = True
            logger.info("Real content generator initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize real content generator: {e}")
            return False
    
    async def generate_video(self, request: RealVideoRequest) -> RealVideoResult:
        """
        Generate a single real content video
        
        Args:
            request: Video generation request
            
        Returns:
            Video generation result
        """
        if not self.loaded:
            raise RuntimeError("Generator not initialized. Call initialize() first.")
        
        start_time = time.time()
        
        try:
            logger.info(f"Generating video for {request.script_name} variation {request.variation_number}")
            
            # Step 1: Analyze script
            script_analysis = self.content_database.scripts_analyzer.get_script_analysis(request.script_name)
            if not script_analysis:
                return RealVideoResult(
                    success=False,
                    output_path="",
                    generation_time=time.time() - start_time,
                    sequence_hash="",
                    clips_used=[],
                    total_duration=0.0,
                    relevance_score=0.0,
                    visual_variety_score=0.0,
                    error_message=f"Script analysis not found: {request.script_name}"
                )
            
            # Step 2: Always use actual script duration from audio file
            script_duration = self.get_audio_duration(request.script_path)
            logger.info(f"Using actual audio duration: {script_duration:.1f}s (ignoring any target duration parameter)")
            
            # Get music synchronization data
            music_beats = []
            if request.music_sync:
                music_info = self.content_database.music_manager.get_track_info()
                if music_info.get('beats_analyzed'):
                    # Get beats for actual script duration
                    music_beats = self.content_database.music_manager.get_beat_timing(0, script_duration)
            
            # Step 3: Select content sequence based on actual script duration
            sequence = await self.content_selector.select_clips_for_script(
                script_analysis, 
                script_duration=script_duration,  # Use actual script duration
                music_beats=music_beats,
                min_clip_duration=request.min_clip_duration,
                variation_seed=hash(f"{request.script_name}_{request.variation_number}")
            )
            
            # Step 4: Check and register uniqueness
            if not await self.uniqueness_engine.register_sequence(sequence, request.script_name, request.variation_number):
                logger.warning(f"Sequence already exists, generating alternative...")
                # Try generating a different sequence
                sequences = self.content_selector.generate_multiple_sequences(script_analysis, count=5, script_duration=script_duration, music_beats=music_beats)
                for seq in sequences:
                    if await self.uniqueness_engine.register_sequence(seq, request.script_name, request.variation_number):
                        sequence = seq
                        break
                else:
                    return RealVideoResult(
                        success=False,
                        output_path="",
                        generation_time=time.time() - start_time,
                        sequence_hash=sequence.sequence_hash,
                        clips_used=[clip.id for clip in sequence.clips],
                        total_duration=sequence.total_duration,
                        relevance_score=sequence.relevance_score,
                        visual_variety_score=sequence.visual_variety_score,
                        error_message="Could not generate unique sequence"
                    )
            
            # Step 5: Generate captions
            caption_data = await self._generate_captions(script_analysis, sequence, request.caption_style)
            
            # Step 6: Process video with music
            output_path = await self._process_video_with_music(
                sequence, 
                script_analysis,
                caption_data,
                request,
                music_beats,
                script_duration  # Pass the actual script duration
            )
            
            generation_time = time.time() - start_time
            
            logger.info(f"✅ Video generated successfully in {generation_time:.3f}s: {Path(output_path).name}")
            
            return RealVideoResult(
                success=True,
                output_path=output_path,
                generation_time=generation_time,
                sequence_hash=sequence.sequence_hash,
                clips_used=[clip.id for clip in sequence.clips],
                total_duration=sequence.total_duration,
                relevance_score=sequence.relevance_score,
                visual_variety_score=sequence.visual_variety_score
            )
            
        except Exception as e:
            generation_time = time.time() - start_time
            logger.error(f"Video generation failed: {e}")
            
            return RealVideoResult(
                success=False,
                output_path="",
                generation_time=generation_time,
                sequence_hash="",
                clips_used=[],
                total_duration=0.0,
                relevance_score=0.0,
                visual_variety_score=0.0,
                error_message=str(e)
            )
    
    async def _generate_captions(self, script_analysis, sequence: SelectedSequence, style: str) -> Dict[str, Any]:
        """Generate captions using pregenerated cache files"""
        try:
            # Extract script name from filename
            script_name = Path(script_analysis.filename).stem
            logger.info(f"Loading pregenerated captions for script: {script_name} in {style} style")
            
            # Try to load pregenerated caption file
            caption_cache_dir = Path("cache/pregenerated_captions")
            cache_filename = f"{script_name}_{style}_captions.json"
            cache_path = caption_cache_dir / cache_filename
            
            if cache_path.exists():
                import json
                with open(cache_path, 'r') as f:
                    caption_data = json.load(f)
                    
                logger.info(f"✅ Loaded {len(caption_data['captions'])} pregenerated {style} captions")
                return caption_data
            else:
                logger.warning(f"Pregenerated captions not found: {cache_filename}")
                # Fallback to old method
                return await self._generate_captions_fallback(script_analysis, sequence, style)
            
        except Exception as e:
            logger.error(f"Pregenerated caption loading failed: {e}")
            # Fallback to old method
            return await self._generate_captions_fallback(script_analysis, sequence, style)
    
    async def _generate_captions_fallback(self, script_analysis, sequence: SelectedSequence, style: str) -> Dict[str, Any]:
        """Fallback caption generation using cached word-by-word timing data"""
        try:
            # Extract script name from filename
            script_name = Path(script_analysis.filename).stem
            logger.info(f"Using fallback caption generation for: {script_name} in {style} style")
            
            # Get cached caption segments (word-by-word for tiktok/default style)
            from captions.preset_manager import CaptionDisplayMode
            
            # Map style to display mode - use default (not uppercase) for most styles
            display_mode = CaptionDisplayMode.ONE_WORD  # Default to word-by-word
            if style == "youtube":
                display_mode = CaptionDisplayMode.TWO_WORDS
            elif style == "cinematic":
                display_mode = CaptionDisplayMode.PHRASE_BASED
            elif style == "minimal":
                display_mode = CaptionDisplayMode.FULL_SENTENCE
            elif style == "karaoke":
                display_mode = CaptionDisplayMode.KARAOKE
            
            # Get cached caption segments
            caption_segments = self.caption_cache.get_captions_for_script(script_name, display_mode)
            
            if not caption_segments:
                logger.warning(f"No cached captions found for {script_name}, falling back to generation")
                return self._generate_fallback_captions(script_analysis, sequence, style)
            
            # Convert cached segments to video processor format
            caption_timing = []
            for segment in caption_segments:
                caption_timing.append({
                    'text': segment['text'].lower(),  # Always use lowercase
                    'start_time': segment['start_time'],
                    'end_time': segment['end_time'],
                    'style': style,
                    'confidence': segment.get('confidence', 1.0),
                    'x_position': None,
                    'y_position': None
                })
            
            logger.info(f"✅ Loaded {len(caption_timing)} cached {display_mode.value} captions")
            
            return {
                'style': style,
                'captions': caption_timing,
                'total_duration': sequence.total_duration,
                'source': 'cached',
                'display_mode': display_mode.value
            }
            
        except Exception as e:
            logger.error(f"Cached caption loading failed: {e}")
            # Fallback to simple captions
            return self._generate_fallback_captions(script_analysis, sequence, style)
    
    def _extract_text_from_filename(self, filename: str) -> str:
        """Extract meaningful text from script filename for caption generation"""
        # Remove file extension and common prefixes
        base_name = Path(filename).stem.lower()
        
        # Map common script names to meaningful text
        filename_text_map = {
            'anxiety1': "When anxiety overwhelms you, remember that this feeling is temporary. You have the strength to overcome this moment.",
            'safe1': "You are safe. You are loved. You are exactly where you need to be in this moment.",
            'miserable1': "Even in the darkest moments, there is light within you waiting to shine. You are stronger than you know.",
            'before': "Before you make that decision, take a breath. Consider the possibilities that await you.",
            'adhd': "Your mind works differently, and that's your superpower. Embrace your unique way of thinking.",
            'deadinside': "Feeling numb doesn't mean you're broken. It means you're protecting yourself while you heal.",
            'diewithphone': "Put down the phone. Look up. The real world is waiting for your presence.",
            'phone1': "Your worth isn't measured by likes or follows. You are valuable just as you are.",
            '4': "Four simple words can change everything: You are not alone.",
            '6': "Six breaths. In and out. Feel yourself returning to the present moment.",
            '500friends': "500 friends online, but do you feel truly connected? Quality over quantity in relationships.",
            'diewithphone': "The digital world can wait. Your mental health cannot."
        }
        
        # Try to find a match
        for key, text in filename_text_map.items():
            if key in base_name:
                return text
        
        # Generic fallback based on detected emotion
        return f"This is a moment of {base_name}. You have the power to transform how you feel."
    
    def _generate_fallback_captions(self, script_analysis, sequence: SelectedSequence, style: str) -> Dict[str, Any]:
        """Generate simple fallback captions when UnifiedCaptionEngine fails"""
        try:
            # Simple emotion-based captions as fallback
            emotion_captions = {
                'anxiety': ["Breathe deeply", "You are safe", "This will pass"],
                'peace': ["Find your center", "Peace within", "Gentle moments"],
                'seeking': ["Keep searching", "Your path awaits", "Trust the journey"],
                'awakening': ["Consciousness expands", "Truth emerges", "Awakening begins"],
                'safe': ["You are safe", "Protected space", "Secure moments"],
                'miserable': ["You matter", "This will change", "Healing comes"]
            }
            
            caption_texts = emotion_captions.get(script_analysis.primary_emotion, ["Inner wisdom", "Sacred moments", "Divine connection"])
            
            # Generate timing
            caption_timing = []
            current_time = 0.0
            clip_duration_per_caption = sequence.total_duration / len(caption_texts)
            
            for i, text in enumerate(caption_texts):
                start_time = current_time
                end_time = current_time + clip_duration_per_caption
                caption_timing.append({
                    'text': text,
                    'start_time': start_time,
                    'end_time': end_time,
                    'clip_index': i % len(sequence.clips)
                })
                current_time = end_time
            
            return {
                'style': style,
                'captions': caption_timing,
                'total_duration': sequence.total_duration
            }
            
        except Exception as e:
            logger.error(f"Fallback caption generation failed: {e}")
            return {'style': style, 'captions': [], 'total_duration': sequence.total_duration}
    
    async def _process_video_with_music(self, 
                                       sequence: SelectedSequence,
                                       script_analysis,
                                       caption_data: Dict[str, Any],
                                       request: RealVideoRequest,
                                       music_beats: List[float],
                                       script_duration: float) -> str:
        """Process video with real clips, captions, and music"""
        
        # Generate output filename
        timestamp = int(time.time())
        output_filename = f"real_content_{request.script_name}_var{request.variation_number}_{timestamp}.mp4"
        output_path = self.output_directory / output_filename
        
        # Prepare clip paths
        clip_paths = [clip.filepath for clip in sequence.clips]
        
        # Prepare music mixing parameters
        music_params = self.content_database.music_manager.prepare_for_mixing(sequence.total_duration)
        
        # Create actual video using real asset processor
        
        logger.info(f"Processing video with {len(clip_paths)} clips and music")
        logger.info(f"Music: {music_params['music_file']} ({music_params['start_time']:.1f}s - {music_params['end_time']:.1f}s)")
        logger.info(f"Captions: {len(caption_data['captions'])} captions in {caption_data['style']} style")
        
        # Use FFmpeg processor for ultra-fast video generation
        from .ffmpeg_video_processor import FFmpegVideoProcessor
        processor = FFmpegVideoProcessor(enable_hardware_acceleration=True, ultra_speed_mode=False)
        
        # Step 1: Use pregenerated mixed audio file
        mixed_audio_path = self._get_pregenerated_mixed_audio(request.script_name, request.script_path)
        
        # Step 2: Create final video with clips, mixed audio, and captions using FFmpeg
        # Use actual script duration instead of sequence total_duration which uses clip durations
        
        composition_result = await processor.process_video_ultra_fast(
            clip_paths=clip_paths,
            audio_path=str(mixed_audio_path),
            caption_data=caption_data,
            output_path=str(output_path),
            target_resolution=(1080, 1936),  # TikTok format
            beat_sync_points=sequence.music_sync_points,  # Pass beat synchronization data
            target_duration=script_duration,  # Use actual script duration, not sequence.total_duration
            burn_in_captions=getattr(request, 'burn_in_captions', False)  # Debug option
        )
        
        if not composition_result.success:
            logger.error(f"Video composition failed: {composition_result.error_message}")
            # Create placeholder as fallback
            await self._create_placeholder_video(output_path, sequence, caption_data, music_params, clip_paths, music_beats)
        else:
            logger.info(f"FFmpeg video composition successful in {composition_result.processing_time:.3f}s")
        
        return str(output_path)
    
    async def _mix_audio_with_ffmpeg(self,
                                   script_path: str,
                                   music_path: str, 
                                   music_start_time: float,
                                   music_duration: float,
                                   output_path: str,
                                   script_volume: float = 1.0,
                                   music_volume: float = 0.3) -> bool:
        """Mix script audio with background music using FFmpeg"""
        try:
            cmd = [
                "ffmpeg", "-y",
                "-i", script_path,  # Script audio (foreground)
                "-i", music_path,   # Background music
                "-filter_complex", 
                f"[1:a]atrim=start={music_start_time}:duration={music_duration},volume={music_volume}[music];"
                f"[0:a]volume={script_volume}[script];"
                f"[script][music]amix=inputs=2:duration=first:dropout_transition=2[out]",
                "-map", "[out]",
                "-c:a", "pcm_s16le",  # Uncompressed audio for quality
                output_path
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                logger.info("Audio mixing successful with FFmpeg")
                return True
            else:
                error_msg = stderr.decode() if stderr else "Unknown audio mixing error"
                logger.warning(f"FFmpeg audio mixing failed: {error_msg}")
                return False
                
        except Exception as e:
            logger.error(f"Audio mixing exception: {e}")
            return False
    
    async def _get_or_create_cached_audio(self,
                                        script_path: str,
                                        script_name: str,
                                        music_params: Dict[str, Any],
                                        script_duration: float) -> str:
        """Get cached mixed audio or create and cache it"""
        try:
            # Create cache directory for mixed audio
            cache_dir = Path("cache/mixed_audio")
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate cache key based on script and music parameters
            music_file = Path(music_params['music_file']).stem
            cache_key = f"{script_name}_{music_file}_{music_params['start_time']:.1f}_{music_params['volume_level']:.2f}"
            cached_audio_path = cache_dir / f"{cache_key}.wav"
            
            # Check if cached version exists and is newer than source script
            if cached_audio_path.exists():
                script_mtime = Path(script_path).stat().st_mtime
                cache_mtime = cached_audio_path.stat().st_mtime
                
                if cache_mtime > script_mtime:
                    logger.info(f"Using cached mixed audio: {cached_audio_path.name}")
                    return str(cached_audio_path)
                else:
                    logger.info(f"Cache outdated, regenerating mixed audio for {script_name}")
            else:
                logger.info(f"Creating new mixed audio cache for {script_name}")
            
            # Create mixed audio and cache it
            mix_success = await self._mix_audio_with_ffmpeg(
                script_path=script_path,
                music_path=music_params['music_file'],
                music_start_time=music_params['start_time'],
                music_duration=script_duration,  # Use actual script duration
                output_path=str(cached_audio_path),
                script_volume=1.0,
                music_volume=music_params['volume_level']
            )
            
            if mix_success:
                logger.info(f"✅ Cached mixed audio: {cached_audio_path.name}")
                return str(cached_audio_path)
            else:
                logger.warning("Audio mixing failed, using script audio only")
                return script_path
                
        except Exception as e:
            logger.error(f"Audio caching failed: {e}")
            return script_path
    
    def _get_pregenerated_mixed_audio(self, script_name: str, script_path: str) -> str:
        """Get pregenerated mixed audio file or fallback to original script"""
        try:
            # Look for pregenerated mixed audio file
            script_dir = Path(script_path).parent
            mixed_audio_path = script_dir / f"{script_name}_mixed.wav"
            
            if mixed_audio_path.exists():
                logger.info(f"Using pregenerated mixed audio: {mixed_audio_path.name}")
                return str(mixed_audio_path)
            else:
                logger.warning(f"Pregenerated mixed audio not found: {mixed_audio_path.name}")
                logger.info(f"Fallback to original script audio: {Path(script_path).name}")
                return script_path
                
        except Exception as e:
            logger.error(f"Error accessing pregenerated audio: {e}")
            return script_path
    
    async def _create_placeholder_video(self, output_path, sequence, caption_data, music_params, clip_paths, music_beats):
        """Create placeholder video when real processing fails"""
        with open(output_path, 'w') as f:
            f.write(f"# Real Content Video (Placeholder - MoviePy processing failed)\n")
            f.write(f"Sequence Hash: {sequence.sequence_hash}\n")
            f.write(f"Clips Used: {len(clip_paths)}\n")
            f.write(f"Duration: {sequence.total_duration:.1f}s\n")
            f.write(f"Relevance Score: {sequence.relevance_score:.2f}\n")
            f.write(f"Visual Variety: {sequence.visual_variety_score:.2f}\n")
            f.write(f"Music Sync: {len(music_beats)} beats\n")
            f.write(f"Captions: {len(caption_data['captions'])}\n")
            f.write(f"\nClip Details:\n")
            for i, clip in enumerate(sequence.clips):
                f.write(f"  {i+1}. {clip.filename}\n")
                f.write(f"     Emotion: {clip.emotional_tags}\n")
                f.write(f"     Lighting: {clip.lighting_type}\n")
                f.write(f"     Duration: {clip.duration:.1f}s\n")
            f.write(f"\nCaption Details:\n")
            for i, cap in enumerate(caption_data['captions']):
                f.write(f"  {i+1}. {cap['start_time']:.1f}s-{cap['end_time']:.1f}s: {cap['text']}\n")
    
    async def generate_batch_videos(self, 
                                   script_names: List[str],
                                   variations_per_script: int = 5,
                                   caption_style: str = "tiktok") -> List[RealVideoResult]:
        """
        Generate multiple videos in batch for testing phase
        
        Args:
            script_names: List of script names to process
            variations_per_script: Number of variations per script
            caption_style: Caption style to use
            
        Returns:
            List of generation results
        """
        if not self.loaded:
            raise RuntimeError("Generator not initialized. Call initialize() first.")
        
        logger.info(f"Generating batch: {len(script_names)} scripts × {variations_per_script} variations = {len(script_names) * variations_per_script} videos")
        
        results = []
        total_start_time = time.time()
        
        for script_name in script_names:
            logger.info(f"Processing script: {script_name}")
            
            for variation in range(1, variations_per_script + 1):
                request = RealVideoRequest(
                    script_path=f"../11-scripts-for-tiktok/{script_name}.wav",
                    script_name=script_name,
                    variation_number=variation,
                    caption_style=caption_style,
                    music_sync=True
                )
                
                result = await self.generate_video(request)
                results.append(result)
                
                if result.success:
                    logger.info(f"  ✅ Variation {variation}: {result.generation_time:.3f}s")
                else:
                    logger.error(f"  ❌ Variation {variation}: {result.error_message}")
        
        total_time = time.time() - total_start_time
        successful = sum(1 for r in results if r.success)
        
        logger.info(f"Batch complete: {successful}/{len(results)} successful in {total_time:.2f}s")
        logger.info(f"Average per video: {total_time/len(results):.3f}s")
        
        return results
    
    def get_generator_stats(self) -> Dict[str, Any]:
        """Get statistics about the generator"""
        if not self.loaded:
            return {'status': 'not_initialized'}
        
        content_stats = self.content_database.get_database_stats()
        uniqueness_stats = self.uniqueness_engine.get_clip_usage_stats()
        
        return {
            'status': 'initialized',
            'clips_available': content_stats['clips']['total_clips'],
            'scripts_available': content_stats['scripts']['total_scripts'],
            'music_track': content_stats['music']['filename'],
            'total_combinations_possible': content_stats['total_possible_combinations'],
            'unique_sequences_generated': uniqueness_stats['total_unique_sequences'],
            'clips_utilized': uniqueness_stats['total_clips_used'],
            'output_directory': str(self.output_directory)
        }