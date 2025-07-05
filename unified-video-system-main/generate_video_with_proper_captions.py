#!/usr/bin/env python3
"""
Generate Video with Proper Captions
Create video with word-by-word TikTok-style captions
"""

import logging
import asyncio
import sys
from pathlib import Path
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from core.real_content_generator import RealContentGenerator, RealVideoRequest
from captions.whisper_transcriber import WhisperTranscriber
from captions.unified_caption_engine import CaptionSyncEngine, WordTiming, Caption
from captions.preset_manager import CaptionDisplayMode, CaptionStyle, CaptionPosition

async def generate_video_with_captions():
    """Generate video with proper word-level captions"""
    logger.info("🎬 GENERATING VIDEO WITH PROPER CAPTIONS")
    
    start_time = time.time()
    
    try:
        # 1. Generate proper captions first
        logger.info("📝 Generating word-level captions...")
        transcriber = WhisperTranscriber()
        caption_sync = CaptionSyncEngine()
        
        # Get transcription for anxiety1
        script_name = "anxiety1"
        if script_name not in transcriber.transcription_cache:
            logger.error(f"No transcription found for {script_name}")
            return False
        
        transcription = transcriber.transcription_cache[script_name]
        logger.info(f"Found transcription with {len(transcription.words)} words")
        
        # Convert to WordTiming objects
        word_timings = []
        for whisper_word in transcription.words:
            word_timing = WordTiming(
                word=whisper_word.word,
                start=whisper_word.start,
                end=whisper_word.end,
                confidence=whisper_word.confidence
            )
            word_timings.append(word_timing)
        
        # Generate one-word captions (TikTok style)
        caption_segments = caption_sync.create_caption_segments(
            word_timings=word_timings,
            display_mode=CaptionDisplayMode.ONE_WORD
        )
        
        logger.info(f"Generated {len(caption_segments)} word-level caption segments")
        
        # Convert to Caption objects with TikTok style
        tiktok_style = CaptionStyle(
            font_family="HelveticaTextNow-ExtraBold",
            font_size=90,
            font_color="white",
            font_weight="extra-bold",
            display_mode=CaptionDisplayMode.ONE_WORD,
            position=CaptionPosition.CENTER,
            outline_color="black",
            outline_width=2
        )
        
        captions = []
        for segment in caption_segments:
            caption = Caption(
                text=segment['text'],
                start_time=segment['start_time'],
                end_time=segment['end_time'],
                style=tiktok_style,
                confidence=segment['confidence']
            )
            captions.append(caption)
        
        logger.info(f"Created {len(captions)} Caption objects")
        
        # Save captions as SRT file for the video system
        srt_content = ""
        for i, caption in enumerate(captions):
            start_h = int(caption.start_time // 3600)
            start_m = int((caption.start_time % 3600) // 60)
            start_s = int(caption.start_time % 60)
            start_ms = int((caption.start_time % 1) * 1000)
            
            end_h = int(caption.end_time // 3600)
            end_m = int((caption.end_time % 3600) // 60)
            end_s = int(caption.end_time % 60)
            end_ms = int((caption.end_time % 1) * 1000)
            
            start_srt = f"{start_h:02d}:{start_m:02d}:{start_s:02d},{start_ms:03d}"
            end_srt = f"{end_h:02d}:{end_m:02d}:{end_s:02d},{end_ms:03d}"
            
            srt_content += f"{i+1}\\n{start_srt} --> {end_srt}\\n{caption.text.upper()}\\n\\n"
        
        # Save SRT file
        srt_file = Path("temp/anxiety1_word_captions.srt")
        srt_file.parent.mkdir(exist_ok=True)
        with open(srt_file, 'w') as f:
            f.write(srt_content)
        
        logger.info(f"✅ Saved {len(captions)} captions to {srt_file}")
        
        # 2. Now generate the video
        logger.info("🔧 Initializing content generator...")
        generator = RealContentGenerator(
            clips_directory="/Users/jamesguo/vibe-content-creation/MJAnime",
            metadata_file="/Users/jamesguo/vibe-content-creation/MJAnime/metadata_final_clean_shots.json", 
            scripts_directory="/Users/jamesguo/vibe-content-creation/11-scripts-for-tiktok",
            music_file="music/Beanie (Slowed).mp3",
            output_directory="output"
        )
        
        # Initialize all components
        init_success = await generator.initialize()
        if not init_success:
            logger.error("❌ Failed to initialize generator")
            return False
        
        logger.info("✅ Generator initialized successfully")
        
        # Clear uniqueness cache
        generator.uniqueness_engine.clear_cache()
        
        # Generate video with word-level captions
        logger.info("🎬 Generating video with word-level captions...")
        
        request = RealVideoRequest(
            script_path="/Users/jamesguo/vibe-content-creation/11-scripts-for-tiktok/anxiety1.wav",
            script_name="anxiety1",
            variation_number=1,
            caption_style="tiktok",
            music_sync=True
        )
        
        video_start = time.time()
        result = await generator.generate_video(request, custom_captions=captions)
        video_time = time.time() - video_start
        
        total_time = time.time() - start_time
        
        if result.success:
            logger.info("=" * 60)
            logger.info("🎉 VIDEO WITH PROPER CAPTIONS SUCCESSFUL!")
            logger.info("=" * 60)
            logger.info(f"✅ Video: {Path(result.output_path).name}")
            logger.info(f"⚡ Generation time: {video_time:.2f}s")
            logger.info(f"🎯 Total time: {total_time:.2f}s")
            logger.info(f"📝 Word-level captions: {len(captions)}")
            logger.info(f"🎵 Audio duration: {result.total_duration:.1f}s")
            logger.info(f"🎬 Clips used: {len(result.clips_used)}")
            
            # Check file
            if Path(result.output_path).exists():
                file_size_mb = Path(result.output_path).stat().st_size / (1024 * 1024)
                logger.info(f"📁 File size: {file_size_mb:.1f}MB")
            
            logger.info("=" * 60)
            return True
        else:
            logger.error(f"❌ Video generation failed: {result.error_message}")
            return False
            
    except Exception as e:
        total_time = time.time() - start_time
        logger.error(f"💥 Generation failed: {e}")
        logger.error(f"   Runtime: {total_time:.1f}s")
        return False

async def main():
    """Main execution"""
    success = await generate_video_with_captions()
    
    if success:
        print("\\n🎉 SUCCESS! Video generated with proper word-level captions!")
        return True
    else:
        print("\\n💥 FAILED! Video generation encountered an error")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    if success:
        sys.exit(0)
    else:
        sys.exit(1)