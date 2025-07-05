#!/usr/bin/env python3
"""
Regenerate Captions with Whisper
Replace existing cached captions with precise Whisper word timing
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

from captions.script_caption_cache import ScriptCaptionCache

async def regenerate_with_whisper():
    """Regenerate all caption cache using Whisper transcriptions"""
    logger.info("🧠 REGENERATING CAPTIONS WITH WHISPER")
    logger.info("🎯 Target: Precise word-by-word timing for all 11 scripts")
    
    start_time = time.time()
    
    try:
        # Initialize cache with Whisper enabled
        logger.info("🔧 Initializing Whisper-enabled caption cache...")
        cache = ScriptCaptionCache(use_whisper=True)
        
        # Regenerate captions for all scripts with force=True
        scripts_directory = "../11-scripts-for-tiktok"
        scripts_dir = Path(scripts_directory)
        
        if not scripts_dir.exists():
            logger.error(f"Scripts directory not found: {scripts_directory}")
            return False
        
        # Find all .wav files
        audio_files = list(scripts_dir.glob("*.wav"))
        logger.info(f"🎵 Found {len(audio_files)} audio scripts")
        
        success_count = 0
        
        # Process each script with force regeneration
        for audio_file in audio_files:
            script_name = audio_file.stem
            try:
                await cache.generate_script_captions(
                    script_name, 
                    str(audio_file), 
                    force_regenerate=True
                )
                success_count += 1
            except Exception as e:
                logger.error(f"Failed to regenerate captions for {script_name}: {e}")
        
        # Save cache
        cache._save_cache()
        
        success = success_count == len(audio_files)
        
        total_time = time.time() - start_time
        
        if success:
            logger.info("=" * 60)
            logger.info("🎉 WHISPER CAPTION REGENERATION COMPLETE!")
            logger.info("=" * 60)
            
            # Show cache status
            status = cache.get_cache_status()
            logger.info(f"📊 Updated cache:")
            logger.info(f"   Scripts processed: {status['total_cached']}")
            logger.info(f"   Cache file: {status['cache_file']}")
            logger.info(f"⚡ Total time: {total_time:.2f}s")
            
            # Test a sample to show word timing accuracy
            if cache.cache:
                sample_script = list(cache.cache.keys())[0]
                sample_cache = cache.cache[sample_script]
                
                logger.info(f"📝 Sample: {sample_script}")
                logger.info(f"   Audio duration: {sample_cache.audio_duration:.1f}s")
                logger.info(f"   Words: {len(sample_cache.word_timings)}")
                
                # Show first few words with precise timing
                if sample_cache.one_word_segments:
                    logger.info("   First 5 words with Whisper timing:")
                    for i, segment in enumerate(sample_cache.one_word_segments[:5]):
                        duration = segment['end_time'] - segment['start_time']
                        logger.info(f"     {i+1}. '{segment['text']}' ({segment['start_time']:.2f}s - {segment['end_time']:.2f}s, {duration:.2f}s)")
            
            logger.info("=" * 60)
            logger.info("✅ All scripts now use precise Whisper word timing!")
            logger.info("🚀 Ready for ultra-accurate word-by-word captions")
            logger.info("=" * 60)
            
            return True
        else:
            logger.error("❌ Some scripts failed to regenerate")
            return False
            
    except Exception as e:
        total_time = time.time() - start_time
        logger.error("=" * 60)
        logger.error(f"❌ WHISPER REGENERATION FAILED: {e}")
        logger.error(f"   Runtime: {total_time:.1f}s")
        logger.error("=" * 60)
        return False

async def main():
    """Main execution"""
    success = await regenerate_with_whisper()
    
    if success:
        print("\n🎉 SUCCESS! All captions regenerated with Whisper precision!")
        print("🧠 Word timing is now based on actual audio analysis")
        print("📝 Each word displays exactly when spoken in the audio")
        return True
    else:
        print("\n💥 FAILED! Whisper caption regeneration needs attention")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    if success:
        sys.exit(0)
    else:
        sys.exit(1)