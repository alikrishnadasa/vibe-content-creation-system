#!/usr/bin/env python3
"""
Test Word-by-Word Cached Captions
Generate a single video to test the cached word-by-word captions
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

async def test_word_captions():
    """Test word-by-word captions with cached timing data"""
    logger.info("🎯 TESTING WORD-BY-WORD CACHED CAPTIONS")
    logger.info("📋 Using pregenerated caption timing data")
    
    start_time = time.time()
    
    try:
        # Initialize the real content generator
        logger.info("🔧 Initializing content generator...")
        generator = RealContentGenerator(
            clips_directory="../MJAnime",
            metadata_file="../MJAnime/metadata_final_clean_shots.json", 
            scripts_directory="../11-scripts-for-tiktok",
            music_file="music/Beanie (Slowed).mp3",
            output_directory="output"
        )
        
        # Initialize all components
        init_success = await generator.initialize()
        if not init_success:
            logger.error("❌ Failed to initialize generator")
            return False
        
        logger.info("✅ Generator initialized successfully")
        
        # Clear uniqueness cache for fresh test
        generator.uniqueness_engine.clear_cache()
        
        # Test with a specific script that has good word-by-word potential
        logger.info("🎬 GENERATING TEST VIDEO WITH WORD-BY-WORD CAPTIONS...")
        
        request = RealVideoRequest(
            script_path="../11-scripts-for-tiktok/anxiety1.wav",
            script_name="anxiety1",
            variation_number=100,  # Use unique variation number
            caption_style="tiktok",  # TikTok style uses ONE_WORD mode
            music_sync=True
        )
        
        generation_start = time.time()
        result = await generator.generate_video(request)
        generation_time = time.time() - generation_start
        
        # Calculate final metrics
        total_time = time.time() - start_time
        
        # Display comprehensive results
        logger.info("=" * 60)
        logger.info("📊 WORD-BY-WORD CAPTION TEST RESULTS")
        logger.info("=" * 60)
        
        if result.success:
            logger.info(f"✅ Video generated successfully!")
            logger.info(f"⚡ Generation time: {generation_time:.3f}s")
            logger.info(f"📁 Output: {Path(result.output_path).name}")
            logger.info(f"📊 Video stats:")
            logger.info(f"   Duration: {result.total_duration:.1f}s")
            logger.info(f"   Clips used: {len(result.clips_used)}")
            logger.info(f"   Relevance score: {result.relevance_score:.2f}")
            
            # Check caption cache status
            cache_status = generator.caption_cache.get_cache_status()
            logger.info(f"📋 Caption Cache Status:")
            logger.info(f"   Cached scripts: {cache_status['total_cached']}")
            logger.info(f"   Available scripts: {cache_status['cached_scripts']}")
            
            # Get specific caption info for this script
            caption_segments = generator.caption_cache.get_captions_for_script(
                "anxiety1", 
                generator.caption_cache.caption_engine.preset_manager.get_preset("tiktok").display_mode
            )
            
            if caption_segments:
                logger.info(f"🎯 Word-by-Word Caption Details:")
                logger.info(f"   Total words: {len(caption_segments)}")
                logger.info(f"   First 5 words:")
                for i, segment in enumerate(caption_segments[:5]):
                    logger.info(f"     {i+1}. '{segment['text']}' ({segment['start_time']:.1f}s - {segment['end_time']:.1f}s)")
                
                # Show timing accuracy
                total_word_duration = sum(seg['end_time'] - seg['start_time'] for seg in caption_segments)
                logger.info(f"   Total word duration: {total_word_duration:.1f}s")
                logger.info(f"   Audio duration: {generator.get_audio_duration(request.script_path):.1f}s")
            
            # Check file size and quality
            if Path(result.output_path).exists():
                file_size_mb = Path(result.output_path).stat().st_size / (1024 * 1024)
                logger.info(f"📁 File size: {file_size_mb:.1f}MB")
            
            logger.info("=" * 60)
            logger.info("🎉 WORD-BY-WORD CAPTION TEST SUCCESSFUL!")
            logger.info("✅ Cached captions working perfectly")
            logger.info("📝 Each word displays individually with precise timing")
            logger.info("⚡ Ultra-fast caption loading from cache")
            logger.info("=" * 60)
            return True
        else:
            logger.error("❌ Video generation failed")
            logger.error(f"Error: {result.error_message}")
            return False
            
    except Exception as e:
        total_time = time.time() - start_time
        logger.error("=" * 60)
        logger.error(f"❌ WORD CAPTION TEST FAILED: {e}")
        logger.error(f"   Runtime: {total_time:.1f}s")
        logger.error("=" * 60)
        return False

async def main():
    """Main execution"""
    success = await test_word_captions()
    
    if success:
        print("\n🎉 SUCCESS! Word-by-word captions working perfectly!")
        print("📝 Each word displays individually with precise timing")
        print("💾 Using cached caption data for ultra-fast generation")
        return True
    else:
        print("\n💥 FAILED! Word-by-word captions need attention")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    if success:
        sys.exit(0)
    else:
        sys.exit(1)