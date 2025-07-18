#!/usr/bin/env python3
"""
Convert Whisper Cache to Caption Cache
Directly convert existing Whisper transcriptions to pregenerated caption format
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_whisper_to_captions():
    """Convert all Whisper cache files to pregenerated caption format"""
    
    # Paths
    whisper_cache_dir = Path("/Users/jamesguo/vibe-content-creation/unified-video-system-main/cache/whisper")
    caption_cache_dir = Path("/Users/jamesguo/vibe-content-creation/unified-video-system-main/cache/pregenerated_captions")
    
    # Find all base whisper files (non-mixed)
    whisper_files = list(whisper_cache_dir.glob("*_base.json"))
    
    logger.info(f"📂 Found {len(whisper_files)} Whisper cache files")
    
    successful = 0
    failed = 0
    
    for whisper_file in whisper_files:
        try:
            # Extract script name (remove _base.json)
            script_name = whisper_file.stem.replace("_base", "")
            logger.info(f"🎯 Processing {script_name}")
            
            # Load Whisper transcription
            with open(whisper_file, 'r') as f:
                whisper_data = json.load(f)
            
            # Convert to caption format
            captions = []
            for word_data in whisper_data.get('words', []):
                caption = {
                    "text": word_data['word'],
                    "start_time": word_data['start'],
                    "end_time": word_data['end'],
                    "confidence": word_data.get('confidence', 1.0),
                    "style": "default"
                }
                captions.append(caption)
            
            # Create caption cache data
            caption_data = {
                "script_name": script_name,
                "style": "default",
                "total_captions": len(captions),
                "total_duration": whisper_data.get('duration', 0),
                "captions": captions,
                "metadata": {
                    "display_mode": "one_word",
                    "word_count": len(captions),
                    "generated_from": "whisper_transcription"
                }
            }
            
            # Save to pregenerated cache
            cache_filename = f"{script_name}_default_captions.json"
            cache_path = caption_cache_dir / cache_filename
            
            with open(cache_path, 'w') as f:
                json.dump(caption_data, f, indent=2)
            
            logger.info(f"   ✅ Created {len(captions)} word-level captions")
            successful += 1
            
        except Exception as e:
            logger.error(f"   ❌ Failed {script_name}: {e}")
            failed += 1
    
    logger.info("=" * 60)
    logger.info("🎉 WHISPER TO CAPTION CONVERSION COMPLETE!")
    logger.info("=" * 60)
    logger.info(f"✅ Successful: {successful}")
    logger.info(f"❌ Failed: {failed}")
    
    # List created files
    if successful > 0:
        logger.info(f"\n📝 Caption cache files created:")
        for cache_file in sorted(caption_cache_dir.glob("*_default_captions.json")):
            size_kb = cache_file.stat().st_size / 1024
            # Count captions
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                    caption_count = len(data.get('captions', []))
                logger.info(f"   {cache_file.name} ({size_kb:.1f}KB, {caption_count} captions)")
            except:
                logger.info(f"   {cache_file.name} ({size_kb:.1f}KB)")

if __name__ == "__main__":
    convert_whisper_to_captions()