#!/usr/bin/env python3
"""
Pregenerate and cache caption data for all 11 scripts
"""

import asyncio
import json
import logging
import time
from pathlib import Path
from core.real_content_generator import RealContentGenerator
from captions.preset_manager import CaptionDisplayMode

async def pregenerate_all_captions():
    """Create and cache caption data for all 11 scripts in all styles"""
    
    # Scripts to process
    scripts = [
        'anxiety1', 'safe1', 'miserable1', 'before', 'adhd', 
        'deadinside', 'diewithphone', '4', '6', '500friends', 'phone1.'
    ]
    
    # Caption styles to generate
    styles = ['tiktok', 'youtube', 'cinematic', 'minimal', 'karaoke']
    
    # Output directory for caption cache
    caption_cache_dir = Path("cache/pregenerated_captions")
    caption_cache_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📝 Pregenerating captions for {len(scripts)} scripts x {len(styles)} styles")
    print(f"📂 Cache directory: {caption_cache_dir}")
    
    # Initialize generator
    generator = RealContentGenerator(
        clips_directory='../MJAnime',
        metadata_file='../MJAnime/metadata_final_clean_shots.json',
        scripts_directory='../11-scripts-for-tiktok',
        music_file='music/Beanie (Slowed).mp3'
    )
    
    await generator.initialize()
    
    successful = 0
    failed = 0
    
    for script_name in scripts:
        print(f"\n🎯 Processing {script_name}")
        
        try:
            # Get script analysis
            script_analysis = generator.content_database.scripts_analyzer.get_script_analysis(script_name)
            if not script_analysis:
                print(f"❌ Script analysis not found: {script_name}")
                failed += len(styles)
                continue
            
            # Get script duration for sequence creation
            script_path = f"../11-scripts-for-tiktok/{script_name}.wav"
            script_duration = generator.get_audio_duration(script_path)
            
            for style in styles:
                try:
                    print(f"   📝 Generating {style} captions...")
                    
                    # Create a dummy sequence for caption generation
                    # We need this to pass to the caption generation function
                    from content.content_selector import SelectedSequence
                    from content.mjanime_loader import ClipMetadata
                    
                    # Create minimal dummy sequence with correct ClipMetadata structure
                    dummy_clips = [ClipMetadata(
                        id="dummy",
                        filename="dummy.mp4",
                        filepath="dummy.mp4",
                        tags=["neutral"],
                        duration=script_duration,
                        resolution="1080x1936",
                        fps=30.0,
                        file_size_mb=1.0,
                        shot_analysis={"lighting": "neutral", "camera_movement": "static", "shot_type": "medium_shot"},
                        created_at="2025-01-01",
                        emotional_tags=["neutral"],
                        lighting_type="neutral",
                        movement_type="static",
                        shot_type="medium_shot"
                    )]
                    
                    dummy_sequence = SelectedSequence(
                        clips=dummy_clips,
                        total_duration=script_duration,
                        relevance_score=1.0,
                        visual_variety_score=1.0,
                        music_sync_points=[],
                        sequence_hash="dummy",
                        selection_timestamp=time.time()
                    )
                    
                    # Generate captions for this style
                    caption_data = await generator._generate_captions(script_analysis, dummy_sequence, style)
                    
                    # Save to cache file
                    cache_filename = f"{script_name}_{style}_captions.json"
                    cache_path = caption_cache_dir / cache_filename
                    
                    with open(cache_path, 'w') as f:
                        json.dump(caption_data, f, indent=2)
                    
                    print(f"   ✅ Cached {len(caption_data['captions'])} {style} captions")
                    successful += 1
                    
                except Exception as e:
                    print(f"   ❌ Failed {style}: {e}")
                    failed += 1
                    
        except Exception as e:
            print(f"💥 Exception for {script_name}: {e}")
            failed += len(styles)
    
    print(f"\n🎉 Caption pregeneration complete!")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    
    # List generated files
    if successful > 0:
        print(f"\n📝 Caption cache files created:")
        for cache_file in sorted(caption_cache_dir.glob("*.json")):
            size_kb = cache_file.stat().st_size / 1024
            print(f"   {cache_file.name} ({size_kb:.1f}KB)")

if __name__ == "__main__":
    asyncio.run(pregenerate_all_captions())