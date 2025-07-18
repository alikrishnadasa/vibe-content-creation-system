#!/usr/bin/env python3
"""
Pregenerate and cache caption data for all scripts
Focuses on 'default' style used in video generation, with option to include other styles
"""

import asyncio
import json
import logging
import time
import argparse
from pathlib import Path
from core.real_content_generator import RealContentGenerator
from captions.preset_manager import CaptionDisplayMode

async def pregenerate_all_captions(include_all_styles=False):
    """Create and cache caption data for all 11 scripts in all styles"""
    
    # Scripts to process
    scripts = [
        'anxiety1', 'safe1', 'miserable1', 'before', 'adhd', 
        'deadinside', 'diewithphone', '4', '6', '500friends', 'phone1.'
    ]
    
    # Caption styles to generate
    if include_all_styles:
        styles = ['default', 'tiktok', 'youtube', 'cinematic', 'minimal', 'karaoke']
        print("🎨 Generating ALL caption styles")
    else:
        styles = ['default']
        print("🎯 Generating ONLY 'default' style (used in video generation)")
        print("   💡 Use --all-styles flag to generate all styles")
    
    # Output directory for caption cache
    caption_cache_dir = Path("/Users/jamesguo/vibe-content-creation/unified-video-system-main/cache/pregenerated_captions")
    caption_cache_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📝 Pregenerating captions for {len(scripts)} scripts x {len(styles)} styles")
    print(f"📂 Cache directory: {caption_cache_dir}")
    
    # Initialize generator with absolute paths
    generator = RealContentGenerator(
        clips_directory='/Users/jamesguo/vibe-content-creation/MJAnime',
        metadata_file='/Users/jamesguo/vibe-content-creation/MJAnime/metadata_final_clean_shots.json',
        scripts_directory='/Users/jamesguo/vibe-content-creation/11-scripts-for-tiktok',
        music_file='/Users/jamesguo/vibe-content-creation/unified-video-system-main/music/Beanie (Slowed).mp3',
        output_directory='/Users/jamesguo/vibe-content-creation/unified-video-system-main/output'
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
            script_path = f"/Users/jamesguo/vibe-content-creation/11-scripts-for-tiktok/{script_name}.wav"
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
                        fps=24.0,
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

def main():
    """Main execution with command line arguments"""
    parser = argparse.ArgumentParser(description="Pregenerate caption cache files for all scripts")
    parser.add_argument("--all-styles", action="store_true", 
                       help="Generate all caption styles (default: only 'default' style)")
    
    args = parser.parse_args()
    
    if args.all_styles:
        print("🎨 Generating captions for ALL styles...")
    else:
        print("🎯 Generating captions for 'default' style only...")
        print("   (This is the style used in video generation)")
    
    asyncio.run(pregenerate_all_captions(include_all_styles=args.all_styles))

if __name__ == "__main__":
    main()