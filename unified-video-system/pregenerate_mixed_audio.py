#!/usr/bin/env python3
"""
Pregenerate mixed audio files for all 11 scripts
"""

import asyncio
import logging
from pathlib import Path
from core.real_content_generator import RealContentGenerator

async def pregenerate_all_mixed_audio():
    """Create mixed audio files for all 11 scripts"""
    
    # Scripts to process
    scripts = [
        'anxiety1', 'safe1', 'miserable1', 'before', 'adhd', 
        'deadinside', 'diewithphone', '4', '6', '500friends', 'phone1.'
    ]
    
    # Output directory for mixed audio
    output_dir = Path("../11-scripts-for-tiktok")
    music_file = "music/Beanie (Slowed).mp3"
    
    print(f"🎵 Pregenerating mixed audio for {len(scripts)} scripts")
    print(f"🎼 Music: {music_file}")
    print(f"📂 Output: {output_dir}")
    
    # Initialize generator to get music parameters
    generator = RealContentGenerator(
        clips_directory='../MJAnime',
        metadata_file='../MJAnime/metadata_final_clean_shots.json',
        scripts_directory='../11-scripts-for-tiktok',
        music_file=music_file
    )
    
    await generator.initialize()
    
    successful = 0
    failed = 0
    
    for script_name in scripts:
        try:
            script_path = output_dir / f"{script_name}.wav"
            mixed_output_path = output_dir / f"{script_name}_mixed.wav"
            
            if not script_path.exists():
                print(f"❌ Script not found: {script_path}")
                failed += 1
                continue
            
            # Get script duration
            script_duration = generator.get_audio_duration(str(script_path))
            
            # Get music parameters (using same logic as video generation)
            music_params = generator.content_database.music_manager.prepare_for_mixing(script_duration)
            
            print(f"🎯 Processing {script_name} ({script_duration:.1f}s)")
            print(f"   Music: {music_params['start_time']:.1f}s - {music_params['start_time'] + script_duration:.1f}s")
            
            # Mix audio using FFmpeg
            mix_success = await generator._mix_audio_with_ffmpeg(
                script_path=str(script_path),
                music_path=music_params['music_file'],
                music_start_time=music_params['start_time'],
                music_duration=script_duration,
                output_path=str(mixed_output_path),
                script_volume=1.0,
                music_volume=music_params['volume_level']
            )
            
            if mix_success:
                print(f"✅ Created: {mixed_output_path.name}")
                successful += 1
            else:
                print(f"❌ Failed to mix: {script_name}")
                failed += 1
                
        except Exception as e:
            print(f"💥 Exception for {script_name}: {e}")
            failed += 1
    
    print(f"\n🎉 Pregeneration complete!")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    
    if successful > 0:
        print(f"\n📝 Mixed audio files created:")
        for script_name in scripts:
            mixed_path = output_dir / f"{script_name}_mixed.wav"
            if mixed_path.exists():
                size_mb = mixed_path.stat().st_size / (1024 * 1024)
                print(f"   {mixed_path.name} ({size_mb:.1f}MB)")

if __name__ == "__main__":
    asyncio.run(pregenerate_all_mixed_audio())