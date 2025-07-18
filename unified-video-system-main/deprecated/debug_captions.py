#!/usr/bin/env python3
"""
Simple caption debugging utility
"""

import asyncio
from core.real_content_generator import RealContentGenerator, RealVideoRequest

async def debug_captions(script_name: str, burn_captions: bool = True):
    """Generate a debug video with captions for testing"""
    
    generator = RealContentGenerator(
        clips_directory='../MJAnime',
        metadata_file='../MJAnime/metadata_final_clean_shots.json',
        scripts_directory='../11-scripts-for-tiktok',
        music_file='music/Beanie (Slowed).mp3'
    )
    
    await generator.initialize()
    
    request = RealVideoRequest(
        script_path=f'../11-scripts-for-tiktok/{script_name}.wav',
        script_name=script_name,
        variation_number=888,  # Debug variation
        caption_style='tiktok',
        output_path=f'output/{script_name}_debug_{"burned" if burn_captions else "subtitles"}.mp4',
        burn_in_captions=burn_captions
    )
    
    print(f"🎬 Generating {script_name} with {'burned' if burn_captions else 'subtitle'} captions...")
    
    result = await generator.generate_video(request)
    
    if result.success:
        print(f"✅ Success! Generated: {result.output_path}")
        print(f"⏱️  Processing time: {result.generation_time:.1f}s")
        return result.output_path
    else:
        print(f"❌ Failed: {result.error_message}")
        return None

if __name__ == "__main__":
    import sys
    
    script = sys.argv[1] if len(sys.argv) > 1 else "anxiety1"
    burn = sys.argv[2].lower() == "true" if len(sys.argv) > 2 else True
    
    result = asyncio.run(debug_captions(script, burn))
    
    if result:
        print(f"\n🔍 Debug video created: {result}")
        if burn:
            print("📱 Captions are burned into the video - visible when playing!")
        else:
            print("📱 Captions are in subtitle track - enable subtitles in player!")