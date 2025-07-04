#!/usr/bin/env python3
"""
Test script for burning captions into video for debugging
"""

import asyncio
from core.real_content_generator import RealContentGenerator, RealVideoRequest

async def test_burned_captions():
    """Test video generation with burned-in captions for debugging"""
    
    generator = RealContentGenerator(
        clips_directory='../MJAnime',
        metadata_file='../MJAnime/metadata_final_clean_shots.json',
        scripts_directory='../11-scripts-for-tiktok',
        music_file='music/Beanie (Slowed).mp3'
    )
    
    # Initialize generator
    print("🔧 Initializing generator...")
    await generator.initialize()
    
    # Test with anxiety1 script - create proper request object with burned captions
    request = RealVideoRequest(
        script_path='../11-scripts-for-tiktok/anxiety1.wav',
        script_name='anxiety1',
        variation_number=999,  # Use different variation to avoid duplication
        caption_style='tiktok',
        output_path='output/anxiety1_burned_captions_debug.mp4',
        burn_in_captions=True  # Enable burned-in captions for debugging
    )
    
    print("🎬 Generating video with burned-in captions...")
    
    result = await generator.generate_video(request)
    
    print(f'📊 Video generation result: {result.success}')
    if result.error_message:
        print(f'❌ Error: {result.error_message}')
    else:
        print(f'✅ Success! Output: {result.output_path}')
    print(f'⏱️  Processing time: {result.generation_time:.1f}s')
    
    if result.success:
        print(f"\n🎯 Debug video created: {result.output_path}")
        print("🔍 Captions are burned into the video - you can see them directly!")
        print("📱 Play the video to verify word-by-word caption timing")

if __name__ == "__main__":
    asyncio.run(test_burned_captions())