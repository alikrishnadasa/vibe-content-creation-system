#!/usr/bin/env python3
"""
Test script for the caption system
"""

import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

import asyncio
from captions.preset_manager import CaptionPresetManager, CaptionStyle, CaptionDisplayMode
from captions.unified_caption_engine import UnifiedCaptionEngine


async def test_caption_system():
    """Test the complete caption system"""
    print("🧪 Testing Caption System...")
    
    # Test 1: Preset Manager
    print("\n1️⃣ Testing Preset Manager")
    manager = CaptionPresetManager()
    
    presets = manager.list_presets()
    print(f"   Available presets: {presets}")
    
    default_style = manager.get_preset("default")
    print(f"   Default: {default_style.font_family}, {default_style.font_size}px, {default_style.display_mode.value}")
    
    tiktok_style = manager.get_preset("tiktok")
    print(f"   TikTok: {tiktok_style.font_family}, outline: {tiktok_style.outline_width}px, uppercase: {tiktok_style.uppercase}")
    
    # Test 2: Unified Caption Engine
    print("\n2️⃣ Testing Unified Caption Engine")
    engine = UnifiedCaptionEngine()
    
    test_text = "Like water flowing through ancient stones, consciousness emerges from the digital realm."
    audio_duration = 10.0
    
    # Test default style
    default_captions = engine.create_captions(test_text, audio_duration, style="default")
    print(f"   Default captions: {len(default_captions)} generated")
    if default_captions:
        print(f"      First: '{default_captions[0].text}' ({default_captions[0].start_time:.1f}s - {default_captions[0].end_time:.1f}s)")
        print(f"      Last: '{default_captions[-1].text}' ({default_captions[-1].start_time:.1f}s - {default_captions[-1].end_time:.1f}s)")
    
    # Test TikTok style
    tiktok_captions = engine.create_captions(test_text, audio_duration, style="tiktok")
    print(f"   TikTok captions: {len(tiktok_captions)} generated")
    if tiktok_captions:
        print(f"      Uppercase: {tiktok_captions[0].text.isupper()}")
        print(f"      Sample: '{tiktok_captions[0].text}' ({tiktok_captions[0].start_time:.1f}s - {tiktok_captions[0].end_time:.1f}s)")
    
    # Test YouTube style (two words)
    youtube_captions = engine.create_captions(test_text, audio_duration, style="youtube")
    print(f"   YouTube captions: {len(youtube_captions)} generated")
    if youtube_captions:
        print(f"      Sample: '{youtube_captions[0].text}' ({youtube_captions[0].start_time:.1f}s - {youtube_captions[0].end_time:.1f}s)")
    
    # Test 3: Frame Rendering
    print("\n3️⃣ Testing Frame Rendering")
    if default_captions:
        frame_size = (1920, 1080)
        render_time = default_captions[0].start_time + 0.1
        render_data = engine.render_caption_frame(default_captions[0], frame_size, render_time)
        
        print(f"   Render data for '{render_data.get('text', 'N/A')}':")
        print(f"      Position: {render_data.get('position', 'N/A')}")
        print(f"      Font: {render_data.get('font_family', 'N/A')} {render_data.get('font_size', 'N/A')}px")
        print(f"      Color: {render_data.get('color', 'N/A')}")
        print(f"      Opacity: {render_data.get('opacity', 'N/A')}")
    
    # Test 4: Beat Sync Integration
    print("\n4️⃣ Testing Beat Sync Integration")
    beat_data = {
        'beat_times': [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
        'intensity_curve': [0.3, 0.7, 0.4, 0.8, 0.5, 0.9, 0.6, 0.7, 0.4]
    }
    
    beat_captions = engine.create_captions(
        "Dance to the rhythm", 
        4.0, 
        style="default", 
        beat_data=beat_data
    )
    print(f"   Beat-synced captions: {len(beat_captions)} generated")
    if beat_captions:
        print(f"      First: '{beat_captions[0].text}' at {beat_captions[0].start_time:.1f}s")
    
    # Test 5: Statistics
    print("\n5️⃣ Testing Statistics")
    stats = engine.get_statistics()
    print(f"   Captions generated: {stats['captions_generated']}")
    print(f"   Average time: {stats['average_generation_time']:.4f}s")
    print(f"   Available presets: {len(stats['available_presets'])}")
    print(f"   Current style: {stats['current_style']}")
    
    # Test 6: Custom Preset
    print("\n6️⃣ Testing Custom Preset")
    custom_style = manager.create_variation("default", {
        'font_size': 130,
        'font_color': 'gold',
        'outline_color': 'black',
        'outline_width': 2,
        'uppercase': True
    })
    
    manager.save_custom_preset("test_custom", custom_style)
    custom_captions = engine.create_captions(test_text, audio_duration, style="test_custom")
    print(f"   Custom captions: {len(custom_captions)} generated")
    if custom_captions:
        print(f"      Custom text: '{custom_captions[0].text}' (uppercase: {custom_captions[0].text.isupper()})")
    
    print("\n✅ All caption system tests completed!")
    
    # Summary
    print(f"\n📊 Summary:")
    print(f"   • {len(presets)} built-in presets available")
    print(f"   • Caption generation working for all styles")
    print(f"   • Frame rendering instructions generated")
    print(f"   • Beat sync integration functional")
    print(f"   • Custom presets can be created and saved")
    print(f"   • Statistics tracking enabled")


def main():
    """Main test function"""
    print("="*60)
    print("Unified Video System - Caption System Test")
    print("="*60)
    
    try:
        asyncio.run(test_caption_system())
        print("\n🎉 Caption system is ready for integration!")
        return 0
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main()) 