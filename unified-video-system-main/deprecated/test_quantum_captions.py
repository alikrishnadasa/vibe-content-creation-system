#!/usr/bin/env python3
"""
Test Quantum Pipeline with Caption Integration
Generate 1 video to verify captions are working
"""

import asyncio
import sys
import time
from pathlib import Path

# Add current directory to Python path
sys.path.append(str(Path(__file__).parent))

from core.quantum_pipeline import UnifiedQuantumPipeline
from video_config import DEFAULT_CONFIG

async def test_quantum_captions():
    """Test quantum pipeline with caption integration"""
    print("🎬 Testing Quantum Pipeline Caption Integration")
    print("=" * 50)
    
    # Initialize quantum pipeline
    print("🔧 Initializing quantum pipeline...")
    pipeline = UnifiedQuantumPipeline()
    
    # Initialize real content mode
    config = DEFAULT_CONFIG
    success = await pipeline.initialize_real_content_mode(
        clips_directory=config.clips_directory,
        metadata_file=config.metadata_file,
        scripts_directory=config.scripts_directory,
        music_file=config.music_file
    )
    
    if not success:
        print("❌ Failed to initialize real content mode")
        return False
    
    print("✅ Real content mode initialized")
    
    # Check available cached captions
    available_scripts = config.get_available_cached_scripts("default")
    print(f"📋 Scripts with cached captions: {len(available_scripts)}")
    for script in available_scripts[:5]:
        print(f"   • {script}")
    
    if not available_scripts:
        print("⚠️  No cached captions found!")
        return False
    
    # Test with a script that has cached captions
    test_script = "anxiety1"  # This has multiple caption styles available
    if test_script not in available_scripts:
        test_script = available_scripts[0]  # Use first available
    
    print(f"\n🎥 Testing caption integration with: {test_script}")
    
    # Generate video with captions
    start_time = time.time()
    result = await pipeline.generate_real_content_video(
        script_name=test_script,
        variation_number=1,
        caption_style="default"  # Use default style with cached captions
    )
    
    generation_time = time.time() - start_time
    
    if result['success']:
        print(f"✅ Caption test SUCCESS!")
        print(f"   Output: {result['output_path']}")
        print(f"   Time: {generation_time:.3f}s")
        print(f"   Target achieved: {result['target_achieved']}")
        
        if 'real_content_data' in result:
            print(f"   Clips used: {len(result['real_content_data']['clips_used'])}")
            print(f"   Duration: {result['real_content_data']['total_duration']:.1f}s")
        
        # Check if video file exists and has reasonable size
        output_path = Path(result['output_path'])
        if output_path.exists():
            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            print(f"   File size: {file_size_mb:.1f}MB")
            
            # Quick verification that captions were likely included
            if file_size_mb > 5:  # Reasonable size for video with captions
                print(f"✅ Video file appears to contain captions (good file size)")
            else:
                print(f"⚠️  Small file size - captions may not be included")
        
        return True
    else:
        print(f"❌ Caption test FAILED")
        print(f"   Error: {result.get('error', 'Unknown error')}")
        return False

async def main():
    """Main test function"""
    try:
        success = await test_quantum_captions()
        if success:
            print("\n✅ Caption integration test successful!")
            sys.exit(0)
        else:
            print("\n❌ Caption integration test failed!")
            sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test failed with exception: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())