#!/usr/bin/env python3
"""
Test Quantum Pipeline - Generate 5 videos using the quantum pipeline
"""

import asyncio
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

# Add current directory to Python path
sys.path.append(str(Path(__file__).parent))

from core.quantum_pipeline import UnifiedQuantumPipeline
from video_config import DEFAULT_CONFIG

async def test_quantum_pipeline_generation():
    """Test the quantum pipeline by generating 5 videos"""
    print("🚀 Testing Quantum Pipeline - Generating 5 videos")
    print("=" * 60)
    
    # Initialize quantum pipeline
    print("🔧 Initializing quantum pipeline...")
    pipeline = UnifiedQuantumPipeline()
    
    # Get available scripts
    config = DEFAULT_CONFIG
    available_scripts = config.get_available_scripts()
    
    if not available_scripts:
        print("❌ No scripts found in scripts directory")
        return False
    
    print(f"📁 Found {len(available_scripts)} available scripts")
    for i, script in enumerate(available_scripts[:5]):
        print(f"   {i+1}. {script.stem}")
    
    # Initialize real content mode
    print("\n🎬 Initializing real content mode...")
    success = await pipeline.initialize_real_content_mode(
        clips_directory=config.clips_directory,
        metadata_file=config.metadata_file,
        scripts_directory=config.scripts_directory,
        music_file=config.music_file
    )
    
    if not success:
        print("❌ Failed to initialize real content mode")
        return False
    
    print("✅ Real content mode initialized successfully")
    
    # Test generation for 5 videos
    print("\n🎥 Generating 5 test videos...")
    results = []
    total_start_time = time.time()
    
    for i in range(5):
        # Use first 5 scripts (or cycle through if fewer)
        script_file = available_scripts[i % len(available_scripts)]
        script_name = script_file.stem
        variation = i + 1
        
        print(f"\n--- Video {i+1}/5: {script_name} (variation {variation}) ---")
        
        try:
            # Generate video using quantum pipeline
            result = await pipeline.generate_real_content_video(
                script_name=script_name,
                variation_number=variation,
                caption_style="default"
            )
            
            results.append({
                'video_number': i + 1,
                'script_name': script_name,
                'variation': variation,
                'result': result
            })
            
            if result['success']:
                print(f"✅ Video {i+1} SUCCESS")
                print(f"   Output: {result['output_path']}")
                print(f"   Time: {result['processing_time']:.3f}s")
                print(f"   Target achieved: {result['target_achieved']}")
                if 'real_content_data' in result:
                    print(f"   Clips used: {len(result['real_content_data']['clips_used'])}")
                    print(f"   Relevance: {result['real_content_data']['relevance_score']:.2f}")
                    print(f"   Variety: {result['real_content_data']['visual_variety_score']:.2f}")
            else:
                print(f"❌ Video {i+1} FAILED")
                print(f"   Error: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"💥 Video {i+1} EXCEPTION: {e}")
            results.append({
                'video_number': i + 1,
                'script_name': script_name,
                'variation': variation,
                'result': {'success': False, 'error': str(e), 'processing_time': 0}
            })
    
    # Calculate overall statistics
    total_time = time.time() - total_start_time
    successful_videos = sum(1 for r in results if r['result']['success'])
    failed_videos = len(results) - successful_videos
    
    print("\n" + "=" * 60)
    print("📊 QUANTUM PIPELINE TEST RESULTS")
    print("=" * 60)
    
    print(f"✅ Successful videos: {successful_videos}/5")
    print(f"❌ Failed videos: {failed_videos}/5")
    print(f"📈 Success rate: {(successful_videos/5)*100:.1f}%")
    print(f"⏱️  Total time: {total_time:.2f}s")
    print(f"⚡ Average time per video: {total_time/5:.3f}s")
    
    # Calculate processing times for successful videos
    successful_times = [r['result']['processing_time'] for r in results if r['result']['success']]
    if successful_times:
        avg_processing_time = sum(successful_times) / len(successful_times)
        fastest_time = min(successful_times)
        slowest_time = max(successful_times)
        target_achieved = sum(1 for t in successful_times if t <= 0.7)
        
        print(f"🎯 Average processing time: {avg_processing_time:.3f}s")
        print(f"🏃 Fastest video: {fastest_time:.3f}s")
        print(f"🐌 Slowest video: {slowest_time:.3f}s")
        print(f"🎯 Videos under 0.7s target: {target_achieved}/{successful_videos}")
    
    # Get performance report from pipeline
    print("\n📈 QUANTUM PIPELINE PERFORMANCE REPORT")
    print("-" * 40)
    
    performance_report = pipeline.get_performance_report()
    if 'statistics' in performance_report:
        stats = performance_report['statistics']
        print(f"📊 Total videos generated: {stats['total_videos']}")
        print(f"⚡ Average time: {stats['average_time']:.3f}s")
        print(f"🏆 Best time: {stats['best_time']:.3f}s")
        print(f"🎯 Times under target: {stats['times_under_target']}")
    
    if 'device_info' in performance_report:
        device_info = performance_report['device_info']
        print(f"💻 Device: {device_info['device']}")
        print(f"🖥️  GPU available: {device_info['gpu_available']}")
        if device_info['gpu_name']:
            print(f"🎮 GPU name: {device_info['gpu_name']}")
    
    # Show individual results
    print("\n📋 INDIVIDUAL VIDEO RESULTS")
    print("-" * 40)
    for r in results:
        status = "✅ SUCCESS" if r['result']['success'] else "❌ FAILED"
        time_str = f"{r['result']['processing_time']:.3f}s" if r['result']['success'] else "N/A"
        print(f"Video {r['video_number']}: {r['script_name']} - {status} ({time_str})")
    
    print("\n🎉 Quantum pipeline test completed!")
    return successful_videos > 0

async def main():
    """Main test function"""
    try:
        success = await test_quantum_pipeline_generation()
        if success:
            print("\n✅ Test completed successfully!")
            sys.exit(0)
        else:
            print("\n❌ Test failed!")
            sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test failed with exception: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())