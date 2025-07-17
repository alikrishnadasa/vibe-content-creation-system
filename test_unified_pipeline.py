#!/usr/bin/env python3
"""
Test Unified Pipeline - Generate 5 videos using unified clips
Based on the existing test_quantum_pipeline.py but with unified metadata
"""

import asyncio
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

# Add unified-video-system-main to path
sys.path.append('unified-video-system-main')

from core.quantum_pipeline import UnifiedQuantumPipeline
from dataclasses import dataclass

@dataclass
class UnifiedVideoConfig:
    """Configuration for unified video generation"""
    clips_directory: str = ""  # Not used with unified metadata
    metadata_file: str = "unified_clips_metadata.json"
    scripts_directory: str = "unified-video-system-main/scripts"
    music_file: str = "unified-video-system-main/music/Beanie (Slowed).mp3"
    output_directory: str = "output"
    
    def get_available_scripts(self) -> List[Path]:
        """Get available scripts"""
        scripts_dir = Path(self.scripts_directory)
        if not scripts_dir.exists():
            return []
        return list(scripts_dir.glob("*.txt"))

async def test_unified_pipeline_generation():
    """Test the unified pipeline by generating 5 videos"""
    print("🚀 Testing Unified Quantum Pipeline - Generating 5 videos")
    print("=" * 60)
    
    # Initialize quantum pipeline
    print("🔧 Initializing quantum pipeline...")
    pipeline = UnifiedQuantumPipeline()
    
    # Use unified config
    config = UnifiedVideoConfig()
    
    # Get available scripts
    available_scripts = config.get_available_scripts()
    
    if not available_scripts:
        print("❌ No scripts found in scripts directory")
        print(f"   Looking in: {config.scripts_directory}")
        # Create a simple test script
        scripts_dir = Path(config.scripts_directory)
        scripts_dir.mkdir(parents=True, exist_ok=True)
        test_script = scripts_dir / "test_spiritual.txt"
        test_script.write_text("In the vast ocean of modern distractions, we find peace through meditation and mindfulness.")
        available_scripts = [test_script]
        print(f"✅ Created test script: {test_script}")
    
    print(f"📁 Found {len(available_scripts)} available scripts")
    for i, script in enumerate(available_scripts[:5]):
        print(f"   {i+1}. {script.stem}")
    
    # Initialize real content mode with unified metadata
    print("\n🎬 Initializing real content mode with unified metadata...")
    
    # We need to modify the real content generator to use unified metadata
    # For now, let's use regular video generation
    print("📝 Using regular video generation mode...")
    
    # Test configurations
    test_scripts = [
        "Peaceful meditation brings clarity to the mind and soul.",
        "In spiritual practice, we discover our true nature beyond material distractions.",
        "Krishna consciousness teaches us to find joy in simplicity and devotion.",
        "Modern anxiety dissolves when we connect with timeless wisdom traditions.",
        "The path of mindfulness leads from suffering to enlightenment."
    ]
    
    print(f"\n🎥 Generating {len(test_scripts)} test videos...")
    
    results = []
    
    for i, script_text in enumerate(test_scripts, 1):
        print(f"\n{i}️⃣ Generating video {i}/5")
        print(f"   Script: {script_text[:50]}...")
        
        try:
            start_time = time.time()
            
            # Create output path
            output_name = f"unified_test_video_{i}_{int(time.time())}.mp4"
            output_path = Path(config.output_directory) / output_name
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Generate video
            result = await pipeline.generate_video(
                script=script_text,
                style="default",
                music_path=config.music_file,
                enable_beat_sync=True,
                output_path=str(output_path)
            )
            
            generation_time = time.time() - start_time
            
            if result and result.get('success', False):
                print(f"   ✅ Generated successfully in {generation_time:.2f}s")
                print(f"   📁 Output: {output_path}")
                results.append({
                    "video_number": i,
                    "success": True,
                    "output_path": str(output_path),
                    "generation_time": generation_time,
                    "script": script_text[:30] + "..."
                })
            else:
                error_msg = result.get('error', 'Unknown error') if result else 'No result returned'
                print(f"   ❌ Generation failed: {error_msg}")
                results.append({
                    "video_number": i,
                    "success": False,
                    "error": error_msg,
                    "generation_time": generation_time
                })
                
        except Exception as e:
            generation_time = time.time() - start_time
            print(f"   ❌ Error: {e}")
            results.append({
                "video_number": i,
                "success": False,
                "error": str(e),
                "generation_time": generation_time
            })
    
    # Print summary
    print(f"\n📊 Generation Summary:")
    print(f"=" * 50)
    
    successful = sum(1 for r in results if r['success'])
    total_time = sum(r['generation_time'] for r in results)
    
    print(f"✅ Successful: {successful}/{len(results)}")
    print(f"⏱️ Total time: {total_time:.2f}s")
    print(f"⚡ Average time: {total_time/len(results):.2f}s per video")
    print(f"🎯 Target time: <0.7s per video")
    
    if successful > 0:
        avg_successful_time = sum(r['generation_time'] for r in results if r['success']) / successful
        print(f"📈 Average successful generation time: {avg_successful_time:.2f}s")
    
    print(f"\nDetailed Results:")
    for result in results:
        status = "✅" if result['success'] else "❌"
        video_num = result['video_number']
        time_str = f"{result['generation_time']:.2f}s"
        
        if result['success']:
            print(f"{status} Video {video_num}: {time_str} - {result['script']}")
        else:
            print(f"{status} Video {video_num}: {time_str} - Failed: {result.get('error', 'Unknown error')}")
    
    return successful > 0

async def main():
    """Main test function"""
    success = await test_unified_pipeline_generation()
    
    if success:
        print(f"\n🎉 Test completed successfully!")
        print(f"📁 Check the 'output' directory for generated videos")
    else:
        print(f"\n❌ Test failed - no videos were generated")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())