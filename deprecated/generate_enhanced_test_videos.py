#!/usr/bin/env python3
"""
Generate 5 test videos using the enhanced semantic system
"""

import sys
import asyncio
import logging
from pathlib import Path

# Add paths
sys.path.append('unified-video-system-main')
sys.path.append('/Users/jamesguo/vibe-content-creation')

from core.real_content_generator import RealContentGenerator

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def generate_enhanced_test_videos():
    """Generate 5 test videos with the enhanced semantic system"""
    
    print("🎬 Generating 5 Test Videos with Enhanced Semantic System")
    print("=" * 70)
    
    # Initialize the real content generator with enhanced metadata
    try:
        generator = RealContentGenerator(
            clips_directory="/Users/jamesguo/vibe-content-creation/midjourney_composite_2025-7-15",
            metadata_file="unified_enhanced_metadata.json",
            scripts_directory="11-scripts-for-tiktok", 
            music_file="11-scripts-for-tiktok/anxiety1.wav",
            output_directory="enhanced_test_output"
        )
        
        # Initialize the generator
        await generator.initialize()
        print("✅ Enhanced content generator initialized")
        
    except Exception as e:
        print(f"❌ Failed to initialize generator: {e}")
        return False
    
    # Test scripts to generate videos for
    test_scripts = [
        "anxiety1",     # Spiritual/material illusion themes
        "safe1",        # Peace/grounding themes  
        "phone1",       # Modern struggles/digital detox
        "deadinside",   # Existential/transformation themes
        "before"        # Life purpose/awakening themes
    ]
    
    results = []
    
    for i, script_name in enumerate(test_scripts, 1):
        print(f"\n🎯 Generating Video {i}/5: {script_name}")
        print("-" * 50)
        
        try:
            # Create request for video generation
            from core.real_content_generator import RealVideoRequest
            
            request = RealVideoRequest(
                script_path=f"11-scripts-for-tiktok/{script_name}.wav",
                script_name=script_name,
                variation_number=1,
                target_duration=60.0,
                output_path=f"enhanced_test_output/{script_name}_enhanced_v1.mp4",
                burn_in_captions=True  # Burn captions directly into video for visibility
            )
            
            # Generate video with enhanced system
            result = await generator.generate_video(request)
            
            if result.success:
                print(f"✅ Video {i} generated successfully!")
                print(f"   Output: {result.output_path}")
                print(f"   Duration: {result.total_duration:.2f}s")
                print(f"   Total clips: {len(result.clips_used)}")
                print(f"   Processing time: {result.generation_time:.2f}s")
                print(f"   Relevance score: {result.relevance_score:.3f}")
                print(f"   Visual variety: {result.visual_variety_score:.3f}")
                
                results.append({
                    'script': script_name,
                    'success': True,
                    'output_path': result.output_path,
                    'duration': result.total_duration,
                    'clips_used': len(result.clips_used)
                })
            else:
                print(f"❌ Video {i} generation failed: {result.error_message}")
                results.append({
                    'script': script_name,
                    'success': False,
                    'error': result.error_message
                })
                
        except Exception as e:
            print(f"❌ Video {i} generation failed with exception: {e}")
            results.append({
                'script': script_name,
                'success': False,
                'error': str(e)
            })
    
    # Summary
    print(f"\n📊 Generation Summary")
    print("=" * 70)
    
    successful = sum(1 for r in results if r['success'])
    print(f"✅ Successful videos: {successful}/5")
    print(f"❌ Failed videos: {5 - successful}/5")
    
    if successful > 0:
        print(f"\n🎬 Generated Videos:")
        for result in results:
            if result['success']:
                print(f"   • {result['script']}: {result['output_path']}")
                print(f"     Duration: {result['duration']:.2f}s, Clips: {result['clips_used']}")
    
    if successful < 5:
        print(f"\n❌ Failed Videos:")
        for result in results:
            if not result['success']:
                print(f"   • {result['script']}: {result['error']}")
    
    print(f"\n🎉 Enhanced semantic system test complete!")
    print(f"Generated {successful} videos with improved clip-to-voiceover matching.")
    
    return successful > 0

async def main():
    """Main function"""
    success = await generate_enhanced_test_videos()
    if not success:
        print("❌ No videos were generated successfully")
        sys.exit(1)
    else:
        print("✅ Enhanced video generation test completed!")

if __name__ == "__main__":
    asyncio.run(main())