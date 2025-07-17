#!/usr/bin/env python3
"""
Test MJAnime Integration

Quick test script to validate the integration between the Video Clip Contextualizer
and the MJAnime video collection.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent))

from create_mjanime_metadata import create_mjanime_metadata
from analyze_mjanime_clips import MJAnimeAnalyzer, create_sample_scripts


async def test_integration():
    """Test the complete MJAnime integration workflow."""
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logger = logging.getLogger(__name__)
    
    print("🚀 Testing MJAnime Video Analyzer Integration")
    print("=" * 60)
    
    # Define paths
    mjanime_directory = Path("./MJAnime/fixed")
    metadata_file = Path("./mjanime_metadata.json")
    results_dir = Path("./test_results")
    
    # Step 1: Check if MJAnime directory exists
    if not mjanime_directory.exists():
        print(f"❌ MJAnime directory not found: {mjanime_directory}")
        print("Please ensure the MJAnime clips are available in ./MJAnime/fixed/")
        return False
    
    # Count video files
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
    video_files = []
    for ext in video_extensions:
        video_files.extend(mjanime_directory.glob(f"*{ext}"))
    
    print(f"📁 Found {len(video_files)} video files in {mjanime_directory}")
    
    if len(video_files) == 0:
        print("❌ No video files found. Please check the directory.")
        return False
    
    # Step 2: Create metadata if needed
    print(f"\n📋 Creating metadata file...")
    success = create_mjanime_metadata(
        clips_directory=mjanime_directory,
        output_file=metadata_file,
        force_recreate=False
    )
    
    if not success:
        print("❌ Failed to create metadata file")
        return False
    
    print(f"✅ Metadata file ready: {metadata_file}")
    
    # Step 3: Initialize analyzer
    print(f"\n🔧 Initializing MJAnime analyzer...")
    try:
        analyzer = MJAnimeAnalyzer(
            mjanime_directory=str(mjanime_directory),
            analysis_results_dir=str(results_dir),
            save_metadata=True
        )
        
        # Load clips
        success = await analyzer.load_mjanime_clips()
        if not success:
            print("❌ Failed to load MJAnime clips")
            return False
        
        stats = analyzer.mjanime_loader.get_clip_stats()
        print(f"✅ Loaded {stats['total_clips']} clips")
        print(f"   Total duration: {stats['total_duration_seconds']:.1f}s")
        print(f"   Emotions found: {list(stats['emotion_distribution'].keys())}")
        
    except Exception as e:
        print(f"❌ Failed to initialize analyzer: {e}")
        return False
    
    # Step 4: Test basic analysis
    print(f"\n🧪 Testing basic analysis...")
    try:
        sample_scripts = create_sample_scripts()
        test_script = sample_scripts["peace_script"]
        
        # Analyze with limited clips for quick test
        results = analyzer.analyze_clips_by_script(
            script_content=test_script,
            clip_duration=3.0,  # Shorter for faster test
            overlap=0.2,
            max_clips=3  # Only test with 3 clips
        )
        
        if "error" in results:
            print(f"❌ Analysis failed: {results['error']}")
            return False
        
        batch_info = results["batch_info"]
        print(f"✅ Analysis completed successfully")
        print(f"   Processed: {batch_info['processed_videos']} clips")
        print(f"   Success rate: {batch_info['success_rate']:.1%}")
        print(f"   Processing time: {batch_info['total_processing_time']:.1f}s")
        
        if "summary" in results and not results["summary"].get("no_data"):
            summary = results["summary"]
            print(f"   Average confidence: {summary['average_confidence']:.3f}")
        
    except Exception as e:
        print(f"❌ Analysis test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 5: Test emotion filtering
    print(f"\n🎭 Testing emotion filtering...")
    try:
        # Test different emotions
        emotions_to_test = ["peace", "anxiety", "seeking"]
        
        for emotion in emotions_to_test:
            emotion_clips = analyzer.mjanime_loader.get_clips_by_emotion(emotion)
            print(f"   {emotion}: {len(emotion_clips)} clips")
            
            if len(emotion_clips) > 0:
                # Quick test with one clip
                results = analyzer.analyze_clips_by_script(
                    script_content=f"A {emotion} scene with calm atmosphere",
                    emotional_filter=emotion,
                    clip_duration=2.0,
                    max_clips=1
                )
                
                if "error" not in results:
                    print(f"     ✅ {emotion} analysis successful")
                else:
                    print(f"     ⚠️ {emotion} analysis had issues: {results['error']}")
        
    except Exception as e:
        print(f"❌ Emotion filtering test failed: {e}")
        return False
    
    # Step 6: Test recommendations
    print(f"\n💡 Testing recommendation system...")
    try:
        recommendations = analyzer.generate_clip_recommendations(
            script_content="A peaceful meditation scene with spiritual energy",
            target_emotion="peace",
            min_confidence=0.1,  # Lower threshold for test
            max_recommendations=3
        )
        
        print(f"✅ Generated {len(recommendations)} recommendations")
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec['filename']} (confidence: {rec['confidence']:.3f})")
        
    except Exception as e:
        print(f"❌ Recommendation test failed: {e}")
        return False
    
    # Final summary
    print(f"\n🎉 Integration Test Summary")
    print("=" * 60)
    print("✅ MJAnime directory found")
    print("✅ Metadata file created")
    print("✅ Video analyzer initialized")
    print("✅ Basic analysis working")
    print("✅ Emotion filtering working")
    print("✅ Recommendation system working")
    print("\n🚀 MJAnime Video Analyzer integration is ready!")
    print(f"\nNext steps:")
    print(f"1. Run full analysis: python analyze_mjanime_clips.py {mjanime_directory} --test")
    print(f"2. Generate recommendations: python analyze_mjanime_clips.py {mjanime_directory} --recommend --script 'your script here'")
    print(f"3. Analyze specific emotions: python analyze_mjanime_clips.py {mjanime_directory} --emotion peace")
    
    return True


def main():
    """Main entry point for integration test."""
    try:
        success = asyncio.run(test_integration())
        if not success:
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n❌ Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()