#!/usr/bin/env python3
"""
Generate 5 Test Videos
Uses the unified quantum pipeline to create test videos with clips from both sources
"""

import asyncio
import sys
import time
from pathlib import Path
from datetime import datetime

# Add unified-video-system-main to path
sys.path.append('unified-video-system-main')

from core.quantum_pipeline import UnifiedQuantumPipeline
from content.content_database import ContentDatabase

async def generate_test_videos():
    """Generate 5 test videos with different configurations"""
    
    print("🚀 Initializing Unified Quantum Pipeline for test video generation...")
    
    # Initialize the quantum pipeline
    pipeline = UnifiedQuantumPipeline()
    
    # Initialize real content mode with unified metadata
    print("⚙️ Initializing real content mode...")
    success = await pipeline.initialize_real_content_mode(
        clips_directory="",  # Not used with unified loader
        metadata_file="unified_clips_metadata.json",
        scripts_directory="unified-video-system-main/scripts",
        music_file="unified-video-system-main/music/Beanie (Slowed).mp3"
    )
    
    if not success:
        print("❌ Failed to initialize real content mode")
        return False
    
    print("✅ Real content mode initialized successfully")
    
    # Access the content database from the pipeline
    content_db = pipeline.real_content_generator.content_db
    
    # Show available clips
    source_stats = content_db.clips_loader.get_source_stats()
    print(f"📊 Available clips: {source_stats}")
    
    # Define 5 different test configurations
    test_configs = [
        {
            "name": "peaceful_mixed",
            "description": "Peaceful clips from both sources",
            "emotion_filter": "peace",
            "clips_count": 8,
            "prefer_source": None
        },
        {
            "name": "mjanime_only", 
            "description": "MJAnime spiritual content only",
            "emotion_filter": None,
            "clips_count": 6,
            "prefer_source": "mjanime"
        },
        {
            "name": "composite_only",
            "description": "Midjourney composite content only", 
            "emotion_filter": None,
            "clips_count": 6,
            "prefer_source": "midjourney_composite"
        },
        {
            "name": "anxiety_themes",
            "description": "Anxiety-themed clips from both sources",
            "emotion_filter": "anxiety",
            "clips_count": 5,
            "prefer_source": None
        },
        {
            "name": "mixed_random",
            "description": "Random mix from all sources",
            "emotion_filter": None,
            "clips_count": 10,
            "prefer_source": None
        }
    ]
    
    print(f"\n🎬 Generating {len(test_configs)} test videos...")
    
    results = []
    
    for i, config in enumerate(test_configs, 1):
        print(f"\n{i}️⃣ Generating: {config['name']}")
        print(f"   Description: {config['description']}")
        
        try:
            # Get clips based on configuration
            if config['prefer_source']:
                available_clips = content_db.clips_loader.get_clips_by_source(config['prefer_source'])
                print(f"   Using source: {config['prefer_source']} ({len(available_clips)} clips)")
            elif config['emotion_filter']:
                available_clips = content_db.clips_loader.get_clips_by_emotion(config['emotion_filter'])
                print(f"   Using emotion: {config['emotion_filter']} ({len(available_clips)} clips)")
            else:
                available_clips = list(content_db.clips_loader.clips.values())
                print(f"   Using all clips ({len(available_clips)} clips)")
            
            if len(available_clips) < config['clips_count']:
                print(f"   ⚠️ Only {len(available_clips)} clips available, using all")
                selected_clips = available_clips
            else:
                # Select clips for video
                selected_clips = content_db.clips_loader.get_random_clips(
                    count=config['clips_count'],
                    emotion=config['emotion_filter'] if not config['prefer_source'] else None
                )
            
            # Show clip sources breakdown
            source_breakdown = {}
            for clip in selected_clips:
                source = clip.source_type
                source_breakdown[source] = source_breakdown.get(source, 0) + 1
            
            print(f"   Selected clips: {source_breakdown}")
            
            # Generate video configuration
            video_config = {
                "content_name": f"test_{config['name']}",
                "script_name": "default",
                "mixed_audio": True,
                "total_variations": 1,
                "selected_clips": selected_clips,
                "output_name": f"test_video_{i}_{config['name']}_{int(time.time())}"
            }
            
            start_time = time.time()
            
            # Generate the video using the quantum pipeline
            print(f"   🎥 Generating video...")
            
            # Create output name
            output_name = f"test_video_{i}_{config['name']}_{int(time.time())}.mp4"
            
            # Use the pipeline's regular video generation with the spiritual script
            result = await pipeline.generate_video(
                script=f"Test video {i}: {config['description']}. Using spiritual content for mindful living.",
                style="default",
                music_path="unified-video-system-main/music/Beanie (Slowed).mp3",
                enable_beat_sync=True,
                output_path=f"output/{output_name}"
            )
            
            generation_time = time.time() - start_time
            
            if result:
                print(f"   ✅ Generated successfully in {generation_time:.2f}s")
                print(f"   📁 Output: {result}")
                results.append({
                    "config": config,
                    "success": True,
                    "output": result,
                    "time": generation_time,
                    "clips_used": len(selected_clips),
                    "source_breakdown": source_breakdown
                })
            else:
                print(f"   ❌ Generation failed")
                results.append({
                    "config": config,
                    "success": False,
                    "error": "Generation failed",
                    "time": generation_time
                })
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
            results.append({
                "config": config,
                "success": False,
                "error": str(e),
                "time": 0
            })
    
    # Print summary
    print(f"\n📊 Generation Summary:")
    print(f"=" * 50)
    
    successful = sum(1 for r in results if r['success'])
    total_time = sum(r['time'] for r in results)
    
    print(f"✅ Successful: {successful}/{len(results)}")
    print(f"⏱️ Total time: {total_time:.2f}s")
    print(f"⚡ Average time: {total_time/len(results):.2f}s per video")
    
    for i, result in enumerate(results, 1):
        status = "✅" if result['success'] else "❌"
        config_name = result['config']['name']
        
        if result['success']:
            clips_info = f"({result['clips_used']} clips)"
            source_info = f"Sources: {result['source_breakdown']}"
            print(f"{status} Video {i}: {config_name} {clips_info} - {result['time']:.2f}s - {source_info}")
        else:
            print(f"{status} Video {i}: {config_name} - Failed: {result.get('error', 'Unknown error')}")
    
    return successful > 0

async def main():
    """Main function"""
    success = await generate_test_videos()
    if not success:
        print("\n❌ No videos were generated successfully")
        sys.exit(1)
    else:
        print(f"\n🎉 Test video generation completed!")

if __name__ == "__main__":
    asyncio.run(main())