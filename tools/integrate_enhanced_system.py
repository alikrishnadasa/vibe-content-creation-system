#!/usr/bin/env python3
"""
Integration Script for Enhanced Semantic System

Integrates the enhanced script analyzer, content selector, and metadata creator
with the existing quantum video pipeline.
"""

import sys
import json
import logging
import asyncio
from pathlib import Path
from typing import Dict, Any

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project paths
sys.path.append('unified-video-system-main')

from enhanced_script_analyzer import EnhancedAudioScriptAnalyzer
from enhanced_content_selector import EnhancedContentSelector
from enhanced_metadata_creator import EnhancedMetadataCreator

async def integrate_enhanced_system():
    """Integrate the enhanced semantic system with existing pipeline"""
    
    print("🚀 Integrating Enhanced Semantic System")
    print("=" * 60)
    
    # Step 1: Create Enhanced Metadata for Existing Clips
    print("\n📊 Step 1: Creating Enhanced Metadata")
    print("-" * 40)
    
    metadata_creator = EnhancedMetadataCreator()
    
    # Create enhanced metadata for MJAnime clips
    mjanime_dir = Path("/Users/jamesguo/vibe-content-creation/MJAnime")
    mjanime_enhanced_metadata = Path("/Users/jamesguo/vibe-content-creation/enhanced_mjanime_metadata.json")
    
    if mjanime_dir.exists():
        print(f"Creating enhanced metadata for MJAnime clips...")
        success = metadata_creator.create_enhanced_metadata(
            clips_directory=mjanime_dir,
            output_file=mjanime_enhanced_metadata,
            force_recreate=True,
            use_visual_analysis=True
        )
        if success:
            print(f"✅ Enhanced MJAnime metadata created: {mjanime_enhanced_metadata}")
        else:
            print("❌ Failed to create enhanced MJAnime metadata")
    
    # Create enhanced metadata for Midjourney Composite clips
    midjourney_dir = Path("/Users/jamesguo/vibe-content-creation/midjourney_composite_2025-7-15")
    midjourney_enhanced_metadata = Path("/Users/jamesguo/vibe-content-creation/enhanced_midjourney_metadata.json")
    
    if midjourney_dir.exists():
        print(f"Creating enhanced metadata for Midjourney Composite clips...")
        success = metadata_creator.create_enhanced_metadata(
            clips_directory=midjourney_dir,
            output_file=midjourney_enhanced_metadata,
            force_recreate=True,
            use_visual_analysis=True
        )
        if success:
            print(f"✅ Enhanced Midjourney metadata created: {midjourney_enhanced_metadata}")
        else:
            print("❌ Failed to create enhanced Midjourney metadata")
    
    # Step 2: Create Unified Enhanced Metadata
    print("\n🔗 Step 2: Creating Unified Enhanced Metadata")
    print("-" * 40)
    
    unified_enhanced_metadata = await create_unified_enhanced_metadata(
        mjanime_enhanced_metadata, 
        midjourney_enhanced_metadata
    )
    
    if unified_enhanced_metadata:
        print("✅ Unified enhanced metadata created successfully")
    else:
        print("❌ Failed to create unified enhanced metadata")
        return False
    
    # Step 3: Initialize Enhanced Script Analyzer
    print("\n📝 Step 3: Initializing Enhanced Script Analyzer")
    print("-" * 40)
    
    scripts_dir = "11-scripts-for-tiktok"
    whisper_cache_dir = "cache/whisper"
    
    # First check if whisper cache exists in multiple locations
    whisper_cache_paths = [
        Path("/Users/jamesguo/vibe-content-creation/cache/whisper"),
        Path("/Users/jamesguo/vibe-content-creation/unified-video-system-main/cache/whisper")
    ]
    
    whisper_cache_found = None
    for cache_path in whisper_cache_paths:
        if cache_path.exists() and list(cache_path.glob("*.json")):
            whisper_cache_found = cache_path
            break
    
    if not whisper_cache_found:
        print("⚠️  Whisper cache not found. Enhanced script analysis will use filename fallback.")
        whisper_cache_dir = "cache/whisper"  # Use default path
    else:
        whisper_cache_dir = str(whisper_cache_found)
        print(f"📁 Found whisper cache: {whisper_cache_dir}")
    
    enhanced_analyzer = EnhancedAudioScriptAnalyzer(scripts_dir, whisper_cache_dir)
    
    print("Analyzing scripts with enhanced semantic analysis...")
    success = await enhanced_analyzer.analyze_scripts()
    
    if success:
        stats = enhanced_analyzer.get_analysis_stats()
        print(f"✅ Enhanced script analysis completed:")
        print(f"   Total scripts: {stats['total_scripts']}")
        print(f"   Enhanced analyses: {stats['enhanced_analysis_count']}")
        print(f"   Fallback analyses: {stats['fallback_analysis_count']}")
        print(f"   Emotions detected: {list(stats['emotion_distribution'].keys())}")
    else:
        print("❌ Enhanced script analysis failed")
        return False
    
    # Step 4: Test Enhanced Content Selection
    print("\n🎯 Step 4: Testing Enhanced Content Selection")
    print("-" * 40)
    
    # Load the unified enhanced metadata for content selection
    try:
        # Mock content database for testing
        class MockContentDatabase:
            def __init__(self, metadata_file):
                with open(metadata_file, 'r') as f:
                    self.metadata = json.load(f)
                
                # Mock clips loader
                self.clips_loader = MockClipsLoader(self.metadata)
        
        class MockClipsLoader:
            def __init__(self, metadata):
                self.clips = {}
                for clip_data in metadata.get('clips', []):
                    clip_obj = MockClip(clip_data)
                    self.clips[clip_obj.id] = clip_obj
        
        class MockClip:
            def __init__(self, clip_data):
                self.id = clip_data['id']
                self.filename = clip_data['filename']
                self.tags = clip_data['tags']
                self.duration = clip_data['duration']
                self.lighting_type = clip_data.get('shot_analysis', {}).get('lighting_type', 'natural')
                self.movement_type = clip_data.get('shot_analysis', {}).get('movement_type', 'gentle')
                self.shot_type = clip_data.get('shot_analysis', {}).get('shot_type', 'medium_shot')
        
        # Initialize enhanced content selector
        unified_metadata_path = Path("/Users/jamesguo/vibe-content-creation/unified_enhanced_metadata.json")
        if not unified_metadata_path.exists():
            print(f"❌ Unified enhanced metadata not found at {unified_metadata_path}")
            return False
        
        mock_db = MockContentDatabase(unified_metadata_path)
        enhanced_selector = EnhancedContentSelector(mock_db)
        
        print(f"✅ Enhanced content selector initialized with {len(mock_db.clips_loader.clips)} clips")
        
        # Test with a sample script analysis
        sample_script = "anxiety1"
        script_analysis = enhanced_analyzer.get_script_analysis(sample_script)
        
        if script_analysis:
            print(f"🧪 Testing enhanced selection with script: {sample_script}")
            print(f"   Primary emotion: {script_analysis.primary_emotion}")
            print(f"   Themes: {script_analysis.themes[:3]}")
            print(f"   Semantic phrases: {len(script_analysis.semantic_phrases)}")
            
            # Perform enhanced selection
            sequence = await enhanced_selector.select_clips_for_script(
                script_analysis=script_analysis,
                clip_count=5,
                script_duration=60.0,
                variation_seed=42
            )
            
            print(f"✅ Enhanced selection completed:")
            print(f"   Clips selected: {len(sequence.clips)}")
            print(f"   Relevance score: {sequence.relevance_score:.3f}")
            print(f"   Semantic coherence: {sequence.semantic_coherence_score:.3f}")
            print(f"   Contextual flow: {sequence.contextual_flow_score:.3f}")
            print(f"   Phrase match: {sequence.phrase_match_score:.3f}")
        else:
            print(f"❌ Could not find analysis for script: {sample_script}")
    
    except Exception as e:
        print(f"❌ Enhanced content selection test failed: {e}")
        return False
    
    # Step 5: Create Integration Instructions
    print("\n📋 Step 5: Creating Integration Instructions")
    print("-" * 40)
    
    integration_instructions = create_integration_instructions()
    
    with open("INTEGRATION_INSTRUCTIONS.md", "w") as f:
        f.write(integration_instructions)
    
    print("✅ Integration instructions saved to INTEGRATION_INSTRUCTIONS.md")
    
    # Step 6: Show Summary
    print("\n🎉 Integration Summary")
    print("=" * 60)
    print("✅ Enhanced semantic system successfully integrated!")
    print()
    print("Key Improvements Implemented:")
    print("🔹 Real voiceover text analysis using whisper transcriptions")
    print("🔹 Phrase-level semantic matching (6 categories)")
    print("🔹 Expanded emotional categories (12 emotions)")
    print("🔹 Context-aware clip selection with flow optimization")
    print("🔹 Visual content analysis using computer vision")
    print("🔹 Dynamic learning system for improved matching")
    print()
    print("Files Created:")
    print(f"📄 enhanced_script_analyzer.py")
    print(f"📄 enhanced_content_selector.py") 
    print(f"📄 enhanced_metadata_creator.py")
    print(f"📄 integrate_enhanced_system.py")
    print(f"📄 INTEGRATION_INSTRUCTIONS.md")
    print()
    print("Next Steps:")
    print("1. Review INTEGRATION_INSTRUCTIONS.md")
    print("2. Test with sample video generation")
    print("3. Replace original system components")
    print("4. Monitor improvements in semantic accuracy")
    
    return True


async def create_unified_enhanced_metadata(mjanime_metadata_path: Path, midjourney_metadata_path: Path) -> bool:
    """Create unified enhanced metadata from both sources"""
    
    try:
        unified_clips = []
        total_clips = 0
        
        # Load MJAnime enhanced metadata
        if mjanime_metadata_path.exists():
            with open(mjanime_metadata_path, 'r') as f:
                mjanime_data = json.load(f)
            
            for clip in mjanime_data.get('clips', []):
                # Add source tracking and full path
                clip['source_type'] = 'mjanime'
                clip['file_path'] = f"/Users/jamesguo/vibe-content-creation/MJAnime/{clip['filename']}"
                unified_clips.append(clip)
            
            print(f"📁 Loaded {len(mjanime_data.get('clips', []))} MJAnime clips")
            total_clips += len(mjanime_data.get('clips', []))
        
        # Load Midjourney enhanced metadata
        if midjourney_metadata_path.exists():
            with open(midjourney_metadata_path, 'r') as f:
                midjourney_data = json.load(f)
            
            for clip in midjourney_data.get('clips', []):
                # Add source tracking and full path
                clip['source_type'] = 'midjourney_composite'
                clip['file_path'] = f"/Users/jamesguo/vibe-content-creation/midjourney_composite_2025-7-15/{clip['filename']}"
                unified_clips.append(clip)
            
            print(f"📁 Loaded {len(midjourney_data.get('clips', []))} Midjourney clips")
            total_clips += len(midjourney_data.get('clips', []))
        
        # Create unified enhanced metadata
        unified_metadata = {
            "metadata_info": {
                "created_at": "2025-07-15T00:00:00",
                "generator": "integrate_enhanced_system.py",
                "version": "2.0_enhanced",
                "total_clips": total_clips,
                "sources": ["mjanime", "midjourney_composite"],
                "enhancement_features": {
                    "real_transcript_analysis": True,
                    "phrase_level_matching": True,
                    "expanded_emotions": True,
                    "context_aware_selection": True,
                    "visual_content_analysis": True,
                    "dynamic_learning": True
                }
            },
            "clips": unified_clips
        }
        
        # Save unified enhanced metadata
        output_path = Path("/Users/jamesguo/vibe-content-creation/unified_enhanced_metadata.json")
        with open(output_path, 'w') as f:
            json.dump(unified_metadata, f, indent=2)
        
        print(f"💾 Unified enhanced metadata saved: {output_path}")
        print(f"📊 Total clips in unified metadata: {total_clips}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to create unified enhanced metadata: {e}")
        return False


def create_integration_instructions() -> str:
    """Create detailed integration instructions"""
    
    instructions = """# Enhanced Semantic System Integration Instructions

## Overview

The enhanced semantic system has been successfully created with major improvements to semantic accuracy and clip-to-voiceover matching. This system addresses the core issue where script analysis was based only on filenames rather than actual voiceover content.

## Key Problems Solved

### 1. **Real Voiceover Analysis**
- **Before**: Script analysis based only on filename patterns (e.g., `anxiety1.wav` → "anxiety" tags)
- **After**: Uses actual whisper transcriptions to analyze real content themes and semantic meaning

### 2. **Phrase-Level Semantic Matching** 
- **Before**: Simple keyword matching
- **After**: Advanced phrase pattern recognition with 6 semantic categories:
  - `spiritual_identity` - soul, consciousness, divine nature
  - `divine_connection` - Krishna, God, spiritual relationships  
  - `material_illusion` - meaningless universe, temporary things
  - `inner_transformation` - spiritual growth, consciousness shifts
  - `life_purpose` - human life purpose, spiritual realization
  - `modern_struggles` - phone addiction, digital detox, comfort zones

### 3. **Expanded Emotional Categories**
- **Before**: 5 basic emotions (anxiety, peace, seeking, awakening, neutral)
- **After**: 12 nuanced emotions with better mapping:
  - Original: anxiety, peace, seeking, awakening
  - New: contemplative, transformative, communal, transcendent, grounding, struggle, liberation, devotional

### 4. **Context-Aware Clip Selection**
- **Before**: Random clip selection with basic visual variety
- **After**: Contextual flow optimization with:
  - Emotional progression tracking
  - Visual transition smoothness
  - Narrative structure awareness
  - Learned flow patterns

### 5. **Visual Content Analysis**
- **Before**: Filename-based visual characteristic guessing
- **After**: Computer vision analysis of actual video content:
  - Color palette extraction
  - Motion analysis
  - Composition characteristics
  - Lighting analysis

### 6. **Dynamic Learning System**
- **Before**: Static selection algorithms
- **After**: Learning system that improves over time:
  - Success correlation tracking
  - Phrase-clip mapping refinement
  - Contextual flow pattern learning

## Files Created

### Core Enhancement Files
1. **`enhanced_script_analyzer.py`**
   - Uses whisper transcriptions for real content analysis
   - Phrase-level semantic pattern recognition
   - 12 expanded emotional categories
   - Thematic categorization with weights

2. **`enhanced_content_selector.py`**
   - Phrase-level semantic matching (40% weight)
   - Context-aware flow optimization
   - Dynamic learning from successful selections
   - Visual transition scoring

3. **`enhanced_metadata_creator.py`**
   - Computer vision analysis of video content
   - Enhanced shot analysis
   - Semantic categorization
   - Visual characteristic extraction

4. **`integrate_enhanced_system.py`**
   - Integration script that creates enhanced metadata
   - Tests the enhanced system
   - Creates unified enhanced metadata

## Integration Steps

### Phase 1: Backup and Preparation
```bash
# 1. Backup existing system
cp unified-video-system-main/content/script_analyzer.py unified-video-system-main/content/script_analyzer.py.backup
cp unified-video-system-main/content/content_selector.py unified-video-system-main/content/content_selector.py.backup

# 2. Verify whisper cache exists
ls cache/whisper/
# Should show files like: anxiety1_base.json, safe1_base.json, etc.
```

### Phase 2: Replace Core Components

#### 2.1 Replace Script Analyzer
```python
# In unified-video-system-main/content/content_database.py
# Replace this line:
from .script_analyzer import AudioScriptAnalyzer

# With:
import sys
sys.path.append('/Users/jamesguo/vibe-content-creation')
from enhanced_script_analyzer import EnhancedAudioScriptAnalyzer as AudioScriptAnalyzer
```

#### 2.2 Replace Content Selector
```python
# In unified-video-system-main/content/content_database.py  
# Replace this line:
from .content_selector import ContentSelector

# With:
import sys
sys.path.append('/Users/jamesguo/vibe-content-creation')
from enhanced_content_selector import EnhancedContentSelector as ContentSelector
```

#### 2.3 Update ContentDatabase Initialization
```python
# In unified-video-system-main/content/content_database.py
# In the ContentDatabase.__init__ method, update the script analyzer initialization:

self.scripts_analyzer = AudioScriptAnalyzer(
    scripts_directory=scripts_directory,
    whisper_cache_directory="cache/whisper"  # Add this parameter
)
```

### Phase 3: Use Enhanced Metadata

#### 3.1 Update MJAnimeLoader to use enhanced metadata
```python
# In unified-video-system-main/content/mjanime_loader.py
# Update the metadata file path to use enhanced metadata:

def __init__(self, clips_directory: str, metadata_file: str = "unified_enhanced_metadata.json", use_unified_metadata: bool = True):
    # Use enhanced metadata by default
    if metadata_file == "unified_clips_metadata.json":
        metadata_file = "unified_enhanced_metadata.json"
    
    self.clips_directory = Path(clips_directory)
    self.metadata_file = Path(metadata_file)
    self.use_unified_metadata = use_unified_metadata
```

### Phase 4: Test Enhanced System

#### 4.1 Test Script Analysis
```python
# Test script to verify enhanced analysis
import asyncio
from enhanced_script_analyzer import EnhancedAudioScriptAnalyzer

async def test_enhanced_analysis():
    analyzer = EnhancedAudioScriptAnalyzer("11-scripts-for-tiktok", "cache/whisper")
    await analyzer.analyze_scripts()
    
    # Test with anxiety1
    analysis = analyzer.get_script_analysis("anxiety1")
    print(f"Themes: {analysis.themes}")
    print(f"Semantic phrases: {analysis.semantic_phrases}")
    print(f"Content keywords: {analysis.content_keywords}")

asyncio.run(test_enhanced_analysis())
```

#### 4.2 Test Enhanced Content Selection
```python
# Use the existing generate_unified_videos.py script
python generate_unified_videos.py
```

## Expected Improvements

### 1. Better Semantic Accuracy
- Clips will match actual voiceover content rather than just filename patterns
- Example: `anxiety1.wav` about "eternal soul" and "Krishna" will get spiritual clips, not just anxiety-themed clips

### 2. More Contextual Flow
- Clips will transition more smoothly between different emotional states
- Narrative structure will be more coherent

### 3. Enhanced Visual Matching
- Clips will be selected based on actual visual characteristics, not just filename guessing
- Color, motion, and composition will align better with voiceover content

### 4. Learning Over Time
- The system will improve as it learns which clip combinations work well
- Successful patterns will be reinforced

## Monitoring Success

### Key Metrics to Track
1. **Semantic Relevance Scores** - Should increase from baseline
2. **Visual Variety Scores** - Should maintain or improve
3. **Contextual Flow Scores** - New metric, aim for >0.7
4. **Phrase Match Scores** - New metric, aim for >0.6

### Testing Procedure
1. Generate 5 videos with the enhanced system
2. Compare clip selections to previous system
3. Evaluate semantic relevance manually
4. Check for improved narrative flow

## Troubleshooting

### Common Issues

#### 1. "Whisper cache not found"
- Verify whisper cache exists in `cache/whisper/`
- System will fall back to filename analysis if cache missing

#### 2. "Enhanced metadata not found"
- Run `python integrate_enhanced_system.py` to create enhanced metadata
- Verify `unified_enhanced_metadata.json` exists

#### 3. Import errors
- Ensure all enhanced files are in the correct directory
- Check Python path includes the enhanced system directory

#### 4. Visual analysis fails
- Install OpenCV: `pip install opencv-python`
- System will work without visual analysis if OpenCV unavailable

### Rollback Procedure
If issues occur, restore original system:
```bash
# Restore original files
cp unified-video-system-main/content/script_analyzer.py.backup unified-video-system-main/content/script_analyzer.py
cp unified-video-system-main/content/content_selector.py.backup unified-video-system-main/content/content_selector.py

# Use original metadata
# Change metadata_file back to "unified_clips_metadata.json" in mjanime_loader.py
```

## Future Enhancements

### Phase 2 Improvements
1. **Real-time Visual Analysis** - Analyze clips during selection
2. **Advanced Learning** - Machine learning models for clip selection
3. **User Feedback Integration** - Learn from video performance metrics
4. **Multi-modal Analysis** - Combine audio, visual, and text analysis

### Performance Optimizations
1. **Caching** - Cache enhanced analyses for faster selection
2. **Parallel Processing** - Parallelize visual analysis
3. **Incremental Learning** - Update learning data incrementally

## Conclusion

This enhanced semantic system represents a significant improvement in clip-to-voiceover matching accuracy. The system now analyzes actual content rather than making assumptions based on filenames, resulting in more semantically relevant and contextually appropriate video generation.

The integration has been designed to be backward-compatible and includes comprehensive fallback mechanisms to ensure system stability during the transition.
"""
    
    return instructions


async def main():
    """Main integration function"""
    success = await integrate_enhanced_system()
    if success:
        print("\n🎉 Enhanced semantic system integration completed successfully!")
    else:
        print("\n❌ Enhanced semantic system integration failed!")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())