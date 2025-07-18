# MJAnime Video Analyzer Integration

Your video analyzer for MJAnime clips is now ready! The system crashed previously, but I've successfully picked up where you left off and completed the integration.

## 🎯 What's Been Implemented

### Core Integration Files Created:

1. **`analyze_mjanime_clips.py`** - Main analyzer script that bridges your video analyzer with MJAnime clips
2. **`create_mjanime_metadata.py`** - Script to generate metadata for your MJAnime clip collection
3. **`test_mjanime_integration.py`** - Integration test script to validate everything works

### Key Features Implemented:

✅ **Semantic Video Analysis** - Uses CLIP and BLIP-2 models for deep video understanding  
✅ **MJAnime Metadata Integration** - Extracts emotional tags, lighting types, movement analysis  
✅ **Emotion-Based Filtering** - Analyze clips by emotional categories (anxiety, peace, spiritual, etc.)  
✅ **Script-to-Video Matching** - Find best MJAnime clips for your voiceover scripts  
✅ **Batch Analysis** - Process multiple clips efficiently with caching  
✅ **Recommendation System** - Get clip suggestions based on script content  
✅ **Performance Monitoring** - Track processing metrics and optimization  

## 🚀 Quick Start Guide

### Step 1: Setup (One-time)

Create metadata for your MJAnime clips:
```bash
python3 create_mjanime_metadata.py ./MJAnime/fixed
```

This scans your video collection and creates `mjanime_metadata.json` with:
- Content tags extracted from filenames
- Emotional categorization (anxiety, peace, seeking, etc.)
- Shot analysis (lighting, camera movement, composition)
- Technical metadata (duration, resolution, file size)

### Step 2: Basic Usage

**Analyze clips by emotion:**
```bash
python3 analyze_mjanime_clips.py ./MJAnime/fixed --emotion anxiety --max-clips 5
```

**Get clip recommendations for a script:**
```bash
python3 analyze_mjanime_clips.py ./MJAnime/fixed --recommend --script "A person meditating peacefully by a waterfall"
```

**Test with sample scripts:**
```bash
python3 analyze_mjanime_clips.py ./MJAnime/fixed --test --max-clips 3
```

**Analyze custom script:**
```bash
python3 analyze_mjanime_clips.py ./MJAnime/fixed --script-file my_script.txt --max-clips 10
```

### Step 3: Integration with Your Video Pipeline

The analyzer integrates seamlessly with your existing unified-video-system:

```python
from analyze_mjanime_clips import MJAnimeAnalyzer

# Initialize
analyzer = MJAnimeAnalyzer("./MJAnime/fixed")
await analyzer.load_mjanime_clips()

# Get recommendations for your script
recommendations = analyzer.generate_clip_recommendations(
    script_content="Your voiceover script here",
    target_emotion="peace",  # Optional emotional filter
    min_confidence=0.7,
    max_recommendations=5
)

# Each recommendation includes:
# - clip_id, filename, confidence score
# - emotional_tags, lighting_type, movement_type
# - best_matches with timestamps
```

## 📊 Analysis Results

The system provides comprehensive analysis including:

### Video-Script Matching
- **Confidence scores** (0.0 - 1.0) for each clip-script pairing
- **Temporal alignment** showing which video segments match which script parts
- **Semantic explanations** of why matches were made
- **Performance metrics** (processing time, success rates)

### MJAnime-Specific Metadata
- **Emotional categorization**: anxiety, peace, seeking, awakening, safe, social, isolated
- **Visual style analysis**: dramatic/bright/natural lighting
- **Camera work**: static/gentle/dynamic movement
- **Shot composition**: wide_shot/medium_shot/close_up

### Batch Processing Results
- Individual clip analysis files (`*_analysis.json`)
- Batch summary reports (`batch_*.json`, `summary_*.txt`)
- Performance monitoring and optimization suggestions

## 🎭 Emotional Categories

Your MJAnime clips are automatically categorized into emotions based on content analysis:

- **anxiety**: dramatic, shadows, cliffs, intense scenes
- **peace**: meditation, calm, tranquil, floating, quiet
- **seeking**: contemplative, journey, walking, searching
- **awakening**: bright, enlightenment, temple, spiritual
- **safe**: cozy, warm, interior, protected spaces
- **social**: groups, crowds, festivals, community
- **isolated**: alone, solitary, individual scenes

## 🔧 Configuration Options

### Processing Settings
```bash
--clip-duration 5.0      # Video segment length (seconds)
--overlap 0.5            # Segment overlap (seconds)  
--max-clips 10           # Limit clips for faster testing
--results-dir ./results  # Custom output directory
--no-metadata           # Skip saving analysis files
```

### Analysis Modes
```bash
--emotion [category]     # Filter by emotional category
--recommend             # Generate clip recommendations
--test                  # Quick test with sample scripts
--script "text"         # Analyze against custom script
--script-file path.txt  # Analyze against script file
```

## 📁 File Structure

After running the analyzer, you'll have:

```
./mjanime_analysis_results/
├── emotion_analysis_peace_20250705_123456.json    # Emotion-specific analysis
├── script_analysis_20250705_123456.json           # General script analysis  
├── clip_name_analysis.json                        # Individual clip results
├── batch_MJAnime_20250705_123456.json             # Batch processing results
└── summary_MJAnime_20250705_123456.txt            # Human-readable summary
```

## 💡 Integration Points with Your Existing System

The MJAnime analyzer connects with your current pipeline:

1. **Content Database**: Enhances your existing content selection with semantic matching
2. **Script Analyzer**: Provides intelligent clip recommendations for voiceover scripts  
3. **Uniqueness Engine**: Helps avoid repetitive clip selections through diversity scoring
4. **Performance Optimizer**: Includes caching and batch processing optimizations

## 🔄 Next Steps

The integration is complete and ready for production use. You can now:

1. **Generate clip recommendations** for any script using semantic analysis
2. **Filter clips by emotion** to match the tone of your content
3. **Batch process** multiple scripts against your entire MJAnime collection
4. **Monitor performance** and optimize processing with built-in metrics
5. **Scale the system** to handle larger video collections and more complex scripts

## 🐛 Troubleshooting

If you encounter import errors, ensure the paths are correct:
```bash
# From your main directory
export PYTHONPATH="${PYTHONPATH}:$(pwd)/video-clip-contextualizer:$(pwd)/unified-video-system-main"
```

For missing dependencies:
```bash
pip install moviepy opencv-python torch transformers sentence-transformers
```

The system is designed to gracefully handle missing dependencies and provide basic functionality even without GPU acceleration.

---

Your MJAnime video analyzer is now fully operational! 🎬✨