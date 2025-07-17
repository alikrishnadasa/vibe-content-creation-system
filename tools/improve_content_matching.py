#!/usr/bin/env python3
"""
Improvements for Content Matching and Caption Issues

1. Fix caption fallback to use available caption styles
2. Improve content matching with better semantic analysis
3. Add debug info for matching scores
"""

import sys
import asyncio
import logging
from pathlib import Path

# Add paths
sys.path.append('unified-video-system-main')
sys.path.append('/Users/jamesguo/vibe-content-creation')

from enhanced_script_analyzer import EnhancedAudioScriptAnalyzer
from enhanced_content_selector import EnhancedContentSelector

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def improve_content_matching():
    """Analyze and improve content matching system"""
    
    print("🔍 Analyzing Content Matching Issues")
    print("=" * 60)
    
    # 1. Caption Issue Analysis
    print("\n📝 Caption Issue Analysis")
    print("-" * 40)
    
    captions_dir = Path("unified-video-system-main/cache/pregenerated_captions")
    caption_files = list(captions_dir.glob("*.json"))
    
    # Analyze available caption styles
    style_analysis = {}
    for file in caption_files:
        parts = file.stem.split('_')
        if len(parts) >= 2:
            script_name = '_'.join(parts[:-1])
            style = parts[-1]
            
            if script_name not in style_analysis:
                style_analysis[script_name] = []
            style_analysis[script_name].append(style)
    
    print("Available caption styles by script:")
    for script, styles in style_analysis.items():
        print(f"  {script}: {styles}")
    
    # Check which scripts have tiktok vs default
    scripts_with_tiktok = [s for s, styles in style_analysis.items() if 'tiktok' in styles]
    scripts_with_default = [s for s, styles in style_analysis.items() if 'default' in styles]
    
    print(f"\n📊 Caption Analysis:")
    print(f"Scripts with TikTok captions: {len(scripts_with_tiktok)} ({scripts_with_tiktok})")
    print(f"Scripts with Default captions: {len(scripts_with_default)}")
    print(f"Missing TikTok captions: {set(scripts_with_default) - set(scripts_with_tiktok)}")
    
    # 2. Content Matching Analysis
    print("\n🎯 Content Matching Analysis")
    print("-" * 40)
    
    # Initialize enhanced analyzer
    analyzer = EnhancedAudioScriptAnalyzer("11-scripts-for-tiktok", "cache/whisper")
    await analyzer.analyze_scripts()
    
    # Get sample analysis
    sample_script = "anxiety1"
    analysis = analyzer.get_script_analysis(sample_script)
    
    if analysis:
        print(f"Enhanced Analysis for {sample_script}:")
        print(f"  Primary emotion: {analysis.primary_emotion}")
        print(f"  Themes: {analysis.themes[:5]}")
        print(f"  Semantic phrases: {len(analysis.semantic_phrases)}")
        print(f"  Content keywords: {len(analysis.content_keywords)}")
        # Check if thematic_weights exists
        if hasattr(analysis, 'thematic_weights'):
            print(f"  Thematic weights: {dict(list(analysis.thematic_weights.items())[:3])}")
        else:
            print(f"  Thematic weights: Not available")
        
        # Check semantic coverage
        print(f"\n🧠 Semantic Coverage Analysis:")
        phrase_categories = {}
        for phrase_data in analysis.semantic_phrases:
            category = phrase_data.get('category', 'unknown')
            if category not in phrase_categories:
                phrase_categories[category] = 0
            phrase_categories[category] += 1
        
        print(f"  Phrase categories: {phrase_categories}")
        print(f"  Top keywords: {analysis.content_keywords[:10]}")
    
    # 3. Improvement Recommendations
    print("\n💡 Improvement Recommendations")
    print("-" * 40)
    
    recommendations = [
        "1. CAPTION FIX: Modify caption selector to fallback to 'default' style when 'tiktok' not available",
        "2. CONTENT MATCHING: Enhance semantic phrase matching with weighted categories",
        "3. VISUAL TAGS: Improve visual characteristic matching with better tag analysis", 
        "4. THEMATIC WEIGHTS: Use dynamic theme weighting based on script analysis",
        "5. DEBUGGING: Add detailed matching score logging for optimization"
    ]
    
    for rec in recommendations:
        print(rec)
    
    return True

async def main():
    """Main analysis function"""
    await improve_content_matching()

if __name__ == "__main__":
    asyncio.run(main())