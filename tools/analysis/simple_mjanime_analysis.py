#!/usr/bin/env python3
"""
Simple MJAnime Analysis
Analyzes 5 MJAnime videos with basic content analysis
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any
import hashlib
from datetime import datetime
import sys

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class SimpleMJAnimeAnalyzer:
    """Simple analyzer for MJAnime clips without complex dependencies"""
    
    def __init__(self, clips_directory: str, metadata_file: str):
        self.clips_directory = Path(clips_directory)
        self.metadata_file = Path(metadata_file)
        self.clips = {}
        
    def load_clips(self):
        """Load clip metadata from JSON file"""
        if not self.metadata_file.exists():
            logger.error(f"Metadata file not found: {self.metadata_file}")
            return False
        
        with open(self.metadata_file, 'r') as f:
            metadata = json.load(f)
        
        clips_data = metadata.get('clips', [])
        for clip_data in clips_data:
            self.clips[clip_data['id']] = clip_data
        
        logger.info(f"Loaded {len(self.clips)} clips")
        return True
    
    def get_clips_by_tags(self, target_tags: List[str], limit: int = 5) -> List[Dict]:
        """Get clips that match target tags"""
        matching_clips = []
        
        for clip_id, clip_data in self.clips.items():
            clip_tags = [tag.lower() for tag in clip_data.get('tags', [])]
            target_tags_lower = [tag.lower() for tag in target_tags]
            
            # Calculate match score
            matches = sum(1 for tag in target_tags_lower if any(tag in clip_tag for clip_tag in clip_tags))
            
            if matches > 0:
                score = matches / len(target_tags_lower)
                matching_clips.append({
                    'clip_data': clip_data,
                    'match_score': score,
                    'matching_tags': [tag for tag in target_tags_lower if any(tag in clip_tag for clip_tag in clip_tags)]
                })
        
        # Sort by match score
        matching_clips.sort(key=lambda x: x['match_score'], reverse=True)
        return matching_clips[:limit]
    
    def analyze_clip_content(self, clip_data: Dict) -> Dict[str, Any]:
        """Analyze individual clip content"""
        filename = clip_data['filename']
        tags = clip_data.get('tags', [])
        shot_analysis = clip_data.get('shot_analysis', {})
        
        # Determine content themes
        themes = []
        if any(tag in ['meditation', 'peaceful', 'calm'] for tag in tags):
            themes.append('peaceful')
        if any(tag in ['spiritual', 'sacred', 'devotional'] for tag in tags):
            themes.append('spiritual')
        if any(tag in ['nature', 'water', 'waterfall', 'mountain'] for tag in tags):
            themes.append('natural')
        if any(tag in ['individual', 'alone'] for tag in tags):
            themes.append('solitary')
        if any(tag in ['community', 'group'] for tag in tags):
            themes.append('social')
        
        # Determine mood
        if 'dramatic' in shot_analysis.get('lighting', ''):
            mood = 'intense'
        elif any(tag in ['peaceful', 'calm', 'tranquil'] for tag in tags):
            mood = 'serene'
        elif any(tag in ['bright', 'awakening'] for tag in tags):
            mood = 'uplifting'
        else:
            mood = 'neutral'
        
        return {
            'filename': filename,
            'themes': themes,
            'mood': mood,
            'lighting': shot_analysis.get('lighting', 'natural'),
            'movement': shot_analysis.get('camera_movement', 'static'),
            'shot_type': shot_analysis.get('shot_type', 'medium_shot'),
            'primary_tags': tags[:5],  # Top 5 tags
            'duration': clip_data.get('duration', 5.21),
            'file_size_mb': clip_data.get('file_size_mb', 0)
        }
    
    def analyze_script_compatibility(self, clip_analysis: Dict, script_keywords: List[str]) -> Dict[str, Any]:
        """Analyze how well a clip matches script keywords"""
        clip_tags = [tag.lower() for tag in clip_analysis['primary_tags']]
        script_keywords_lower = [kw.lower() for kw in script_keywords]
        
        # Find matches
        matches = []
        for keyword in script_keywords_lower:
            for tag in clip_tags:
                if keyword in tag or tag in keyword:
                    matches.append((keyword, tag))
        
        # Calculate compatibility score
        base_score = len(matches) / max(len(script_keywords), 1)
        
        # Boost score for thematic alignment
        if 'peaceful' in script_keywords_lower and 'peaceful' in clip_analysis['themes']:
            base_score += 0.2
        if 'spiritual' in script_keywords_lower and 'spiritual' in clip_analysis['themes']:
            base_score += 0.2
        if 'meditation' in script_keywords_lower and clip_analysis['mood'] == 'serene':
            base_score += 0.1
        
        compatibility_score = min(base_score, 1.0)  # Cap at 1.0
        
        return {
            'compatibility_score': compatibility_score,
            'keyword_matches': matches,
            'thematic_alignment': {
                'clip_themes': clip_analysis['themes'],
                'clip_mood': clip_analysis['mood']
            },
            'recommendation_level': 'high' if compatibility_score >= 0.7 else 'medium' if compatibility_score >= 0.4 else 'low'
        }

def analyze_5_videos():
    """Analyze 5 MJAnime videos"""
    
    print("🎬 Analyzing 5 MJAnime Videos")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = SimpleMJAnimeAnalyzer(
        clips_directory="./MJAnime/fixed",
        metadata_file="./mjanime_metadata.json"
    )
    
    # Load clips
    if not analyzer.load_clips():
        print("❌ Failed to load clips")
        return
    
    # Test scripts with different themes
    test_scripts = {
        "Peaceful Meditation": {
            "text": "A devotee sits in peaceful meditation by a tranquil waterfall, finding inner serenity",
            "keywords": ["peaceful", "meditation", "devotee", "tranquil", "waterfall", "serenity"]
        },
        "Spiritual Journey": {
            "text": "Walking barefoot along a sacred path, chanting mantras with spiritual devotion",
            "keywords": ["walking", "barefoot", "sacred", "chanting", "spiritual", "devotional"]
        },
        "Natural Contemplation": {
            "text": "In a quiet garden, surrounded by nature's beauty, deep contemplation unfolds",
            "keywords": ["quiet", "garden", "nature", "contemplation", "peaceful", "natural"]
        },
        "Temple Worship": {
            "text": "Inside a sacred temple, devotees offer prayers with reverent hearts",
            "keywords": ["temple", "sacred", "devotees", "prayers", "reverent", "spiritual"]
        },
        "Mountain Solitude": {
            "text": "Alone on a mountain cliff, finding wisdom in solitary reflection",
            "keywords": ["mountain", "cliff", "alone", "solitary", "reflection", "wisdom"]
        }
    }
    
    # Analyze each script
    all_results = []
    
    for script_name, script_info in test_scripts.items():
        print(f"\n📖 Analyzing: {script_name}")
        print(f"Script: {script_info['text']}")
        print("-" * 40)
        
        # Find matching clips
        matching_clips = analyzer.get_clips_by_tags(script_info['keywords'], limit=5)
        
        if not matching_clips:
            print("❌ No matching clips found")
            continue
        
        script_results = {
            'script_name': script_name,
            'script_text': script_info['text'],
            'keywords': script_info['keywords'],
            'clip_analyses': []
        }
        
        # Analyze top matching clips
        for i, match in enumerate(matching_clips, 1):
            clip_data = match['clip_data']
            
            # Analyze clip content
            clip_analysis = analyzer.analyze_clip_content(clip_data)
            
            # Check script compatibility
            compatibility = analyzer.analyze_script_compatibility(clip_analysis, script_info['keywords'])
            
            # Combine results
            full_analysis = {
                'rank': i,
                'initial_match_score': match['match_score'],
                'matching_tags': match['matching_tags'],
                'clip_analysis': clip_analysis,
                'script_compatibility': compatibility
            }
            
            script_results['clip_analyses'].append(full_analysis)
            
            # Print results
            print(f"{i}. {clip_analysis['filename']}")
            print(f"   Themes: {', '.join(clip_analysis['themes'])}")
            print(f"   Mood: {clip_analysis['mood']}")
            print(f"   Compatibility: {compatibility['compatibility_score']:.2f} ({compatibility['recommendation_level']})")
            print(f"   Matches: {', '.join([f'{kw}→{tag}' for kw, tag in compatibility['keyword_matches']])}")
            print(f"   Style: {clip_analysis['lighting']} lighting, {clip_analysis['movement']} movement")
        
        all_results.append(script_results)
    
    # Summary analysis
    print(f"\n📊 ANALYSIS SUMMARY")
    print("=" * 50)
    
    total_clips_analyzed = sum(len(result['clip_analyses']) for result in all_results)
    high_compatibility_clips = sum(
        1 for result in all_results 
        for analysis in result['clip_analyses'] 
        if analysis['script_compatibility']['recommendation_level'] == 'high'
    )
    
    print(f"Total scripts tested: {len(test_scripts)}")
    print(f"Total clips analyzed: {total_clips_analyzed}")
    print(f"High compatibility clips: {high_compatibility_clips}")
    print(f"Success rate: {high_compatibility_clips/total_clips_analyzed:.1%}")
    
    # Top recommendations across all scripts
    print(f"\n🌟 TOP RECOMMENDATIONS")
    print("-" * 30)
    
    all_clips = []
    for result in all_results:
        for analysis in result['clip_analyses']:
            all_clips.append({
                'script': result['script_name'],
                'filename': analysis['clip_analysis']['filename'],
                'compatibility': analysis['script_compatibility']['compatibility_score'],
                'themes': analysis['clip_analysis']['themes']
            })
    
    # Sort by compatibility
    all_clips.sort(key=lambda x: x['compatibility'], reverse=True)
    
    for i, clip in enumerate(all_clips[:10], 1):
        print(f"{i}. {clip['filename']}")
        print(f"   Best for: {clip['script']}")
        print(f"   Score: {clip['compatibility']:.2f}")
        print(f"   Themes: {', '.join(clip['themes'])}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"mjanime_analysis_results_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump({
            'analysis_timestamp': datetime.now().isoformat(),
            'total_clips_analyzed': total_clips_analyzed,
            'scripts_tested': len(test_scripts),
            'results': all_results,
            'summary': {
                'high_compatibility_clips': high_compatibility_clips,
                'success_rate': high_compatibility_clips/total_clips_analyzed
            }
        }, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: {results_file}")
    print("\n✅ Analysis complete!")

if __name__ == "__main__":
    analyze_5_videos()