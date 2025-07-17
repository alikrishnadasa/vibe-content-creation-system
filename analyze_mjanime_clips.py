#!/usr/bin/env python3
"""
MJAnime Clip Analyzer

Analyzes MJAnime video clips using the Video Clip Contextualizer system.
Integrates the semantic video analysis with the existing MJAnime content pipeline.
"""

import argparse
import asyncio
import logging
import sys
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# Add video-clip-contextualizer to path
sys.path.insert(0, str(Path(__file__).parent / "video-clip-contextualizer"))
sys.path.insert(0, str(Path(__file__).parent / "unified-video-system-main"))

from video_clip_contextualizer.src.batch_folder_analyzer import BatchFolderAnalyzer
from unified_video_system_main.content.mjanime_loader import MJAnimeLoader, ClipMetadata


class MJAnimeAnalyzer:
    """Analyzes MJAnime clips using the Video Clip Contextualizer."""
    
    def __init__(self, 
                 mjanime_directory: str,
                 analysis_results_dir: str = "./mjanime_analysis_results",
                 save_metadata: bool = True):
        """
        Initialize MJAnime analyzer.
        
        Args:
            mjanime_directory: Path to MJAnime clips directory
            analysis_results_dir: Directory to save analysis results
            save_metadata: Whether to save analysis metadata
        """
        self.mjanime_directory = Path(mjanime_directory)
        self.analysis_results_dir = Path(analysis_results_dir)
        self.save_metadata = save_metadata
        self.logger = logging.getLogger(__name__)
        
        # Initialize MJAnime loader
        self.mjanime_loader = MJAnimeLoader(
            clips_directory=str(self.mjanime_directory),
            metadata_file=str(self.mjanime_directory.parent / "mjanime_metadata.json")
        )
        
        # Initialize video analyzer
        self.video_analyzer = BatchFolderAnalyzer(
            save_metadata=save_metadata,
            metadata_dir=str(self.analysis_results_dir),
            save_in_folder=False,
            overwrite_existing=False,
            cleanup_temp=True
        )
        
        # Create analysis results directory
        if save_metadata:
            self.analysis_results_dir.mkdir(parents=True, exist_ok=True)
    
    async def load_mjanime_clips(self) -> bool:
        """Load MJAnime clips metadata."""
        self.logger.info("Loading MJAnime clips...")
        success = await self.mjanime_loader.load_clips()
        
        if success:
            stats = self.mjanime_loader.get_clip_stats()
            self.logger.info(f"Loaded {stats['total_clips']} MJAnime clips")
            self.logger.info(f"Total duration: {stats['total_duration_seconds']:.1f}s")
            self.logger.info(f"Emotion distribution: {stats['emotion_distribution']}")
        
        return success
    
    def analyze_clips_by_script(self, 
                               script_content: str,
                               emotional_filter: Optional[str] = None,
                               clip_duration: float = 5.0,
                               overlap: float = 0.5,
                               max_clips: Optional[int] = None) -> Dict[str, Any]:
        """
        Analyze MJAnime clips against a specific script.
        
        Args:
            script_content: Script text to match against clips
            emotional_filter: Filter clips by emotion ('anxiety', 'peace', etc.)
            clip_duration: Duration for video segmentation
            overlap: Overlap between segments
            max_clips: Maximum number of clips to analyze (for testing)
            
        Returns:
            Analysis results
        """
        if not self.mjanime_loader.loaded:
            raise RuntimeError("MJAnime clips not loaded. Call load_mjanime_clips() first.")
        
        # Get clips to analyze
        if emotional_filter:
            available_clips = self.mjanime_loader.get_clips_by_emotion(emotional_filter)
            self.logger.info(f"Found {len(available_clips)} clips matching emotion: {emotional_filter}")
        else:
            available_clips = list(self.mjanime_loader.clips.values())
            self.logger.info(f"Analyzing all {len(available_clips)} clips")
        
        if max_clips and len(available_clips) > max_clips:
            available_clips = available_clips[:max_clips]
            self.logger.info(f"Limited to {max_clips} clips for analysis")
        
        if not available_clips:
            return {"error": "No clips found matching criteria"}
        
        # Create temporary folder with symlinks to selected clips
        temp_clips_dir = self.analysis_results_dir / "temp_clips"
        temp_clips_dir.mkdir(exist_ok=True)
        
        # Clean existing symlinks
        for existing_link in temp_clips_dir.glob("*"):
            if existing_link.is_symlink():
                existing_link.unlink()
        
        # Create symlinks to selected clips
        clip_paths = []
        for clip in available_clips:
            src_path = Path(clip.filepath)
            if src_path.exists():
                link_path = temp_clips_dir / src_path.name
                try:
                    link_path.symlink_to(src_path.resolve())
                    clip_paths.append(str(link_path))
                except Exception as e:
                    self.logger.warning(f"Failed to create symlink for {src_path}: {e}")
                    # Fallback: use original path
                    clip_paths.append(str(src_path))
        
        self.logger.info(f"Created temporary analysis folder with {len(clip_paths)} clips")
        
        try:
            # Run batch analysis
            self.video_analyzer.clip_duration = clip_duration
            self.video_analyzer.overlap = overlap
            
            results = self.video_analyzer.analyze_folder(
                video_folder=str(temp_clips_dir),
                script_text=script_content,
                clip_duration=clip_duration,
                overlap=overlap,
                matching_strategy="optimal",
                file_pattern="*"
            )
            
            # Enhance results with MJAnime metadata
            enhanced_results = self._enhance_results_with_metadata(results, available_clips)
            
            return enhanced_results
            
        finally:
            # Cleanup temporary symlinks
            for link_path in temp_clips_dir.glob("*"):
                if link_path.is_symlink():
                    link_path.unlink()
    
    def analyze_clips_by_emotion(self, 
                                emotion: str,
                                sample_scripts: Dict[str, str],
                                clip_duration: float = 5.0,
                                max_clips: int = 10) -> Dict[str, Any]:
        """
        Analyze clips of a specific emotion against multiple sample scripts.
        
        Args:
            emotion: Target emotion to analyze
            sample_scripts: Dictionary of script_name -> script_content
            clip_duration: Duration for video segmentation
            max_clips: Maximum clips to analyze per script
            
        Returns:
            Analysis results for emotion category
        """
        emotion_clips = self.mjanime_loader.get_clips_by_emotion(emotion)
        
        if not emotion_clips:
            return {"error": f"No clips found for emotion: {emotion}"}
        
        if len(emotion_clips) > max_clips:
            emotion_clips = emotion_clips[:max_clips]
        
        results = {
            "emotion": emotion,
            "clips_analyzed": len(emotion_clips),
            "script_analyses": {}
        }
        
        for script_name, script_content in sample_scripts.items():
            self.logger.info(f"Analyzing {emotion} clips against script: {script_name}")
            
            script_results = self.analyze_clips_by_script(
                script_content=script_content,
                emotional_filter=emotion,
                clip_duration=clip_duration,
                max_clips=max_clips
            )
            
            results["script_analyses"][script_name] = script_results
        
        return results
    
    def generate_clip_recommendations(self, 
                                    script_content: str,
                                    target_emotion: Optional[str] = None,
                                    min_confidence: float = 0.7,
                                    max_recommendations: int = 5) -> List[Dict[str, Any]]:
        """
        Generate clip recommendations for a script.
        
        Args:
            script_content: Script to find clips for
            target_emotion: Preferred emotional tone
            min_confidence: Minimum matching confidence
            max_recommendations: Maximum clips to recommend
            
        Returns:
            List of recommended clips with metadata
        """
        # Analyze all clips or filtered by emotion
        analysis_results = self.analyze_clips_by_script(
            script_content=script_content,
            emotional_filter=target_emotion,
            clip_duration=3.0,  # Shorter for faster analysis
            overlap=0.2
        )
        
        if "error" in analysis_results:
            return []
        
        # Extract high-confidence matches
        recommendations = []
        
        for video_path, result in analysis_results.get("results", {}).items():
            if not result.get("processed", False):
                continue
            
            # Get clip metadata
            clip_filename = Path(video_path).name
            clip_metadata = None
            for clip in self.mjanime_loader.clips.values():
                if Path(clip.filepath).name == clip_filename:
                    clip_metadata = clip
                    break
            
            if not clip_metadata:
                continue
            
            # Check match quality
            avg_confidence = result["matching"]["metrics"]["avg_confidence"]
            if avg_confidence >= min_confidence:
                recommendations.append({
                    "clip_id": clip_metadata.id,
                    "filename": clip_metadata.filename,
                    "confidence": avg_confidence,
                    "emotional_tags": clip_metadata.emotional_tags,
                    "matches_found": result["matching"]["matches_found"],
                    "duration": clip_metadata.duration,
                    "lighting_type": clip_metadata.lighting_type,
                    "movement_type": clip_metadata.movement_type,
                    "shot_type": clip_metadata.shot_type,
                    "best_matches": result["matches"][:3]  # Top 3 matches
                })
        
        # Sort by confidence and limit
        recommendations.sort(key=lambda x: x["confidence"], reverse=True)
        return recommendations[:max_recommendations]
    
    def _enhance_results_with_metadata(self, 
                                     analysis_results: Dict[str, Any],
                                     analyzed_clips: List[ClipMetadata]) -> Dict[str, Any]:
        """Enhance analysis results with MJAnime metadata."""
        
        # Create lookup for clip metadata by filename
        clip_lookup = {}
        for clip in analyzed_clips:
            filename = Path(clip.filepath).name
            clip_lookup[filename] = clip
        
        # Enhance individual results
        enhanced_results = analysis_results.copy()
        
        for video_path, result in enhanced_results.get("results", {}).items():
            filename = Path(video_path).name
            if filename in clip_lookup:
                clip_metadata = clip_lookup[filename]
                
                # Add MJAnime metadata to result
                result["mjanime_metadata"] = {
                    "clip_id": clip_metadata.id,
                    "emotional_tags": clip_metadata.emotional_tags,
                    "lighting_type": clip_metadata.lighting_type,
                    "movement_type": clip_metadata.movement_type,
                    "shot_type": clip_metadata.shot_type,
                    "original_tags": clip_metadata.tags,
                    "shot_analysis": clip_metadata.shot_analysis
                }
                
                # Enhance semantic analysis with MJAnime context
                if "semantic_analysis" in result:
                    result["semantic_analysis"]["mjanime_context"] = {
                        "emotional_category": clip_metadata.emotional_tags[0] if clip_metadata.emotional_tags else "neutral",
                        "visual_style": clip_metadata.lighting_type,
                        "camera_work": clip_metadata.movement_type,
                        "composition": clip_metadata.shot_type
                    }
        
        # Add MJAnime summary to batch info
        enhanced_results["mjanime_summary"] = {
            "clips_analyzed": len(analyzed_clips),
            "emotion_distribution": self._get_emotion_distribution(analyzed_clips),
            "lighting_distribution": self._get_lighting_distribution(analyzed_clips),
            "movement_distribution": self._get_movement_distribution(analyzed_clips)
        }
        
        return enhanced_results
    
    def _get_emotion_distribution(self, clips: List[ClipMetadata]) -> Dict[str, int]:
        """Get emotion distribution for analyzed clips."""
        distribution = {}
        for clip in clips:
            for emotion in clip.emotional_tags:
                distribution[emotion] = distribution.get(emotion, 0) + 1
        return distribution
    
    def _get_lighting_distribution(self, clips: List[ClipMetadata]) -> Dict[str, int]:
        """Get lighting distribution for analyzed clips."""
        distribution = {}
        for clip in clips:
            lighting = clip.lighting_type or "unknown"
            distribution[lighting] = distribution.get(lighting, 0) + 1
        return distribution
    
    def _get_movement_distribution(self, clips: List[ClipMetadata]) -> Dict[str, int]:
        """Get camera movement distribution for analyzed clips."""
        distribution = {}
        for clip in clips:
            movement = clip.movement_type or "unknown"
            distribution[movement] = distribution.get(movement, 0) + 1
        return distribution
    
    def save_analysis_report(self, 
                           results: Dict[str, Any], 
                           report_name: str) -> Path:
        """Save analysis results as a formatted report."""
        if not self.save_metadata:
            raise RuntimeError("Metadata saving is disabled")
        
        report_file = self.analysis_results_dir / f"{report_name}.json"
        
        # Add timestamp and analysis metadata
        enhanced_results = {
            "analysis_timestamp": datetime.now().isoformat(),
            "analyzer_version": "1.0.0",
            "mjanime_integration": True,
            "results": results
        }
        
        with open(report_file, 'w') as f:
            json.dump(enhanced_results, f, indent=2, default=str)
        
        self.logger.info(f"Analysis report saved: {report_file}")
        return report_file


def create_sample_scripts() -> Dict[str, str]:
    """Create sample scripts for testing different emotional tones."""
    return {
        "anxiety_script": """
        Life feels overwhelming. The pressure builds like storm clouds gathering. 
        Every decision seems monumental, every path uncertain. The weight of 
        expectations crushes down, making each breath feel labored. Darkness 
        creeps in from the edges of vision, and escape feels impossible.
        """,
        
        "peace_script": """
        In this moment, everything is still. The gentle breeze carries away 
        all worries, leaving only tranquil silence. Meditation brings clarity 
        to the mind. Floating on calm waters, feeling completely safe and 
        centered. The lotus blooms in perfect serenity.
        """,
        
        "spiritual_awakening_script": """
        Light breaks through the darkness, revealing ancient wisdom. The temple 
        bells ring with sacred resonance. Enlightenment dawns like sunrise over 
        mountain peaks. Every step on this spiritual journey brings deeper 
        understanding. The divine presence illuminates the path forward.
        """,
        
        "social_connection_script": """
        Friends gather in celebration, sharing joy and laughter. The community 
        comes together in harmony. Festivals bring people from all walks of life. 
        In this crowd, no one feels alone. The energy of togetherness creates 
        something beautiful and powerful.
        """,
        
        "isolation_script": """
        Standing alone on the cliff edge, contemplating existence. The solitary 
        figure walks through empty landscapes. In quiet moments of introspection, 
        the individual discovers inner strength. Sometimes the deepest insights 
        come from being completely alone with one's thoughts.
        """
    }


def setup_logging(level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('mjanime_analysis.log')
        ]
    )


async def main():
    parser = argparse.ArgumentParser(
        description="Analyze MJAnime clips using Video Clip Contextualizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze anxiety clips against sample scripts
  python analyze_mjanime_clips.py /path/to/MJAnime/fixed --emotion anxiety
  
  # Test with custom script
  python analyze_mjanime_clips.py /path/to/MJAnime/fixed --script "custom script text"
  
  # Generate recommendations for a script
  python analyze_mjanime_clips.py /path/to/MJAnime/fixed --recommend --script-file script.txt
  
  # Quick test with limited clips
  python analyze_mjanime_clips.py /path/to/MJAnime/fixed --test --max-clips 5
        """
    )
    
    parser.add_argument("mjanime_directory", 
                       help="Path to MJAnime clips directory")
    parser.add_argument("--emotion", 
                       choices=["anxiety", "peace", "seeking", "awakening", "safe", "social", "isolated", "neutral"],
                       help="Analyze clips of specific emotion")
    parser.add_argument("--script", 
                       help="Script text to analyze against")
    parser.add_argument("--script-file", 
                       help="Path to script file")
    parser.add_argument("--recommend", action="store_true",
                       help="Generate clip recommendations")
    parser.add_argument("--test", action="store_true",
                       help="Run quick test with sample scripts")
    parser.add_argument("--max-clips", type=int, default=10,
                       help="Maximum clips to analyze (default: 10)")
    parser.add_argument("--clip-duration", type=float, default=5.0,
                       help="Video clip duration for analysis (default: 5.0)")
    parser.add_argument("--results-dir", default="./mjanime_analysis_results",
                       help="Directory to save results")
    parser.add_argument("--no-metadata", action="store_true",
                       help="Don't save metadata to disk")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       default="INFO", help="Set logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize analyzer
        analyzer = MJAnimeAnalyzer(
            mjanime_directory=args.mjanime_directory,
            analysis_results_dir=args.results_dir,
            save_metadata=not args.no_metadata
        )
        
        # Load MJAnime clips
        success = await analyzer.load_mjanime_clips()
        if not success:
            print("Failed to load MJAnime clips", file=sys.stderr)
            sys.exit(1)
        
        # Determine script content
        if args.script_file:
            script_path = Path(args.script_file)
            if not script_path.exists():
                print(f"Script file not found: {script_path}", file=sys.stderr)
                sys.exit(1)
            with open(script_path, 'r') as f:
                script_content = f.read()
        elif args.script:
            script_content = args.script
        else:
            script_content = None
        
        # Run analysis based on mode
        if args.test:
            # Quick test mode
            logger.info("Running quick test with sample scripts...")
            sample_scripts = create_sample_scripts()
            
            for script_name, script_text in sample_scripts.items():
                print(f"\n{'='*60}")
                print(f"Testing: {script_name}")
                print(f"{'='*60}")
                
                results = analyzer.analyze_clips_by_script(
                    script_content=script_text,
                    clip_duration=args.clip_duration,
                    max_clips=args.max_clips
                )
                
                if "error" not in results:
                    batch_info = results["batch_info"]
                    print(f"Processed: {batch_info['processed_videos']} clips")
                    print(f"Success rate: {batch_info['success_rate']:.1%}")
                    
                    if "summary" in results:
                        summary = results["summary"]
                        print(f"Average confidence: {summary['average_confidence']:.3f}")
                else:
                    print(f"Error: {results['error']}")
        
        elif args.recommend and script_content:
            # Recommendation mode
            logger.info("Generating clip recommendations...")
            recommendations = analyzer.generate_clip_recommendations(
                script_content=script_content,
                target_emotion=args.emotion,
                max_recommendations=args.max_clips
            )
            
            print(f"\n{'='*60}")
            print(f"CLIP RECOMMENDATIONS")
            print(f"{'='*60}")
            
            if recommendations:
                for i, rec in enumerate(recommendations, 1):
                    print(f"\n{i}. {rec['filename']}")
                    print(f"   Confidence: {rec['confidence']:.3f}")
                    print(f"   Emotions: {', '.join(rec['emotional_tags'])}")
                    print(f"   Style: {rec['lighting_type']} lighting, {rec['movement_type']} movement")
                    print(f"   Matches: {rec['matches_found']}")
            else:
                print("No high-confidence recommendations found")
        
        elif args.emotion:
            # Emotion-specific analysis
            logger.info(f"Analyzing {args.emotion} clips...")
            
            if script_content:
                results = analyzer.analyze_clips_by_script(
                    script_content=script_content,
                    emotional_filter=args.emotion,
                    clip_duration=args.clip_duration,
                    max_clips=args.max_clips
                )
            else:
                # Use sample scripts for this emotion
                sample_scripts = create_sample_scripts()
                emotion_script = f"{args.emotion}_script"
                if emotion_script in sample_scripts:
                    script_content = sample_scripts[emotion_script]
                else:
                    script_content = list(sample_scripts.values())[0]
                
                results = analyzer.analyze_clips_by_script(
                    script_content=script_content,
                    emotional_filter=args.emotion,
                    clip_duration=args.clip_duration,
                    max_clips=args.max_clips
                )
            
            if "error" not in results:
                # Save results
                if not args.no_metadata:
                    report_path = analyzer.save_analysis_report(
                        results, 
                        f"emotion_analysis_{args.emotion}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    )
                    print(f"Results saved to: {report_path}")
                
                # Print summary
                batch_info = results["batch_info"]
                print(f"\n{'='*60}")
                print(f"EMOTION ANALYSIS: {args.emotion.upper()}")
                print(f"{'='*60}")
                print(f"Clips processed: {batch_info['processed_videos']}")
                print(f"Success rate: {batch_info['success_rate']:.1%}")
                
                if "summary" in results and not results["summary"].get("no_data"):
                    summary = results["summary"]
                    print(f"Average confidence: {summary['average_confidence']:.3f}")
                    print(f"Total matches: {summary['total_matches']}")
            else:
                print(f"Error: {results['error']}")
        
        elif script_content:
            # General script analysis
            logger.info("Analyzing all clips against provided script...")
            results = analyzer.analyze_clips_by_script(
                script_content=script_content,
                clip_duration=args.clip_duration,
                max_clips=args.max_clips
            )
            
            if "error" not in results:
                if not args.no_metadata:
                    report_path = analyzer.save_analysis_report(
                        results, 
                        f"script_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    )
                    print(f"Results saved to: {report_path}")
                
                batch_info = results["batch_info"]
                print(f"\nAnalyzed {batch_info['processed_videos']} clips")
                print(f"Success rate: {batch_info['success_rate']:.1%}")
            else:
                print(f"Error: {results['error']}")
        
        else:
            print("Please specify --emotion, --script, --script-file, --test, or --recommend")
            parser.print_help()
    
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())