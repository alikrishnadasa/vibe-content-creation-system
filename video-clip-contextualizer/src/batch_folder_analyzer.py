#!/usr/bin/env python3
"""
Batch Folder Analyzer for Video Clip Contextualizer

Analyzes all videos in a folder against scripts with optional metadata persistence.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import hashlib
import shutil

from .processing.video_processor import VideoProcessor
from .processing.text_processor import TextProcessor
from .matching.semantic_matcher import SemanticMatcher
from .monitoring.performance_monitor import PerformanceMonitor
from .analysis.ai_semantic_tagger import AISemanticTagger
from .config import get_config


class BatchFolderAnalyzer:
    """Batch analyze videos in a folder with configurable metadata persistence."""
    
    def __init__(self, 
                 save_metadata: bool = True,
                 metadata_dir: Optional[str] = None,
                 save_in_folder: bool = False,
                 overwrite_existing: bool = False,
                 cleanup_temp: bool = True):
        """
        Initialize batch folder analyzer.
        
        Args:
            save_metadata: Whether to save analysis results to disk
            metadata_dir: Directory to save metadata (default: ./analysis_results)
            save_in_folder: Whether to save metadata in the same folder as videos
            overwrite_existing: Whether to overwrite existing analysis files
            cleanup_temp: Whether to clean up temporary files after processing
        """
        self.config = get_config()
        self.logger = logging.getLogger(__name__)
        
        # Metadata persistence settings
        self.save_metadata = save_metadata
        self.save_in_folder = save_in_folder
        self.metadata_dir = Path(metadata_dir) if metadata_dir else Path("./analysis_results")
        self.overwrite_existing = overwrite_existing
        self.cleanup_temp = cleanup_temp
        
        # Note: metadata_dir will be set to video folder in analyze_folder if save_in_folder=True
        
        # Initialize processors
        self.video_processor = VideoProcessor()
        self.text_processor = TextProcessor()
        self.semantic_matcher = None  # Disable semantic matching for CPU-only mode
        self.ai_semantic_tagger = None  # Disable AI semantic analysis for now
        self.performance_monitor = PerformanceMonitor()
        
        # Supported video formats
        self.video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.m4v'}
        
        # Analysis statistics
        self.stats = {
            "total_videos": 0,
            "processed_videos": 0,
            "skipped_videos": 0,
            "failed_videos": 0,
            "total_processing_time": 0.0,
            "start_time": None,
            "end_time": None
        }
    
    def analyze_folder(self, 
                      video_folder: str,
                      script_text: str,
                      clip_duration: float = 5.0,
                      overlap: float = 0.5,
                      matching_strategy: str = "optimal",
                      file_pattern: str = "*") -> Dict[str, Any]:
        """
        Analyze all videos in a folder against a script.
        
        Args:
            video_folder: Path to folder containing videos
            script_text: Text script to match against all videos
            clip_duration: Duration of video clips in seconds
            overlap: Overlap between clips in seconds
            matching_strategy: Matching algorithm to use
            file_pattern: File pattern to match (default: all files)
            
        Returns:
            Batch analysis results
        """
        video_folder = Path(video_folder)
        if not video_folder.exists():
            raise FileNotFoundError(f"Video folder not found: {video_folder}")
        
        self.logger.info(f"Starting batch analysis of folder: {video_folder}")
        self.stats["start_time"] = datetime.now()
        
        # Set metadata directory to video folder if requested
        if self.save_metadata and self.save_in_folder:
            self.metadata_dir = video_folder
            self.logger.info(f"Metadata will be saved in video folder: {self.metadata_dir}")
        elif self.save_metadata:
            self.metadata_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Metadata will be saved to: {self.metadata_dir}")
        
        # Configure processors
        self.video_processor.clip_duration = clip_duration
        self.video_processor.overlap = overlap
        
        # Find all video files
        video_files = self._find_video_files(video_folder, file_pattern)
        self.stats["total_videos"] = len(video_files)
        
        if not video_files:
            self.logger.warning(f"No video files found in {video_folder}")
            return self._create_batch_results({})
        
        self.logger.info(f"Found {len(video_files)} video files to process")
        
        # Process text once (shared across all videos)
        text_segments = self.text_processor.segment_text(script_text)
        text_segments = self.text_processor.generate_embeddings(text_segments)
        
        # Process each video
        results = {}
        for video_file in video_files:
            try:
                result = self._analyze_single_video(
                    video_file, 
                    text_segments, 
                    script_text,
                    matching_strategy
                )
                results[str(video_file)] = result
                self.stats["processed_videos"] += 1
                
            except Exception as e:
                self.logger.error(f"Failed to process {video_file}: {str(e)}")
                results[str(video_file)] = {
                    "error": str(e),
                    "processed": False,
                    "timestamp": datetime.now().isoformat()
                }
                self.stats["failed_videos"] += 1
        
        self.stats["end_time"] = datetime.now()
        self.stats["total_processing_time"] = (
            self.stats["end_time"] - self.stats["start_time"]
        ).total_seconds()
        
        # Create final batch results
        batch_results = self._create_batch_results(results)
        
        # Save metadata if requested
        if self.save_metadata:
            self._save_batch_metadata(batch_results, video_folder)
        
        # Cleanup if requested
        if self.cleanup_temp:
            self._cleanup_temp_files()
        
        self.logger.info(f"Batch analysis completed: {self.stats['processed_videos']}/{self.stats['total_videos']} videos processed")
        
        return batch_results
    
    def _find_video_files(self, folder: Path, pattern: str) -> List[Path]:
        """Find all video files in folder matching pattern."""
        video_files = []
        
        for file_path in folder.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in self.video_extensions:
                video_files.append(file_path)
        
        # Sort for consistent processing order
        return sorted(video_files)
    
    def _analyze_single_video(self, 
                             video_file: Path,
                             text_segments: List,
                             script_text: str,
                             matching_strategy: str) -> Dict[str, Any]:
        """Analyze a single video file."""
        self.logger.info(f"Processing: {video_file.name}")
        
        # Check if already processed and not overwriting
        if not self.overwrite_existing:
            existing_result = self._load_existing_result(video_file)
            if existing_result:
                self.logger.info(f"Skipping {video_file.name} (already processed)")
                self.stats["skipped_videos"] += 1
                return existing_result
        
        # Start performance tracking
        request_id = self.performance_monitor.start_request(f"batch_{video_file.name}")
        
        try:
            start_time = datetime.now()
            
            # Process video
            video_segments = self.video_processor.process_video_file(str(video_file))
            
            # Perform matching
            if self.semantic_matcher:
                matches = self.semantic_matcher.match_video_to_text(
                    video_segments, 
                    text_segments, 
                    matching_strategy
                )
                # Calculate metrics
                metrics = self.semantic_matcher.compute_matching_metrics(matches)
            else:
                # Create basic placeholder matches
                matches = []
                for i, segment in enumerate(video_segments):
                    if i < len(text_segments):
                        # Create a simple match between video segment and text segment
                        from dataclasses import dataclass
                        
                        @dataclass
                        class SimpleMatch:
                            video_start: float
                            video_end: float
                            text_start: int
                            text_end: int
                            text_segment_idx: int
                            confidence: float
                            explanation: dict
                            
                        match = SimpleMatch(
                            video_start=segment.start_time,
                            video_end=segment.end_time,
                            text_start=text_segments[i].start_pos,
                            text_end=text_segments[i].end_pos,
                            text_segment_idx=i,
                            confidence=0.5,
                            explanation={
                                "match_type": "basic_temporal",
                                "video_duration": segment.duration,
                                "text_length": len(text_segments[i].text),
                                "text_preview": text_segments[i].text[:50],
                                "keywords": [],
                                "temporal_alignment": {"video_position": i / len(video_segments), "text_position": i / len(text_segments)}
                            }
                        )
                        matches.append(match)
                
                # Basic metrics
                metrics = {
                    "avg_confidence": 0.5,
                    "high_confidence_ratio": 0.0,
                    "total_matches": len(matches),
                    "min_confidence": 0.5,
                    "max_confidence": 0.5,
                    "std_confidence": 0.0
                }
            
            # Perform AI semantic analysis on video segments
            semantic_analyses = []
            if self.ai_semantic_tagger:
                for segment in video_segments:
                    semantic_analysis = self.ai_semantic_tagger.analyze_video_semantics(
                        segment.frames, segment.duration
                    )
                    semantic_analyses.append(semantic_analysis)
            else:
                # Create basic placeholder semantic analysis
                for segment in video_segments:
                    semantic_analyses.append({
                        "semantic_tags": {
                            "objects": ["video", "content"],
                            "people": [],
                            "environment": ["unknown"],
                            "activities": ["viewing"],
                            "emotions": ["neutral"],
                            "vibes": ["neutral"],
                            "concepts": ["video", "segment"],
                            "attributes": ["visual"]
                        },
                        "descriptions": {
                            "overall_description": "Video segment content",
                            "frame_captions": [],
                            "targeted_descriptions": {},
                            "combined_text": ""
                        },
                        "confidence_scores": {"overall": 0.5}
                    })
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # End performance tracking
            self.performance_monitor.end_request(
                request_id,
                success=True,
                confidence_score=metrics.get('avg_confidence', 0),
                video_segments=len(video_segments),
                text_segments=len(text_segments),
                matches_found=len(matches)
            )
            
            # Aggregate semantic analysis across all segments
            aggregated_semantics = self._aggregate_semantic_analysis(semantic_analyses)
            
            # Create result
            result = {
                "video_file": str(video_file),
                "video_name": video_file.name,
                "processed": True,
                "timestamp": datetime.now().isoformat(),
                "processing_time": processing_time,
                "video_info": {
                    "duration": video_segments[-1].end_time if video_segments else 0,
                    "segments_count": len(video_segments),
                    "file_size": video_file.stat().st_size
                },
                "script_info": {
                    "character_count": len(script_text),
                    "segments_count": len(text_segments)
                },
                "matching": {
                    "strategy": matching_strategy,
                    "matches_found": len(matches),
                    "metrics": metrics
                },
                "semantic_analysis": aggregated_semantics,
                "segment_semantics": [
                    {
                        "segment_index": i,
                        "time_range": {"start": segment.start_time, "end": segment.end_time},
                        "semantic_tags": analysis.get("semantic_tags", {}),
                        "tag_summary": self._get_basic_tag_summary(analysis)
                    }
                    for i, (segment, analysis) in enumerate(zip(video_segments, semantic_analyses))
                ],
                "matches": [
                    {
                        "video_segment": {
                            "start": match.video_start,
                            "end": match.video_end
                        },
                        "text_segment": {
                            "start": match.text_start,
                            "end": match.text_end,
                            "text": text_segments[match.text_segment_idx].text
                        },
                        "confidence": match.confidence,
                        "explanation": match.explanation
                    }
                    for match in matches
                ],
                "file_hash": self._calculate_file_hash(video_file)
            }
            
            # Save individual result if metadata saving is enabled
            if self.save_metadata:
                self._save_individual_result(video_file, result)
            
            return result
            
        except Exception as e:
            self.performance_monitor.end_request(request_id, success=False)
            raise
    
    def _create_batch_results(self, individual_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create comprehensive batch analysis results."""
        return {
            "batch_info": {
                "total_videos": self.stats["total_videos"],
                "processed_videos": self.stats["processed_videos"],
                "skipped_videos": self.stats["skipped_videos"],
                "failed_videos": self.stats["failed_videos"],
                "success_rate": self.stats["processed_videos"] / max(self.stats["total_videos"], 1),
                "total_processing_time": self.stats["total_processing_time"],
                "start_time": self.stats["start_time"].isoformat() if self.stats["start_time"] else None,
                "end_time": self.stats["end_time"].isoformat() if self.stats["end_time"] else None
            },
            "configuration": {
                "clip_duration": self.video_processor.clip_duration,
                "overlap": self.video_processor.overlap,
                "save_metadata": self.save_metadata,
                "metadata_dir": str(self.metadata_dir) if self.save_metadata else None,
                "cleanup_temp": self.cleanup_temp
            },
            "results": individual_results,
            "performance": self.performance_monitor.get_current_stats(),
            "summary": self._create_summary_stats(individual_results)
        }
    
    def _create_summary_stats(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Create summary statistics from individual results."""
        processed_results = [r for r in results.values() if r.get("processed", False)]
        
        if not processed_results:
            return {"no_data": True}
        
        # Aggregate metrics
        total_matches = sum(r["matching"]["matches_found"] for r in processed_results)
        avg_confidence = sum(r["matching"]["metrics"]["avg_confidence"] for r in processed_results) / len(processed_results)
        total_processing_time = sum(r["processing_time"] for r in processed_results)
        
        # Find best and worst matches
        best_video = max(processed_results, key=lambda r: r["matching"]["metrics"]["avg_confidence"])
        worst_video = min(processed_results, key=lambda r: r["matching"]["metrics"]["avg_confidence"])
        
        return {
            "total_matches": total_matches,
            "average_matches_per_video": total_matches / len(processed_results),
            "average_confidence": avg_confidence,
            "total_processing_time": total_processing_time,
            "average_processing_time": total_processing_time / len(processed_results),
            "best_match": {
                "video": best_video["video_name"],
                "confidence": best_video["matching"]["metrics"]["avg_confidence"]
            },
            "worst_match": {
                "video": worst_video["video_name"],
                "confidence": worst_video["matching"]["metrics"]["avg_confidence"]
            }
        }
    
    def _save_batch_metadata(self, batch_results: Dict[str, Any], video_folder: Path):
        """Save batch analysis metadata to disk."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = video_folder.name
        
        # Save comprehensive batch results
        if self.save_in_folder:
            batch_file = self.metadata_dir / f".video_analysis_complete.json"
            summary_file = self.metadata_dir / f".video_analysis_summary.txt"
        else:
            batch_file = self.metadata_dir / f"batch_{folder_name}_{timestamp}.json"
            summary_file = self.metadata_dir / f"summary_{folder_name}_{timestamp}.txt"
        
        with open(batch_file, 'w') as f:
            json.dump(batch_results, f, indent=2, default=str)
        
        # Save summary report
        self._save_summary_report(batch_results, summary_file)
        
        self.logger.info(f"Batch metadata saved to: {batch_file}")
        self.logger.info(f"Summary report saved to: {summary_file}")
    
    def _save_individual_result(self, video_file: Path, result: Dict[str, Any]):
        """Save individual video analysis result."""
        video_name = video_file.stem
        result_file = self.metadata_dir / f"{video_name}_analysis.json"
        
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2, default=str)
    
    def _load_existing_result(self, video_file: Path) -> Optional[Dict[str, Any]]:
        """Load existing analysis result if available."""
        if not self.save_metadata:
            return None
        
        video_name = video_file.stem
        result_file = self.metadata_dir / f"{video_name}_analysis.json"
        
        if not result_file.exists():
            return None
        
        try:
            with open(result_file, 'r') as f:
                existing_result = json.load(f)
            
            # Verify file hasn't changed
            current_hash = self._calculate_file_hash(video_file)
            if existing_result.get("file_hash") == current_hash:
                return existing_result
            
        except Exception as e:
            self.logger.warning(f"Failed to load existing result for {video_file}: {e}")
        
        return None
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of file for change detection."""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def _save_summary_report(self, batch_results: Dict[str, Any], report_file: Path):
        """Save human-readable summary report."""
        with open(report_file, 'w') as f:
            f.write("VIDEO CLIP CONTEXTUALIZER - BATCH ANALYSIS REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            # Batch info
            batch_info = batch_results["batch_info"]
            f.write(f"Analysis Date: {batch_info['end_time']}\n")
            f.write(f"Total Videos: {batch_info['total_videos']}\n")
            f.write(f"Processed: {batch_info['processed_videos']}\n")
            f.write(f"Skipped: {batch_info['skipped_videos']}\n")
            f.write(f"Failed: {batch_info['failed_videos']}\n")
            f.write(f"Success Rate: {batch_info['success_rate']:.1%}\n")
            f.write(f"Total Processing Time: {batch_info['total_processing_time']:.1f}s\n\n")
            
            # Configuration
            config = batch_results["configuration"]
            f.write("Configuration:\n")
            f.write(f"  Clip Duration: {config['clip_duration']}s\n")
            f.write(f"  Overlap: {config['overlap']}s\n")
            f.write(f"  Metadata Saved: {config['save_metadata']}\n\n")
            
            # Summary stats
            if "summary" in batch_results and not batch_results["summary"].get("no_data"):
                summary = batch_results["summary"]
                f.write("Summary Statistics:\n")
                f.write(f"  Total Matches: {summary['total_matches']}\n")
                f.write(f"  Average Matches per Video: {summary['average_matches_per_video']:.1f}\n")
                f.write(f"  Average Confidence: {summary['average_confidence']:.3f}\n")
                f.write(f"  Average Processing Time: {summary['average_processing_time']:.2f}s\n\n")
                
                f.write(f"Best Match: {summary['best_match']['video']} ({summary['best_match']['confidence']:.3f})\n")
                f.write(f"Worst Match: {summary['worst_match']['video']} ({summary['worst_match']['confidence']:.3f})\n\n")
            
            # Individual results
            f.write("Individual Results:\n")
            f.write("-" * 40 + "\n")
            
            for video_path, result in batch_results["results"].items():
                if result.get("processed"):
                    f.write(f"{result['video_name']}:\n")
                    f.write(f"  Matches: {result['matching']['matches_found']}\n")
                    f.write(f"  Avg Confidence: {result['matching']['metrics']['avg_confidence']:.3f}\n")
                    f.write(f"  Processing Time: {result['processing_time']:.2f}s\n")
                else:
                    f.write(f"{Path(video_path).name}: FAILED ({result.get('error', 'Unknown error')})\n")
    
    def _aggregate_semantic_analysis(self, semantic_analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate semantic analysis from all video segments."""
        if not semantic_analyses:
            return {}
        
        # Aggregate all semantic tags
        aggregated_tags = {
            "objects": [],
            "people": [],
            "environment": [],
            "activities": [],
            "emotions": [],
            "vibes": [],
            "concepts": [],
            "attributes": []
        }
        
        all_descriptions = []
        all_emotions = []
        all_vibes = []
        people_present_count = 0
        environment_types = []
        
        for analysis in semantic_analyses:
            # Aggregate semantic tags
            semantic_tags = analysis.get("semantic_tags", {})
            for category in aggregated_tags:
                if category in semantic_tags:
                    aggregated_tags[category].extend(semantic_tags[category])
            
            # Collect descriptions
            descriptions = analysis.get("descriptions", {})
            if descriptions.get("overall_description"):
                all_descriptions.append(descriptions["overall_description"])
            
            # Collect emotional analysis
            emotional = analysis.get("emotional_analysis", {})
            if emotional.get("overall_mood") and emotional["overall_mood"] != "neutral":
                all_emotions.append(emotional["overall_mood"])
            
            for vibe, score in emotional.get("vibe_score", {}).items():
                if score > 0.3:
                    all_vibes.append(vibe)
            
            # Count people presence
            people = analysis.get("people_analysis", {})
            if people.get("present"):
                people_present_count += 1
            
            # Collect environment types
            env = analysis.get("environment_analysis", {})
            if env.get("type") and env["type"] != "unknown":
                environment_types.append(env["type"])
        
        # Deduplicate and count occurrences
        from collections import Counter
        
        # Remove duplicates and get most common items
        for category in aggregated_tags:
            if aggregated_tags[category]:
                tag_counts = Counter(aggregated_tags[category])
                # Keep tags that appear in at least 20% of segments or top 10
                min_occurrences = max(1, len(semantic_analyses) * 0.2)
                aggregated_tags[category] = [
                    tag for tag, count in tag_counts.most_common(10)
                    if count >= min_occurrences
                ]
        
        # Determine overall characteristics
        emotion_counts = Counter(all_emotions)
        vibe_counts = Counter(all_vibes)
        env_counts = Counter(environment_types)
        
        return {
            "overall_semantic_tags": aggregated_tags,
            "video_characteristics": {
                "primary_emotion": emotion_counts.most_common(1)[0][0] if emotion_counts else "neutral",
                "dominant_vibes": [vibe for vibe, _ in vibe_counts.most_common(3)],
                "environment_type": env_counts.most_common(1)[0][0] if env_counts else "unknown",
                "people_presence_ratio": people_present_count / len(semantic_analyses) if semantic_analyses else 0,
                "scene_complexity": "high" if sum(len(tags) for tags in aggregated_tags.values()) > 20 else "moderate" if sum(len(tags) for tags in aggregated_tags.values()) > 10 else "simple"
            },
            "content_summary": {
                "description": ". ".join(all_descriptions[:3]) if all_descriptions else "No description available",
                "key_objects": aggregated_tags["objects"][:5],
                "main_activities": aggregated_tags["activities"][:3],
                "emotional_tone": emotion_counts.most_common(1)[0][0] if emotion_counts else "neutral",
                "atmosphere": vibe_counts.most_common(1)[0][0] if vibe_counts else "neutral"
            },
            "analysis_metadata": {
                "segments_analyzed": len(semantic_analyses),
                "total_tags_found": sum(len(tags) for tags in aggregated_tags.values()),
                "confidence_level": "high" if len(semantic_analyses) >= 3 else "moderate" if len(semantic_analyses) >= 2 else "low"
            }
        }
    
    def _cleanup_temp_files(self):
        """Clean up temporary files created during processing."""
        temp_dir = Path(self.config.storage.temp_dir)
        if temp_dir.exists():
            try:
                # Remove temp files older than 1 hour
                import time
                cutoff_time = time.time() - 3600
                
                for temp_file in temp_dir.glob("*"):
                    if temp_file.stat().st_mtime < cutoff_time:
                        if temp_file.is_file():
                            temp_file.unlink()
                        elif temp_file.is_dir():
                            shutil.rmtree(temp_file)
                
                self.logger.info("Temporary files cleaned up")
                
            except Exception as e:
                self.logger.warning(f"Failed to cleanup temp files: {e}")
    
    def _get_basic_tag_summary(self, semantic_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Get a basic summary of semantic tags when AI tagger is disabled."""
        tags = semantic_analysis.get("semantic_tags", {})
        
        return {
            "primary_objects": tags.get("objects", [])[:5],
            "key_concepts": tags.get("concepts", [])[:5],
            "people_present": len(tags.get("people", [])) > 0,
            "environment_type": tags.get("environment", ["unknown"])[0] if tags.get("environment") else "unknown",
            "primary_emotion": tags.get("emotions", ["neutral"])[0] if tags.get("emotions") else "neutral",
            "dominant_vibes": tags.get("vibes", [])[:3],
            "main_activities": tags.get("activities", [])[:3],
            "scene_attributes": tags.get("attributes", [])[:5],
            "complexity_score": "simple",
            "confidence_level": "low"
        }
    
    def get_metadata_summary(self) -> Dict[str, Any]:
        """Get summary of what metadata would be saved."""
        return {
            "metadata_persistence": {
                "enabled": self.save_metadata,
                "location": str(self.metadata_dir) if self.save_metadata else None,
                "individual_files": "One .json file per video with full analysis",
                "batch_files": "Comprehensive batch results and summary report",
                "file_tracking": "SHA-256 hash to detect file changes",
                "overwrite_policy": "Overwrite existing" if self.overwrite_existing else "Skip if exists"
            },
            "temp_files": {
                "cleanup_enabled": self.cleanup_temp,
                "temp_directory": self.config.storage.temp_dir,
                "description": "Temporary video processing files (cleaned after 1 hour)"
            }
        }


def create_batch_analyzer(save_metadata: bool = True,
                         metadata_dir: Optional[str] = None,
                         save_in_folder: bool = False,
                         clean_run: bool = False) -> BatchFolderAnalyzer:
    """
    Create a batch analyzer with specified metadata persistence.
    
    Args:
        save_metadata: Whether to save analysis results
        metadata_dir: Where to save metadata (default: ./analysis_results)
        save_in_folder: Whether to save metadata in the same folder as videos
        clean_run: Whether to overwrite existing results and clean temps
    
    Returns:
        Configured BatchFolderAnalyzer
    """
    return BatchFolderAnalyzer(
        save_metadata=save_metadata,
        metadata_dir=metadata_dir,
        save_in_folder=save_in_folder,
        overwrite_existing=clean_run,
        cleanup_temp=clean_run
    )