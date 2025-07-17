#!/usr/bin/env python3
"""
Video Clip Contextualizer - Main Entry Point

This is the main entry point for the Video Clip Contextualizer system.
It provides both CLI and programmatic interfaces for video-text matching.
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any

from src.processing.video_processor import VideoProcessor
from src.processing.text_processor import TextProcessor
from src.matching.semantic_matcher import SemanticMatcher
from src.monitoring.performance_monitor import PerformanceMonitor
from src.config import get_config


def setup_logging(level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('video_contextualizer.log')
        ]
    )


def analyze_video_cli(
    video_path: str,
    script_path: str,
    clip_duration: float = 5.0,
    overlap: float = 0.5,
    matching_strategy: str = "optimal",
    output_format: str = "json"
) -> Dict[str, Any]:
    """
    Analyze video from command line interface.
    
    Args:
        video_path: Path to video file
        script_path: Path to script file
        clip_duration: Duration of video clips in seconds
        overlap: Overlap between clips in seconds
        matching_strategy: Matching algorithm to use
        output_format: Output format (json, summary, detailed)
    
    Returns:
        Analysis results
    """
    logger = logging.getLogger(__name__)
    
    # Validate inputs
    if not Path(video_path).exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    if not Path(script_path).exists():
        raise FileNotFoundError(f"Script file not found: {script_path}")
    
    # Read script
    with open(script_path, 'r', encoding='utf-8') as f:
        script = f.read()
    
    logger.info(f"Starting analysis of {video_path}")
    logger.info(f"Clip duration: {clip_duration}s, Overlap: {overlap}s")
    
    # Initialize processors
    video_processor = VideoProcessor(clip_duration=clip_duration, overlap=overlap)
    text_processor = TextProcessor()
    matcher = SemanticMatcher()
    monitor = PerformanceMonitor()
    
    try:
        # Start performance monitoring
        request_id = monitor.start_request("cli_analysis")
        
        # Process video
        logger.info("Processing video...")
        video_segments = video_processor.process_video_file(video_path)
        logger.info(f"Created {len(video_segments)} video segments")
        
        # Process text
        logger.info("Processing script...")
        text_segments = text_processor.segment_text(script)
        text_segments = text_processor.generate_embeddings(text_segments)
        logger.info(f"Created {len(text_segments)} text segments")
        
        # Perform matching
        logger.info(f"Matching using {matching_strategy} strategy...")
        matches = matcher.match_video_to_text(
            video_segments, 
            text_segments, 
            matching_strategy
        )
        
        # Calculate metrics
        metrics = matcher.compute_matching_metrics(matches)
        
        # End performance monitoring
        monitor.end_request(
            request_id,
            success=True,
            confidence_score=metrics.get('avg_confidence', 0),
            video_segments=len(video_segments),
            text_segments=len(text_segments),
            matches_found=len(matches)
        )
        
        logger.info(f"Analysis complete: {len(matches)} matches found")
        
        # Format results
        results = {
            "video_file": video_path,
            "script_file": script_path,
            "processing_config": {
                "clip_duration": clip_duration,
                "overlap": overlap,
                "matching_strategy": matching_strategy
            },
            "metrics": metrics,
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
            "performance": monitor.get_current_stats()
        }
        
        return results
        
    except Exception as e:
        monitor.end_request(request_id, success=False)
        logger.error(f"Analysis failed: {str(e)}")
        raise
    
    finally:
        monitor.stop_monitoring()


def print_results(results: Dict[str, Any], format: str = "summary"):
    """Print analysis results in specified format."""
    
    if format == "json":
        import json
        print(json.dumps(results, indent=2))
        return
    
    # Summary format
    print(f"\n{'='*60}")
    print(f"VIDEO CLIP CONTEXTUALIZER - ANALYSIS RESULTS")
    print(f"{'='*60}")
    
    print(f"\nInput Files:")
    print(f"  Video: {results['video_file']}")
    print(f"  Script: {results['script_file']}")
    
    config = results['processing_config']
    print(f"\nProcessing Configuration:")
    print(f"  Clip duration: {config['clip_duration']}s")
    print(f"  Overlap: {config['overlap']}s")
    print(f"  Matching strategy: {config['matching_strategy']}")
    
    metrics = results['metrics']
    print(f"\nMatching Metrics:")
    print(f"  Total matches: {metrics['total_matches']}")
    print(f"  Average confidence: {metrics['avg_confidence']:.3f}")
    print(f"  High confidence ratio: {metrics['high_confidence_ratio']:.1%}")
    print(f"  Min confidence: {metrics['min_confidence']:.3f}")
    print(f"  Max confidence: {metrics['max_confidence']:.3f}")
    
    if format == "detailed":
        print(f"\nDetailed Matches:")
        for i, match in enumerate(results['matches']):
            print(f"\n  Match {i+1}:")
            print(f"    Video: {match['video_segment']['start']:.1f}s - {match['video_segment']['end']:.1f}s")
            print(f"    Text: \"{match['text_segment']['text'][:100]}{'...' if len(match['text_segment']['text']) > 100 else ''}\"")
            print(f"    Confidence: {match['confidence']:.3f}")
            print(f"    Match type: {match['explanation']['match_type']}")
            
            if 'keywords' in match['explanation']:
                keywords = match['explanation']['keywords'][:5]
                print(f"    Keywords: {', '.join(keywords)}")
    
    perf = results['performance']['system_stats']
    print(f"\nPerformance:")
    print(f"  Processing time: {perf['avg_processing_time']:.3f}s")
    print(f"  Success rate: {perf['successful_requests'] / max(perf['total_requests'], 1):.1%}")
    
    print(f"\n{'='*60}")


def run_server(host: str = "0.0.0.0", port: int = 8000, workers: int = 1):
    """Run the FastAPI server."""
    import uvicorn
    from src.api.main import app
    
    print(f"Starting Video Clip Contextualizer API server...")
    print(f"Server will be available at: http://{host}:{port}")
    print(f"API documentation: http://{host}:{port}/docs")
    
    uvicorn.run(
        "src.api.main:app",
        host=host,
        port=port,
        workers=workers,
        reload=False
    )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Video Clip Contextualizer - AI-powered video-to-script matching",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a video with script
  python main.py analyze video.mp4 script.txt

  # Custom clip duration and overlap
  python main.py analyze video.mp4 script.txt --clip-duration 3.0 --overlap 0.5

  # Use different matching strategy
  python main.py analyze video.mp4 script.txt --strategy greedy

  # Get detailed output
  python main.py analyze video.mp4 script.txt --format detailed

  # Start API server
  python main.py server --host 0.0.0.0 --port 8000

  # Start server with multiple workers
  python main.py server --workers 4
        """
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set logging level"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze video with script")
    analyze_parser.add_argument("video", help="Path to video file")
    analyze_parser.add_argument("script", help="Path to script file")
    analyze_parser.add_argument(
        "--clip-duration", 
        type=float, 
        default=5.0,
        help="Duration of video clips in seconds (default: 5.0)"
    )
    analyze_parser.add_argument(
        "--overlap", 
        type=float, 
        default=0.5,
        help="Overlap between clips in seconds (default: 0.5)"
    )
    analyze_parser.add_argument(
        "--strategy",
        choices=["optimal", "greedy", "threshold"],
        default="optimal",
        help="Matching strategy (default: optimal)"
    )
    analyze_parser.add_argument(
        "--format",
        choices=["json", "summary", "detailed"],
        default="summary",
        help="Output format (default: summary)"
    )
    analyze_parser.add_argument(
        "--output",
        help="Save results to file"
    )
    
    # Server command
    server_parser = subparsers.add_parser("server", help="Start API server")
    server_parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind server (default: 0.0.0.0)"
    )
    server_parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind server (default: 8000)"
    )
    server_parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes (default: 1)"
    )
    
    # Config command
    config_parser = subparsers.add_parser("config", help="Show configuration")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    if args.command == "analyze":
        try:
            results = analyze_video_cli(
                args.video,
                args.script,
                args.clip_duration,
                args.overlap,
                args.strategy,
                args.format
            )
            
            if args.output:
                import json
                with open(args.output, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"Results saved to: {args.output}")
            else:
                print_results(results, args.format)
                
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
    
    elif args.command == "server":
        try:
            run_server(args.host, args.port, args.workers)
        except KeyboardInterrupt:
            print("\nServer stopped by user")
        except Exception as e:
            print(f"Server error: {e}", file=sys.stderr)
            sys.exit(1)
    
    elif args.command == "config":
        config = get_config()
        print("Current Configuration:")
        print(f"  Clip duration: {config.clip_duration.min}-{config.clip_duration.max}s (default: {config.clip_duration.default}s)")
        print(f"  Overlap: {config.clip_duration.overlap}s")
        print(f"  Device: {config.processing.device}")
        print(f"  Precision: {config.processing.precision}")
        print(f"  Batch size: {config.processing.batch_size}")
        print(f"  Video encoder: {config.models.video_encoder}")
        print(f"  Text encoder: {config.models.text_encoder}")
        print(f"  BLIP-2 model: {config.models.blip2_model}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()