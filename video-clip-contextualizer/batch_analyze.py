#!/usr/bin/env python3
"""
Batch Video Folder Analyzer CLI

Command-line interface for batch analyzing folders of video clips.
"""

import argparse
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.batch_folder_analyzer import BatchFolderAnalyzer, create_batch_analyzer


def setup_logging(level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def main():
    parser = argparse.ArgumentParser(
        description="Batch analyze video clips in a folder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze folder with metadata persistence (default)
  python batch_analyze.py /path/to/videos script.txt
  
  # No metadata persistence (clean run)
  python batch_analyze.py /path/to/videos script.txt --no-metadata
  
  # Custom metadata location
  python batch_analyze.py /path/to/videos script.txt --metadata-dir ./my_results
  
  # Overwrite existing results
  python batch_analyze.py /path/to/videos script.txt --clean-run
  
  # Custom processing settings
  python batch_analyze.py /path/to/videos script.txt --clip-duration 3.0 --overlap 1.0
  
  # Specific file pattern
  python batch_analyze.py /path/to/videos script.txt --pattern "*.mp4"
  
  # Show what metadata would be saved (dry run)
  python batch_analyze.py /path/to/videos script.txt --show-metadata
        """
    )
    
    # Required arguments
    parser.add_argument("video_folder", help="Folder containing video files")
    parser.add_argument("script", help="Path to script file or text content")
    
    # Processing options
    parser.add_argument("--clip-duration", type=float, default=5.0,
                       help="Duration of video clips in seconds (default: 5.0)")
    parser.add_argument("--overlap", type=float, default=0.5,
                       help="Overlap between clips in seconds (default: 0.5)")
    parser.add_argument("--strategy", choices=["optimal", "greedy", "threshold"],
                       default="optimal", help="Matching strategy (default: optimal)")
    parser.add_argument("--pattern", default="*",
                       help="File pattern to match (default: * for all files)")
    
    # Metadata persistence options
    parser.add_argument("--no-metadata", action="store_true",
                       help="Don't save any metadata to disk")
    parser.add_argument("--metadata-dir", 
                       help="Directory to save metadata (default: ./analysis_results)")
    parser.add_argument("--save-in-folder", action="store_true",
                       help="Save metadata files in the same folder as videos")
    parser.add_argument("--clean-run", action="store_true",
                       help="Overwrite existing results and clean temp files")
    parser.add_argument("--show-metadata", action="store_true",
                       help="Show what metadata would be saved and exit")
    
    # Output options
    parser.add_argument("--quiet", action="store_true",
                       help="Minimize output (errors only)")
    parser.add_argument("--verbose", action="store_true",
                       help="Verbose output")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       default="INFO", help="Set logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    if args.quiet:
        log_level = "ERROR"
    elif args.verbose:
        log_level = "DEBUG"
    else:
        log_level = args.log_level
    
    setup_logging(log_level)
    logger = logging.getLogger(__name__)
    
    try:
        # Validate inputs
        video_folder = Path(args.video_folder)
        if not video_folder.exists():
            print(f"Error: Video folder not found: {video_folder}", file=sys.stderr)
            sys.exit(1)
        
        # Read script
        script_path = Path(args.script)
        if script_path.exists():
            with open(script_path, 'r', encoding='utf-8') as f:
                script_text = f.read()
            logger.info(f"Loaded script from: {script_path}")
        else:
            # Treat as literal text
            script_text = args.script
            logger.info("Using provided text as script")
        
        # Create analyzer
        analyzer = create_batch_analyzer(
            save_metadata=not args.no_metadata,
            metadata_dir=args.metadata_dir,
            save_in_folder=args.save_in_folder,
            clean_run=args.clean_run
        )
        
        # Show metadata info if requested
        if args.show_metadata:
            metadata_info = analyzer.get_metadata_summary()
            print("METADATA PERSISTENCE SUMMARY")
            print("=" * 40)
            
            persistence = metadata_info["metadata_persistence"]
            print(f"Enabled: {persistence['enabled']}")
            if persistence['enabled']:
                print(f"Location: {persistence['location']}")
                print(f"Individual files: {persistence['individual_files']}")
                print(f"Batch files: {persistence['batch_files']}")
                print(f"File tracking: {persistence['file_tracking']}")
                print(f"Overwrite policy: {persistence['overwrite_policy']}")
            
            temp_info = metadata_info["temp_files"]
            print(f"\nTemporary Files:")
            print(f"Cleanup enabled: {temp_info['cleanup_enabled']}")
            print(f"Temp directory: {temp_info['temp_directory']}")
            print(f"Description: {temp_info['description']}")
            
            return
        
        # Run analysis
        print(f"Starting batch analysis...")
        print(f"Video folder: {video_folder}")
        print(f"Script: {script_path if script_path.exists() else 'inline text'}")
        print(f"Metadata persistence: {'Enabled' if not args.no_metadata else 'Disabled'}")
        
        if not args.no_metadata:
            print(f"Results will be saved to: {analyzer.metadata_dir}")
        
        results = analyzer.analyze_folder(
            video_folder=str(video_folder),
            script_text=script_text,
            clip_duration=args.clip_duration,
            overlap=args.overlap,
            matching_strategy=args.strategy,
            file_pattern=args.pattern
        )
        
        # Print summary
        batch_info = results["batch_info"]
        print(f"\n{'='*60}")
        print(f"BATCH ANALYSIS COMPLETE")
        print(f"{'='*60}")
        print(f"Total videos: {batch_info['total_videos']}")
        print(f"Processed: {batch_info['processed_videos']}")
        print(f"Skipped: {batch_info['skipped_videos']}")
        print(f"Failed: {batch_info['failed_videos']}")
        print(f"Success rate: {batch_info['success_rate']:.1%}")
        print(f"Total time: {batch_info['total_processing_time']:.1f}s")
        
        if "summary" in results and not results["summary"].get("no_data"):
            summary = results["summary"]
            print(f"\nSUMMARY STATISTICS:")
            print(f"Total matches: {summary['total_matches']}")
            print(f"Avg matches per video: {summary['average_matches_per_video']:.1f}")
            print(f"Average confidence: {summary['average_confidence']:.3f}")
            print(f"Avg processing time: {summary['average_processing_time']:.2f}s")
            
            print(f"\nBest match: {summary['best_match']['video']} ({summary['best_match']['confidence']:.3f})")
            print(f"Worst match: {summary['worst_match']['video']} ({summary['worst_match']['confidence']:.3f})")
        
        # Show where results were saved
        if not args.no_metadata:
            print(f"\nResults saved to: {analyzer.metadata_dir}")
            print(f"- Individual video analyses: *_analysis.json")
            print(f"- Batch results: batch_*.json")
            print(f"- Summary report: summary_*.txt")
        
        print(f"{'='*60}")
        
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()