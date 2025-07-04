#!/usr/bin/env python3
"""
Unified Video System CLI
Command-line interface for ultra-fast video generation
"""

import asyncio
import argparse
import sys
from pathlib import Path
import json
from typing import Optional
import logging

# Try importing with graceful fallbacks
try:
    from rich.console import Console
    from rich.table import Table
    from rich.progress import track
    console = Console()
except ImportError:
    print("Warning: Rich library not installed. Install with: pip install rich")
    class Console:
        def print(self, *args, **kwargs):
            print(*args)
    console = Console()

# Import core components
try:
    from core.quantum_pipeline import UnifiedQuantumPipeline
except ImportError as e:
    console.print(f"[red]Error importing core components: {e}[/red]")
    console.print("Make sure you're running from the unified-video-system directory")
    sys.exit(1)


def setup_logging(verbose: bool = False):
    """Set up logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('unified_video_system.log'),
            logging.StreamHandler()
        ]
    )


async def generate_video_command(args):
    """Generate a video from script"""
    console.print(f"[cyan]Generating video from script...[/cyan]")
    
    # Initialize pipeline
    pipeline = UnifiedQuantumPipeline(args.config)
    
    # Read script
    if args.script_file:
        script_path = Path(args.script_file)
        if not script_path.exists():
            console.print(f"[red]Error: Script file not found: {script_path}[/red]")
            return
        script = script_path.read_text()
    else:
        script = args.script
    
    # Generate video
    try:
        result = await pipeline.generate_video(
            script=script,
            style=args.style,
            music_path=args.music,
            enable_beat_sync=args.beat_sync,
            output_path=args.output
        )
        
        # Display results
        if result['success']:
            console.print(f"\n[green]✓ Video generated successfully![/green]")
            console.print(f"Output: {result['output_path']}")
            console.print(f"Processing time: {result['processing_time']:.2f}s")
            
            if result['target_achieved']:
                console.print(f"[green]✓ Target time achieved (<0.7s)[/green]")
            else:
                console.print(f"[yellow]⚠ Target time not achieved (>0.7s)[/yellow]")
            
            # Show statistics
            stats = result.get('statistics', {})
            if stats:
                table = Table(title="Performance Statistics")
                table.add_column("Metric", style="cyan")
                table.add_column("Value", style="green")
                
                table.add_row("Cache Hit Rate", f"{stats.get('cache_hit_rate', 0):.1%}")
                table.add_row("Average Time", f"{stats.get('average_processing_time', 0):.2f}s")
                table.add_row("Total Videos", str(stats.get('total_videos_generated', 0)))
                
                console.print(table)
        else:
            console.print(f"[red]✗ Video generation failed[/red]")
            
    except Exception as e:
        console.print(f"[red]Error during generation: {e}[/red]")
        logging.exception("Generation error")


async def batch_generate_command(args):
    """Generate multiple videos in batch"""
    console.print(f"[cyan]Batch generating {args.count} videos...[/cyan]")
    
    # Initialize pipeline
    pipeline = UnifiedQuantumPipeline(args.config)
    
    # Read scripts from file or generate variations
    if args.scripts_file:
        scripts_path = Path(args.scripts_file)
        if not scripts_path.exists():
            console.print(f"[red]Error: Scripts file not found: {scripts_path}[/red]")
            return
        
        with open(scripts_path, 'r') as f:
            scripts = [line.strip() for line in f if line.strip()]
    else:
        # Generate variations of base script
        base_script = args.script or "Like water flowing through ancient stones, consciousness emerges."
        scripts = [f"{base_script} - Variation {i+1}" for i in range(args.count)]
    
    # Generate videos
    success_count = 0
    failed_count = 0
    
    for i, script in enumerate(track(scripts[:args.count], description="Generating videos")):
        try:
            output_path = f"output/batch_video_{i+1:03d}.mp4"
            result = await pipeline.generate_video(
                script=script,
                style=args.style,
                music_path=args.music,
                enable_beat_sync=args.beat_sync,
                output_path=output_path
            )
            
            if result['success']:
                success_count += 1
            else:
                failed_count += 1
                
        except Exception as e:
            console.print(f"[red]Error on video {i+1}: {e}[/red]")
            failed_count += 1
    
    # Summary
    console.print(f"\n[cyan]Batch generation complete![/cyan]")
    console.print(f"[green]✓ Success: {success_count}[/green]")
    console.print(f"[red]✗ Failed: {failed_count}[/red]")
    
    # Performance report
    report = pipeline.get_performance_report()
    if report:
        console.print(f"\nAverage processing time: {report['statistics']['average_time']:.2f}s")
        console.print(f"Cache hit rate: {report['statistics']['cache_hits'] / max(report['statistics']['cache_hits'] + report['statistics']['cache_misses'], 1):.1%}")


def test_command(args):
    """Run system tests"""
    console.print("[cyan]Running system tests...[/cyan]")
    
    tests = []
    
    # Test 1: Import test
    try:
        from core.quantum_pipeline import UnifiedQuantumPipeline
        from core.neural_cache import NeuralPredictiveCache
        from core.gpu_engine import GPUEngine
        from core.zero_copy_engine import ZeroCopyVideoEngine
        tests.append(("Import modules", True, ""))
    except Exception as e:
        tests.append(("Import modules", False, str(e)))
    
    # Test 2: GPU availability
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        device_name = torch.cuda.get_device_name(0) if gpu_available else "CPU only"
        tests.append(("GPU availability", True, device_name))
    except Exception as e:
        tests.append(("GPU availability", False, str(e)))
    
    # Test 3: Configuration
    try:
        config_path = Path(args.config)
        config_exists = config_path.exists()
        tests.append(("Configuration file", config_exists, 
                     "Found" if config_exists else "Using defaults"))
    except Exception as e:
        tests.append(("Configuration file", False, str(e)))
    
    # Test 4: Dependencies
    required_libs = [
        ('torch', 'torch'), 
        ('numpy', 'numpy'), 
        ('opencv-python', 'cv2'), 
        ('moviepy', 'moviepy'), 
        ('pyyaml', 'yaml')
    ]
    missing_libs = []
    
    for lib_name, import_name in required_libs:
        try:
            __import__(import_name)
        except ImportError:
            missing_libs.append(lib_name)
    
    tests.append(("Required libraries", 
                 len(missing_libs) == 0,
                 "All installed" if not missing_libs else f"Missing: {', '.join(missing_libs)}"))
    
    # Display results
    table = Table(title="System Test Results")
    table.add_column("Test", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Details", style="yellow")
    
    all_passed = True
    for test_name, passed, details in tests:
        status = "[green]✓ PASS[/green]" if passed else "[red]✗ FAIL[/red]"
        table.add_row(test_name, status, details)
        if not passed:
            all_passed = False
    
    console.print(table)
    
    if all_passed:
        console.print("\n[green]All tests passed! System is ready.[/green]")
    else:
        console.print("\n[red]Some tests failed. Please check the errors above.[/red]")
    
    return 0 if all_passed else 1


async def real_content_command(args):
    """Generate video with real content"""
    console.print(f"[cyan]Generating real content video from {args.script_path}...[/cyan]")
    
    try:
        # Import real content generator
        from core.real_content_generator import RealContentGenerator, RealVideoRequest
        
        # Initialize generator
        generator = RealContentGenerator(
            clips_directory="../MJAnime",
            metadata_file="../MJAnime/metadata_final_clean_shots.json", 
            scripts_directory="../11-scripts-for-tiktok",
            music_file="music/Beanie (Slowed).mp3",
            output_directory="output"
        )
        
        # Initialize
        init_success = await generator.initialize()
        if not init_success:
            console.print("[red]Failed to initialize real content generator[/red]")
            return
        
        console.print("[green]Real content generator initialized[/green]")
        
        # Generate variations
        results = []
        for variation in range(1, args.variations + 1):
            console.print(f"\n[cyan]Generating variation {variation}/{args.variations}...[/cyan]")
            
            # Create request
            script_path = Path(args.script_path)
            script_name = script_path.stem
            
            request = RealVideoRequest(
                script_path=args.script_path,
                script_name=script_name,
                variation_number=variation,
                caption_style=args.style,
                music_sync=not args.no_music,
                target_duration=None,  # Always use actual audio duration
                min_clip_duration=args.min_clip_duration
            )
            
            # Generate video
            result = await generator.generate_video(request)
            results.append(result)
            
            if result.success:
                output_path = Path(result.output_path)
                file_size_mb = output_path.stat().st_size / (1024 * 1024) if output_path.exists() else 0
                console.print(f"[green]✓ Generated: {output_path.name} ({file_size_mb:.1f}MB, {result.generation_time:.2f}s)[/green]")
            else:
                console.print(f"[red]✗ Failed: {result.error_message}[/red]")
        
        # Summary
        successful = sum(1 for r in results if r.success)
        console.print(f"\n[bold]Summary:[/bold]")
        console.print(f"Generated: {successful}/{len(results)} videos")
        
        if successful > 0:
            avg_time = sum(r.generation_time for r in results if r.success) / successful
            console.print(f"Average generation time: {avg_time:.2f}s")
            
            # Show output files
            console.print("\n[bold]Output files:[/bold]")
            for result in results:
                if result.success:
                    console.print(f"  • {Path(result.output_path).name}")
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


async def batch_real_content_command(args):
    """Batch process real content"""
    console.print(f"[cyan]Starting batch processing...[/cyan]")
    
    try:
        # Import required components
        from core.real_content_generator import RealContentGenerator
        from pipelines.content_pipeline import ContentPipeline
        
        # Initialize generator
        console.print("Initializing real content generator...")
        generator = RealContentGenerator(
            clips_directory="../MJAnime",
            metadata_file="../MJAnime/metadata_final_clean_shots.json", 
            scripts_directory=args.scripts_dir,
            music_file="music/Beanie (Slowed).mp3",
            output_directory="output"
        )
        
        init_success = await generator.initialize()
        if not init_success:
            console.print("[red]Failed to initialize generator[/red]")
            return
        
        console.print("[green]Generator initialized[/green]")
        
        # Initialize batch pipeline
        pipeline = ContentPipeline(
            content_generator=generator,
            max_concurrent_videos=args.concurrent,
            performance_monitoring=True
        )
        
        # Process scripts
        if args.scripts:
            # Process specific scripts
            console.print(f"Processing specific scripts: {args.scripts}")
            result = await pipeline.process_specific_scripts(
                script_names=args.scripts,
                scripts_directory=args.scripts_dir,
                variations_per_script=args.variations,
                caption_style=args.style
            )
        else:
            # Process all scripts
            console.print("Processing all available scripts...")
            result = await pipeline.process_all_scripts(
                scripts_directory=args.scripts_dir,
                variations_per_script=args.variations,
                caption_style=args.style
            )
        
        # Save report
        pipeline.save_batch_report(result, args.report)
        
        # Display results
        console.print(f"\n[bold]Batch Processing Results:[/bold]")
        console.print(f"Total videos: {result.total_videos}")
        console.print(f"Successful: {result.successful_videos}")
        console.print(f"Failed: {result.failed_videos}")
        console.print(f"Total time: {result.total_time:.2f}s")
        console.print(f"Average per video: {result.average_time_per_video:.2f}s")
        
        if result.performance_stats:
            console.print(f"Generation rate: {result.performance_stats.get('generation_rate_videos_per_second', 0):.2f} videos/second")
            console.print(f"Success rate: {result.performance_stats.get('success_rate', 0):.1%}")
        
        console.print(f"\nReport saved: {args.report}")
        
        if result.success:
            console.print("[green]✓ Batch processing completed successfully![/green]")
        else:
            console.print("[red]✗ Batch processing completed with errors[/red]")
        
        # Cleanup
        await pipeline.cleanup()
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


def status_command(args):
    """Show system status and statistics"""
    console.print("[cyan]System Status[/cyan]\n")
    
    try:
        # Initialize pipeline to get stats
        pipeline = UnifiedQuantumPipeline(args.config)
        
        # Get performance report
        report = pipeline.get_performance_report()
        
        # Device info
        console.print("[bold]Device Information:[/bold]")
        console.print(f"Device: {report['device_info']['device']}")
        console.print(f"GPU Available: {report['device_info']['gpu_available']}")
        if report['device_info']['gpu_name']:
            console.print(f"GPU Name: {report['device_info']['gpu_name']}")
        
        # Configuration
        console.print("\n[bold]Configuration:[/bold]")
        console.print(f"Target Processing Time: {report['config']['target_time']}s")
        console.print(f"Quantum Mode: {report['config']['quantum_mode']}")
        console.print(f"Zero-Copy Enabled: {report['config']['zero_copy']}")
        
        # Statistics
        stats = report['statistics']
        console.print("\n[bold]Performance Statistics:[/bold]")
        console.print(f"Total Videos Generated: {stats['total_videos']}")
        console.print(f"Average Processing Time: {stats['average_time']:.2f}s")
        console.print(f"Cache Hits: {stats['cache_hits']}")
        console.print(f"Cache Misses: {stats['cache_misses']}")
        
        # Cache details
        if hasattr(pipeline, 'neural_cache') and pipeline.neural_cache:
            cache_stats = pipeline.neural_cache.get_cache_stats()
            console.print("\n[bold]Cache Statistics:[/bold]")
            console.print(f"Total Entries: {cache_stats['total_entries']}")
            console.print(f"Cache Size: {cache_stats['total_size_mb']:.1f} MB")
            console.print(f"Hit Rate: {cache_stats['hit_rate']:.1%}")
            
    except Exception as e:
        console.print(f"[red]Error getting status: {e}[/red]")


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Unified Video System - Ultra-fast video generation with perfect sync"
    )
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('-c', '--config', default='config/system_config.yaml',
                       help='Path to configuration file')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Generate command (original synthetic)
    generate_parser = subparsers.add_parser('generate', help='Generate a single video')
    
    # Real content generation commands
    real_parser = subparsers.add_parser('real', help='Generate video with real content')
    real_parser.add_argument('script_path', help='Path to audio script file')
    real_parser.add_argument('-v', '--variations', type=int, default=1,
                           help='Number of variations to generate (default: 1)')
    real_parser.add_argument('-s', '--style', default='tiktok',
                           help='Caption style (default: tiktok)')
    # Duration parameter removed - system now uses actual audio duration
    # real_parser.add_argument('-d', '--duration', type=float, default=15.0,
    #                        help='Target duration in seconds (default: 15.0)')
    real_parser.add_argument('--min-clip-duration', type=float, default=2.5,
                           help='Minimum clip duration in seconds (groups beats if needed, default: 2.5)')
    real_parser.add_argument('--no-music', action='store_true',
                           help='Disable background music')
    
    # Batch real content processing commands
    batch_real_parser = subparsers.add_parser('batch-real', help='Batch process real content')
    batch_real_parser.add_argument('-v', '--variations', type=int, default=5,
                            help='Number of variations per script (default: 5)')
    batch_real_parser.add_argument('-s', '--style', default='tiktok',
                            help='Caption style (default: tiktok)')
    batch_real_parser.add_argument('--scripts-dir', default='../11-scripts-for-tiktok',
                            help='Directory containing script files')
    batch_real_parser.add_argument('--scripts', nargs='+',
                            help='Specific script names to process (without extension)')
    batch_real_parser.add_argument('--concurrent', type=int, default=3,
                            help='Maximum concurrent video generations (default: 3)')
    batch_real_parser.add_argument('--report', default='batch_report.json',
                            help='Output file for batch report (default: batch_report.json)')
    generate_parser.add_argument('script', nargs='?', help='Script text')
    generate_parser.add_argument('-f', '--script-file', help='Script file path')
    generate_parser.add_argument('-s', '--style', default='default',
                               help='Caption style preset')
    generate_parser.add_argument('-m', '--music', help='Music file for beat sync')
    generate_parser.add_argument('-b', '--beat-sync', action='store_true',
                               help='Enable beat synchronization')
    generate_parser.add_argument('-o', '--output', help='Output video path')
    
    # Batch command
    batch_parser = subparsers.add_parser('batch', help='Generate multiple videos')
    batch_parser.add_argument('count', type=int, help='Number of videos to generate')
    batch_parser.add_argument('-f', '--scripts-file', help='File with scripts (one per line)')
    batch_parser.add_argument('-s', '--script', help='Base script for variations')
    batch_parser.add_argument('--style', default='default', help='Caption style')
    batch_parser.add_argument('-m', '--music', help='Music file for all videos')
    batch_parser.add_argument('-b', '--beat-sync', action='store_true',
                            help='Enable beat synchronization')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Run system tests')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show system status')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Execute command
    if args.command == 'generate':
        asyncio.run(generate_video_command(args))
    elif args.command == 'real':
        asyncio.run(real_content_command(args))
    elif args.command == 'batch':
        # Original batch command
        asyncio.run(batch_generate_command(args))
    elif args.command == 'batch-real':
        # New batch real content command
        asyncio.run(batch_real_content_command(args))
    elif args.command == 'test':
        sys.exit(test_command(args))
    elif args.command == 'status':
        status_command(args)
    else:
        parser.print_help()
        console.print("\n[yellow]Tip: Run 'python main.py test' to check system setup[/yellow]")
        console.print("[cyan]New commands:[/cyan]")
        console.print("  [green]real[/green] - Generate video with real MJAnime clips")
        console.print("  [green]batch-real[/green] - Batch process multiple scripts with variations")


if __name__ == "__main__":
    main() 