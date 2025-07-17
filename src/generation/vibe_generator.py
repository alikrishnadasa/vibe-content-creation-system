#!/usr/bin/env python3
"""
Vibe Content Creation - Master Generator
Unified interface combining Quantum Pipeline + Enhanced Semantic System

This is the consolidated entry point for all video generation functionality.
"""

import asyncio
import argparse
import logging
import sys
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import time

# Rich console for better output
try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
    from rich.table import Table
    console = Console()
except ImportError:
    print("Warning: Rich library not installed. Install with: pip install rich")
    class Console:
        def print(self, *args, **kwargs):
            print(*args)
    console = Console()

# Setup logging
def setup_logging(verbose: bool = False):
    """Set up logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('vibe_content_creation.log'),
            logging.StreamHandler()
        ]
    )

logger = logging.getLogger(__name__)

@dataclass
class GenerationConfig:
    """Configuration for video generation"""
    # Core settings
    script_name: str
    variation_number: int = 1
    caption_style: str = "tiktok"
    
    # Generation mode
    use_enhanced_system: bool = True
    use_quantum_pipeline: bool = True
    
    # Output settings
    output_directory: str = "output"
    burn_in_captions: bool = True
    
    # Music settings
    music_sync: bool = True
    min_clip_duration: float = 2.5
    
    # Batch settings
    batch_count: int = 1
    scripts_list: Optional[List[str]] = None

class VibeGenerator:
    """Master generator that orchestrates all video generation systems"""
    
    def __init__(self, config: GenerationConfig):
        self.config = config
        self.quantum_pipeline = None
        self.enhanced_system = None
        self.real_content_generator = None
        
        # Performance tracking
        self.generation_stats = {
            'total_videos': 0,
            'successful_videos': 0,
            'failed_videos': 0,
            'average_time': 0.0,
            'total_time': 0.0
        }
    
    async def initialize(self) -> bool:
        """Initialize all systems based on configuration"""
        try:
            console.print("[cyan]Initializing Vibe Content Creation System...[/cyan]")
            
            # Add system paths
            sys.path.append('unified-video-system-main')
            
            # Initialize Enhanced System if requested
            if self.config.use_enhanced_system:
                await self._initialize_enhanced_system()
            
            # Initialize Quantum Pipeline if requested
            if self.config.use_quantum_pipeline:
                await self._initialize_quantum_pipeline()
            
            # Initialize Real Content Generator (always needed)
            await self._initialize_real_content_generator()
            
            console.print("[green]✓ All systems initialized successfully[/green]")
            return True
            
        except Exception as e:
            console.print(f"[red]✗ Initialization failed: {e}[/red]")
            logger.exception("System initialization failed")
            return False
    
    async def _initialize_enhanced_system(self):
        """Initialize Enhanced Semantic System"""
        try:
            from enhanced_script_analyzer import EnhancedAudioScriptAnalyzer
            from enhanced_content_selector import EnhancedContentSelector
            from enhanced_metadata_creator import EnhancedMetadataCreator
            
            # Check if enhanced metadata exists
            enhanced_metadata_path = Path("unified_enhanced_metadata.json")
            if not enhanced_metadata_path.exists():
                console.print("[yellow]Enhanced metadata not found, creating...[/yellow]")
                await self._create_enhanced_metadata()
            
            console.print("[green]✓ Enhanced Semantic System ready[/green]")
            
        except ImportError as e:
            console.print(f"[yellow]⚠ Enhanced system components not available: {e}[/yellow]")
            self.config.use_enhanced_system = False
    
    async def _initialize_quantum_pipeline(self):
        """Initialize Quantum Pipeline"""
        try:
            from core.quantum_pipeline import UnifiedQuantumPipeline
            
            self.quantum_pipeline = UnifiedQuantumPipeline()
            
            # Initialize real content mode
            success = await self.quantum_pipeline.initialize_real_content_mode(
                clips_directory="MJAnime",
                metadata_file="unified_enhanced_metadata.json" if self.config.use_enhanced_system else "unified_clips_metadata.json",
                scripts_directory="11-scripts-for-tiktok",
                music_file="unified-video-system-main/music/Beanie (Slowed).mp3"
            )
            
            if not success:
                console.print("[yellow]⚠ Quantum Pipeline real content mode initialization failed[/yellow]")
                self.quantum_pipeline = None
                self.config.use_quantum_pipeline = False
            else:
                console.print("[green]✓ Quantum Pipeline ready[/green]")
                
        except ImportError as e:
            console.print(f"[yellow]⚠ Quantum Pipeline not available: {e}[/yellow]")
            self.config.use_quantum_pipeline = False
    
    async def _initialize_real_content_generator(self):
        """Initialize Real Content Generator (fallback)"""
        try:
            from core.real_content_generator import RealContentGenerator
            
            self.real_content_generator = RealContentGenerator(
                clips_directory="MJAnime",
                metadata_file="unified_enhanced_metadata.json" if self.config.use_enhanced_system else "unified_clips_metadata.json",
                scripts_directory="11-scripts-for-tiktok",
                music_file="unified-video-system-main/music/Beanie (Slowed).mp3",
                output_directory=self.config.output_directory
            )
            
            success = await self.real_content_generator.initialize()
            if success:
                console.print("[green]✓ Real Content Generator ready[/green]")
            else:
                raise RuntimeError("Real Content Generator initialization failed")
                
        except Exception as e:
            console.print(f"[red]✗ Real Content Generator failed: {e}[/red]")
            raise
    
    async def _create_enhanced_metadata(self):
        """Create enhanced metadata if it doesn't exist"""
        try:
            from enhanced_metadata_creator import EnhancedMetadataCreator
            
            metadata_creator = EnhancedMetadataCreator()
            
            # Create enhanced metadata for MJAnime
            mjanime_dir = Path("MJAnime")
            if mjanime_dir.exists():
                success = metadata_creator.create_enhanced_metadata(
                    clips_directory=mjanime_dir,
                    output_file=Path("enhanced_mjanime_metadata.json"),
                    force_recreate=False,
                    use_visual_analysis=True
                )
                
                if success:
                    console.print("[green]✓ Enhanced metadata created[/green]")
                    
        except Exception as e:
            console.print(f"[yellow]⚠ Enhanced metadata creation failed: {e}[/yellow]")
    
    async def generate_video(self, script_name: str, variation: int = 1) -> Dict[str, Any]:
        """Generate a single video using the best available system"""
        start_time = time.time()
        
        try:
            # Prepare generation request
            script_path = f"11-scripts-for-tiktok/{script_name}.wav"
            
            # Try Quantum Pipeline first (most advanced)
            if self.quantum_pipeline:
                console.print(f"[cyan]Generating with Quantum Pipeline: {script_name} v{variation}[/cyan]")
                
                result = await self.quantum_pipeline.generate_real_content_video(
                    script_name=script_name,
                    variation_number=variation,
                    caption_style=self.config.caption_style
                )
                
                if result['success']:
                    return self._format_result(result, time.time() - start_time, "Quantum Pipeline")
            
            # Fallback to Real Content Generator
            if self.real_content_generator:
                console.print(f"[cyan]Generating with Real Content Generator: {script_name} v{variation}[/cyan]")
                
                from core.real_content_generator import RealVideoRequest
                
                request = RealVideoRequest(
                    script_path=script_path,
                    script_name=script_name,
                    variation_number=variation,
                    caption_style=self.config.caption_style,
                    music_sync=self.config.music_sync,
                    min_clip_duration=self.config.min_clip_duration,
                    burn_in_captions=self.config.burn_in_captions
                )
                
                result = await self.real_content_generator.generate_video(request)
                
                if result.success:
                    return self._format_result({
                        'success': True,
                        'output_path': result.output_path,
                        'processing_time': result.generation_time,
                        'real_content_data': {
                            'clips_used': result.clips_used,
                            'relevance_score': result.relevance_score,
                            'visual_variety_score': result.visual_variety_score
                        }
                    }, time.time() - start_time, "Real Content Generator")
                else:
                    return self._format_result({
                        'success': False,
                        'error': result.error_message
                    }, time.time() - start_time, "Real Content Generator")
            
            return {
                'success': False,
                'error': 'No generation system available',
                'processing_time': time.time() - start_time
            }
            
        except Exception as e:
            logger.exception(f"Video generation failed for {script_name}")
            return {
                'success': False,
                'error': str(e),
                'processing_time': time.time() - start_time
            }
    
    def _format_result(self, result: Dict[str, Any], total_time: float, system_used: str) -> Dict[str, Any]:
        """Format generation result with consistent structure"""
        return {
            'success': result.get('success', False),
            'output_path': result.get('output_path', ''),
            'processing_time': total_time,
            'system_used': system_used,
            'generation_data': result.get('real_content_data', {}),
            'error': result.get('error', None)
        }
    
    async def generate_batch(self, scripts: List[str], variations: int = 1) -> List[Dict[str, Any]]:
        """Generate multiple videos in batch"""
        console.print(f"[cyan]Starting batch generation: {len(scripts)} scripts × {variations} variations[/cyan]")
        
        batch_start_time = time.time()
        results = []
        
        for script_name in scripts:
            for variation in range(1, variations + 1):
                result = await self.generate_video(script_name, variation)
                results.append({
                    'script': script_name,
                    'variation': variation,
                    **result
                })
                
                # Update stats
                self.generation_stats['total_videos'] += 1
                if result['success']:
                    self.generation_stats['successful_videos'] += 1
                    console.print(f"[green]✓ {script_name} v{variation}: {Path(result['output_path']).name}[/green]")
                else:
                    self.generation_stats['failed_videos'] += 1
                    console.print(f"[red]✗ {script_name} v{variation}: {result['error']}[/red]")
        
        # Calculate batch statistics
        batch_time = time.time() - batch_start_time
        self.generation_stats['total_time'] = batch_time
        self.generation_stats['average_time'] = batch_time / len(results) if results else 0
        
        # Display summary
        self._display_batch_summary(results, batch_time)
        
        return results
    
    def _display_batch_summary(self, results: List[Dict[str, Any]], batch_time: float):
        """Display batch generation summary"""
        successful = self.generation_stats['successful_videos']
        failed = self.generation_stats['failed_videos']
        total = self.generation_stats['total_videos']
        
        table = Table(title="Batch Generation Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Total Videos", str(total))
        table.add_row("Successful", str(successful))
        table.add_row("Failed", str(failed))
        table.add_row("Success Rate", f"{successful/total*100:.1f}%" if total > 0 else "0%")
        table.add_row("Total Time", f"{batch_time:.2f}s")
        table.add_row("Average Time", f"{self.generation_stats['average_time']:.2f}s")
        
        console.print(table)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        return {
            'enhanced_system_available': self.config.use_enhanced_system,
            'quantum_pipeline_available': self.config.use_quantum_pipeline,
            'real_content_generator_available': self.real_content_generator is not None,
            'generation_stats': self.generation_stats,
            'config': {
                'caption_style': self.config.caption_style,
                'music_sync': self.config.music_sync,
                'output_directory': self.config.output_directory
            }
        }

async def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(
        description="Vibe Content Creation - Master Video Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s single anxiety1                    # Generate single video
  %(prog)s single anxiety1 -v 3              # Generate 3 variations
  %(prog)s batch anxiety1 safe1 phone1.      # Generate batch
  %(prog)s batch anxiety1 safe1 -v 2         # Generate batch with variations
  %(prog)s status                            # Show system status
        """
    )
    
    parser.add_argument('-V', '--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('-s', '--style', default='tiktok', help='Caption style (default: tiktok)')
    parser.add_argument('-o', '--output', default='output', help='Output directory (default: output)')
    parser.add_argument('--no-enhanced', action='store_true', help='Disable enhanced semantic system')
    parser.add_argument('--no-quantum', action='store_true', help='Disable quantum pipeline')
    parser.add_argument('--no-music', action='store_true', help='Disable music sync')
    parser.add_argument('--no-captions', action='store_true', help='Disable burned-in captions')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Single video generation
    single_parser = subparsers.add_parser('single', help='Generate a single video')
    single_parser.add_argument('script', help='Script name (without .wav extension)')
    single_parser.add_argument('-v', '--variations', type=int, default=1, help='Number of variations (default: 1)')
    
    # Batch generation
    batch_parser = subparsers.add_parser('batch', help='Generate multiple videos')
    batch_parser.add_argument('scripts', nargs='+', help='Script names (without .wav extension)')
    batch_parser.add_argument('-v', '--variations', type=int, default=1, help='Number of variations per script (default: 1)')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show system status')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Create configuration
    config = GenerationConfig(
        script_name=getattr(args, 'script', ''),
        caption_style=args.style,
        output_directory=args.output,
        use_enhanced_system=not args.no_enhanced,
        use_quantum_pipeline=not args.no_quantum,
        music_sync=not args.no_music,
        burn_in_captions=not args.no_captions
    )
    
    # Initialize generator
    generator = VibeGenerator(config)
    
    if args.command == 'status':
        # Show system status without initialization
        console.print("[cyan]System Status (Pre-initialization)[/cyan]")
        console.print(f"Enhanced System: {'Enabled' if config.use_enhanced_system else 'Disabled'}")
        console.print(f"Quantum Pipeline: {'Enabled' if config.use_quantum_pipeline else 'Disabled'}")
        console.print(f"Caption Style: {config.caption_style}")
        console.print(f"Output Directory: {config.output_directory}")
        return
    
    # Initialize system
    if not await generator.initialize():
        console.print("[red]System initialization failed[/red]")
        sys.exit(1)
    
    # Execute command
    if args.command == 'single':
        result = await generator.generate_video(args.script, args.variations)
        if result['success']:
            console.print(f"[green]✓ Video generated: {result['output_path']}[/green]")
        else:
            console.print(f"[red]✗ Generation failed: {result['error']}[/red]")
            sys.exit(1)
            
    elif args.command == 'batch':
        results = await generator.generate_batch(args.scripts, args.variations)
        successful = sum(1 for r in results if r['success'])
        if successful == 0:
            console.print("[red]All videos failed to generate[/red]")
            sys.exit(1)
        elif successful < len(results):
            console.print(f"[yellow]Partial success: {successful}/{len(results)} videos generated[/yellow]")
        else:
            console.print(f"[green]All {len(results)} videos generated successfully[/green]")
    
    else:
        parser.print_help()
        console.print("\n[yellow]Tip: Use 'single' or 'batch' commands to generate videos[/yellow]")

if __name__ == "__main__":
    asyncio.run(main())