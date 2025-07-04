#!/usr/bin/env python3
"""
Performance Benchmark Suite for Unified Video System
Tests and validates the 0.7s target achievement
"""

import asyncio
import sys
import time
import statistics
from pathlib import Path
from typing import List, Dict, Any
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core.quantum_pipeline import UnifiedQuantumPipeline

console = Console()


class PerformanceBenchmark:
    """
    Comprehensive performance benchmark for the video generation system.
    """
    
    def __init__(self):
        self.pipeline = None
        self.results = []
        
    async def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """Run complete performance benchmark suite"""
        console.print(Panel.fit(
            "[bold green]Unified Video System - Performance Benchmark[/bold green]\n"
            "Target: Generate videos in <0.7 seconds",
            border_style="green"
        ))
        
        # Initialize pipeline
        console.print("\n[cyan]Initializing system...[/cyan]")
        self.pipeline = UnifiedQuantumPipeline()
        
        # Warm up system
        await self._warmup_system()
        
        # Run benchmark tests
        tests = [
            ("Basic Text Video", self._test_basic_text_video),
            ("Multi-Caption Video", self._test_multi_caption_video),
            ("Beat-Sync Video", self._test_beat_sync_video),
            ("Complex Style Video", self._test_complex_style_video),
            ("Long Form Video", self._test_long_form_video)
        ]
        
        console.print(f"\n[cyan]Running {len(tests)} benchmark tests...[/cyan]")
        
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=console
        ) as progress:
            
            main_task = progress.add_task("Running benchmarks...", total=len(tests))
            
            for test_name, test_func in tests:
                progress.update(main_task, description=f"Testing: {test_name}")
                
                # Run test multiple times for statistical significance
                times = []
                for i in range(5):  # 5 runs per test
                    start_time = time.perf_counter()
                    result = await test_func()
                    end_time = time.perf_counter()
                    
                    generation_time = end_time - start_time
                    times.append(generation_time)
                    
                    if not result['success']:
                        console.print(f"[red]Test {test_name} failed![/red]")
                        break
                
                # Calculate statistics
                if times:
                    avg_time = statistics.mean(times)
                    min_time = min(times)
                    max_time = max(times)
                    std_dev = statistics.stdev(times) if len(times) > 1 else 0
                    
                    self.results.append({
                        'test_name': test_name,
                        'times': times,
                        'avg_time': avg_time,
                        'min_time': min_time,
                        'max_time': max_time,
                        'std_dev': std_dev,
                        'target_achieved': avg_time <= 0.7,
                        'success_rate': sum(1 for t in times if t <= 0.7) / len(times)
                    })
                
                progress.advance(main_task)
        
        # Generate report
        return self._generate_performance_report()
    
    async def _warmup_system(self):
        """Warm up the system for accurate benchmarking"""
        console.print("[yellow]Warming up system...[/yellow]")
        
        # Warm up GPU
        if self.pipeline.gpu_engine:
            self.pipeline.gpu_engine.warmup()
        
        # Warm up with a simple generation
        try:
            await self.pipeline.generate_video(
                script="Warmup test",
                style="default"
            )
        except:
            pass  # Ignore warmup failures
        
        console.print("[green]✓ System warmed up[/green]")
    
    async def _test_basic_text_video(self) -> Dict[str, Any]:
        """Test basic text video generation"""
        return await self.pipeline.generate_video(
            script="Hello world! This is a test video.",
            style="default"
        )
    
    async def _test_multi_caption_video(self) -> Dict[str, Any]:
        """Test video with multiple captions"""
        return await self.pipeline.generate_video(
            script="This is sentence one. This is sentence two. This is sentence three. And this is the final sentence.",
            style="tiktok"
        )
    
    async def _test_beat_sync_video(self) -> Dict[str, Any]:
        """Test beat-synchronized video generation"""
        return await self.pipeline.generate_video(
            script="Music synchronized video with perfect timing.",
            style="impact",
            enable_beat_sync=True
        )
    
    async def _test_complex_style_video(self) -> Dict[str, Any]:
        """Test video with complex styling"""
        return await self.pipeline.generate_video(
            script="Complex styled video with cinematic effects.",
            style="cinematic"
        )
    
    async def _test_long_form_video(self) -> Dict[str, Any]:
        """Test longer video generation"""
        script = " ".join([
            "This is a longer video test.",
            "It contains multiple sentences.",
            "Each sentence should be properly timed.",
            "The system should handle this efficiently.",
            "Performance should remain under 0.7 seconds."
        ])
        
        return await self.pipeline.generate_video(
            script=script,
            style="youtube"
        )
    
    def _generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        if not self.results:
            return {'error': 'No benchmark results available'}
        
        # Calculate overall statistics
        all_times = []
        for result in self.results:
            all_times.extend(result['times'])
        
        overall_stats = {
            'total_tests': len(self.results),
            'total_runs': len(all_times),
            'overall_avg': statistics.mean(all_times),
            'overall_min': min(all_times),
            'overall_max': max(all_times),
            'overall_std': statistics.stdev(all_times) if len(all_times) > 1 else 0,
            'target_achievement_rate': sum(1 for t in all_times if t <= 0.7) / len(all_times),
            'tests_under_target': sum(1 for r in self.results if r['target_achieved'])
        }
        
        # Display results
        self._display_results(overall_stats)
        
        return {
            'overall_stats': overall_stats,
            'test_results': self.results,
            'target_achieved': overall_stats['overall_avg'] <= 0.7,
            'recommendation': self._get_performance_recommendation(overall_stats)
        }
    
    def _display_results(self, overall_stats: Dict[str, Any]):
        """Display benchmark results in a formatted table"""
        console.print(f"\n{'='*80}")
        console.print("[bold]BENCHMARK RESULTS[/bold]")
        console.print(f"{'='*80}")
        
        # Test results table
        table = Table(title="Individual Test Results")
        table.add_column("Test", style="cyan", width=20)
        table.add_column("Avg Time (s)", style="yellow", justify="right")
        table.add_column("Min Time (s)", style="green", justify="right")
        table.add_column("Max Time (s)", style="red", justify="right")
        table.add_column("Std Dev", style="blue", justify="right")
        table.add_column("Target Met", style="magenta", justify="center")
        table.add_column("Success Rate", style="white", justify="right")
        
        for result in self.results:
            table.add_row(
                result['test_name'],
                f"{result['avg_time']:.3f}",
                f"{result['min_time']:.3f}",
                f"{result['max_time']:.3f}",
                f"{result['std_dev']:.3f}",
                "✅" if result['target_achieved'] else "❌",
                f"{result['success_rate']:.1%}"
            )
        
        console.print(table)
        
        # Overall statistics
        console.print(f"\n[bold]OVERALL PERFORMANCE[/bold]")
        console.print(f"Target: <0.7s per video")
        console.print(f"Average time: {overall_stats['overall_avg']:.3f}s")
        console.print(f"Best time: {overall_stats['overall_min']:.3f}s")
        console.print(f"Worst time: {overall_stats['overall_max']:.3f}s")
        console.print(f"Standard deviation: {overall_stats['overall_std']:.3f}s")
        console.print(f"Target achievement rate: {overall_stats['target_achievement_rate']:.1%}")
        console.print(f"Tests meeting target: {overall_stats['tests_under_target']}/{overall_stats['total_tests']}")
        
        # Visual indicator
        if overall_stats['overall_avg'] <= 0.7:
            console.print("\n[bold green]🎉 TARGET ACHIEVED! 🎉[/bold green]")
            console.print("[green]The system consistently generates videos under 0.7 seconds![/green]")
        else:
            console.print(f"\n[bold yellow]⚠️  TARGET MISSED[/bold yellow]")
            console.print(f"[yellow]Average time ({overall_stats['overall_avg']:.3f}s) exceeds 0.7s target[/yellow]")
            speedup_needed = ((overall_stats['overall_avg'] - 0.7) / 0.7) * 100
            console.print(f"[yellow]Need {speedup_needed:.1f}% speedup to reach target[/yellow]")
    
    def _get_performance_recommendation(self, stats: Dict[str, Any]) -> str:
        """Get performance improvement recommendations"""
        if stats['overall_avg'] <= 0.7:
            return "Excellent performance! System meets target consistently."
        
        recommendations = []
        
        if stats['overall_avg'] > 1.0:
            recommendations.append("Consider optimizing GPU operations and memory transfers")
        
        if stats['overall_std'] > 0.2:
            recommendations.append("High variance detected - optimize cache hit rates")
        
        if stats['target_achievement_rate'] < 0.5:
            recommendations.append("Less than 50% runs meet target - review parallel processing")
        
        if not recommendations:
            recommendations.append("Minor optimizations needed - fine-tune encoding settings")
        
        return "; ".join(recommendations)


async def run_stress_test():
    """Run stress test with multiple concurrent generations"""
    console.print(f"\n{'='*80}")
    console.print("[bold]STRESS TEST - Concurrent Generation[/bold]")
    console.print(f"{'='*80}")
    
    pipeline = UnifiedQuantumPipeline()
    
    # Generate multiple videos concurrently
    scripts = [
        "Concurrent test video number one",
        "Concurrent test video number two", 
        "Concurrent test video number three",
        "Concurrent test video number four",
        "Concurrent test video number five"
    ]
    
    start_time = time.perf_counter()
    
    tasks = []
    for i, script in enumerate(scripts):
        task = pipeline.generate_video(
            script=script,
            style="default",
            output_path=f"output/stress_test_{i}.mp4"
        )
        tasks.append(task)
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    end_time = time.perf_counter()
    total_time = end_time - start_time
    
    successful = sum(1 for r in results if isinstance(r, dict) and r.get('success', False))
    
    console.print(f"Generated {successful}/{len(scripts)} videos in {total_time:.3f}s")
    console.print(f"Average time per video: {total_time/len(scripts):.3f}s")
    
    if total_time/len(scripts) <= 0.7:
        console.print("[bold green]✅ Stress test passed![/bold green]")
    else:
        console.print("[bold yellow]⚠️  Stress test shows degradation under load[/bold yellow]")


async def main():
    """Main benchmark execution"""
    benchmark = PerformanceBenchmark()
    
    # Run comprehensive benchmark
    results = await benchmark.run_comprehensive_benchmark()
    
    # Run stress test
    await run_stress_test()
    
    # Final summary
    console.print(f"\n{'='*80}")
    console.print("[bold]BENCHMARK COMPLETE[/bold]")
    console.print(f"{'='*80}")
    
    if results.get('target_achieved', False):
        console.print("[bold green]🎉 SYSTEM READY FOR PRODUCTION 🎉[/bold green]")
        console.print("[green]Phase 5 optimization goals achieved![/green]")
    else:
        console.print("[bold yellow]⚠️  ADDITIONAL OPTIMIZATION NEEDED[/bold yellow]")
        console.print(f"[yellow]Recommendation: {results.get('recommendation', 'Unknown')}[/yellow]")
    
    return results


if __name__ == "__main__":
    # Run the benchmark
    try:
        results = asyncio.run(main())
        
        # Exit with appropriate code
        if results.get('target_achieved', False):
            sys.exit(0)  # Success
        else:
            sys.exit(1)  # Performance target not met
            
    except KeyboardInterrupt:
        console.print("\n[yellow]Benchmark interrupted by user[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"\n[red]Benchmark failed: {e}[/red]")
        sys.exit(1)