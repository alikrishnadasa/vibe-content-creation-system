#!/usr/bin/env python3
"""
Phase 5 Performance Test - Validate 0.7s Target Achievement
"""

import asyncio
import sys
import time
from pathlib import Path
from rich.console import Console
from rich.panel import Panel

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core.quantum_pipeline import UnifiedQuantumPipeline

console = Console()


async def test_phase5_performance():
    """Test Phase 5 performance optimizations"""
    
    console.print(Panel.fit(
        "[bold green]Phase 5 Performance Test[/bold green]\n"
        "Validating 0.7s target achievement",
        border_style="green"
    ))
    
    # Initialize pipeline
    console.print("\n[cyan]Initializing optimized pipeline...[/cyan]")
    pipeline = UnifiedQuantumPipeline()
    
    # Test scripts of varying complexity
    test_cases = [
        {
            'name': 'Simple Text',
            'script': 'Hello world!',
            'style': 'default'
        },
        {
            'name': 'Multi-sentence',
            'script': 'This is sentence one. This is sentence two. Final sentence here.',
            'style': 'tiktok'
        },
        {
            'name': 'Beat Sync',
            'script': 'Music synchronized video content.',
            'style': 'impact',
            'enable_beat_sync': True
        },
        {
            'name': 'Complex Style',
            'script': 'Advanced video with cinematic styling and effects.',
            'style': 'cinematic'
        }
    ]
    
    console.print(f"\n[cyan]Running {len(test_cases)} performance tests...[/cyan]")
    
    results = []
    total_time = 0
    
    for i, test_case in enumerate(test_cases):
        console.print(f"\nTest {i+1}: {test_case['name']}")
        
        # Run test
        start_time = time.perf_counter()
        
        try:
            result = await pipeline.generate_video(
                script=test_case['script'],
                style=test_case['style'],
                enable_beat_sync=test_case.get('enable_beat_sync', False),
                output_path=f"output/phase5_test_{i}.mp4"
            )
            
            end_time = time.perf_counter()
            generation_time = end_time - start_time
            total_time += generation_time
            
            # Check result
            success = result.get('success', False)
            target_achieved = generation_time <= 0.7
            
            results.append({
                'name': test_case['name'],
                'time': generation_time,
                'success': success,
                'target_achieved': target_achieved
            })
            
            # Display result
            status = "✅" if target_achieved else "❌"
            color = "green" if target_achieved else "yellow"
            console.print(f"  {status} Time: {generation_time:.3f}s", style=color)
            
        except Exception as e:
            console.print(f"  ❌ Failed: {e}", style="red")
            results.append({
                'name': test_case['name'],
                'time': float('inf'),
                'success': False,
                'target_achieved': False
            })
    
    # Calculate overall performance
    successful_tests = [r for r in results if r['success']]
    avg_time = total_time / len(successful_tests) if successful_tests else float('inf')
    target_achievements = sum(1 for r in results if r['target_achieved'])
    
    # Display summary
    console.print(f"\n{'='*60}")
    console.print("[bold]PHASE 5 PERFORMANCE SUMMARY[/bold]")
    console.print(f"{'='*60}")
    
    console.print(f"Tests completed: {len(results)}")
    console.print(f"Successful: {len(successful_tests)}")
    console.print(f"Target achieved: {target_achievements}/{len(results)}")
    console.print(f"Average time: {avg_time:.3f}s")
    console.print(f"Target time: 0.7s")
    
    # Overall result
    if avg_time <= 0.7 and target_achievements >= len(results) * 0.8:
        console.print("\n[bold green]🎉 PHASE 5 SUCCESS! 🎉[/bold green]")
        console.print("[green]The system achieves the 0.7s target consistently![/green]")
        success = True
    else:
        console.print("\n[bold yellow]⚠️  PHASE 5 NEEDS TUNING[/bold yellow]")
        if avg_time > 0.7:
            speedup_needed = ((avg_time - 0.7) / 0.7) * 100
            console.print(f"[yellow]Need {speedup_needed:.1f}% speedup to reach target[/yellow]")
        success = False
    
    # Performance breakdown
    console.print(f"\n[cyan]Individual Test Results:[/cyan]")
    for result in results:
        status = "✅" if result['target_achieved'] else "❌"
        console.print(f"  {status} {result['name']}: {result['time']:.3f}s")
    
    # System stats
    stats = pipeline.get_performance_report()
    console.print(f"\n[cyan]System Statistics:[/cyan]")
    console.print(f"  Device: {stats.get('device_info', {}).get('device', 'unknown')}")
    
    if 'statistics' in stats:
        stats_data = stats['statistics']
        console.print(f"  Videos generated: {stats_data.get('total_videos_generated', 0)}")
        console.print(f"  Cache hit rate: {stats_data.get('cache_hit_rate', 0):.1%}")
        console.print(f"  Best time: {stats_data.get('best_time', 0):.3f}s")
        console.print(f"  Times under target: {stats_data.get('times_under_target', 0)}")
    
    return success


async def main():
    """Main test execution"""
    try:
        success = await test_phase5_performance()
        return 0 if success else 1
    except Exception as e:
        console.print(f"\n[red]Test failed with error: {e}[/red]")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)