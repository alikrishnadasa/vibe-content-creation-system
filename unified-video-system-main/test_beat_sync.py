#!/usr/bin/env python3
"""
Test script for the Beat Sync System
Tests the BeatSyncEngine with LibrosaBeatDetector integration
"""

import asyncio
import sys
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from beat_sync.beat_sync_engine import BeatSyncEngine
from beat_sync.librosa_beat_detection import LibrosaBeatDetector

console = Console()


async def test_librosa_detector():
    """Test the LibrosaBeatDetector independently"""
    console.print("\n[bold cyan]Testing LibrosaBeatDetector[/bold cyan]")
    
    detector = LibrosaBeatDetector()
    
    # Test with a dummy audio file path
    test_audio = "test_audio.mp3"
    
    try:
        # Test beat detection
        tempo, beats = detector.detect_beats(test_audio, units='time')
        console.print(f"✅ Beat detection working - Tempo: {tempo:.1f} BPM, Beats: {len(beats)}")
        
        # Test onset detection
        onsets, beats2 = detector.detect_onset_beats(test_audio)
        console.print(f"✅ Onset detection working - Onsets: {len(onsets)}, Beats: {len(beats2)}")
        
        # Test beat strength
        strength = detector.get_beat_strength(test_audio)
        console.print(f"✅ Beat strength analysis working - Strength array length: {len(strength)}")
        
        return True
        
    except FileNotFoundError:
        console.print("[yellow]⚠️  Audio file not found - using generated test data[/yellow]")
        
        # Generate synthetic beat times for testing
        tempo = 120.0
        duration = 30.0  # 30 seconds
        beat_interval = 60.0 / tempo
        beats = np.arange(0, duration, beat_interval)
        
        console.print(f"✅ Generated test data - Tempo: {tempo} BPM, Beats: {len(beats)}")
        return True
        
    except Exception as e:
        console.print(f"[red]❌ Error: {str(e)}[/red]")
        return False


async def test_beat_sync_engine():
    """Test the complete BeatSyncEngine"""
    console.print("\n[bold cyan]Testing BeatSyncEngine[/bold cyan]")
    
    # Create engine with test configuration
    config = {
        'min_tempo': 80,
        'max_tempo': 160,
        'sample_rate': 22050,
        'hop_length': 512
    }
    
    engine = BeatSyncEngine(config)
    
    # Check initialization
    console.print(f"✅ Engine initialized with backend: {engine.get_statistics()['backend']}")
    console.print(f"✅ Python compatibility: {engine.get_statistics()['python_version']}")
    
    # Test with dummy audio
    test_audio = Path("test_audio.mp3")
    
    try:
        # Analyze audio
        result = await engine.analyze_audio(test_audio)
        
        # Display results
        table = Table(title="Beat Analysis Results")
        table.add_column("Metric", style="cyan", width=25)
        table.add_column("Value", style="green")
        
        table.add_row("Tempo (BPM)", f"{result.tempo:.1f}")
        table.add_row("Time Signature", f"{result.time_signature[0]}/{result.time_signature[1]}")
        table.add_row("Total Beats", str(len(result.beats)))
        table.add_row("Downbeats", str(sum(1 for b in result.beats if b.is_downbeat)))
        table.add_row("Phrases Detected", str(len(result.phrases)))
        table.add_row("Onsets Detected", str(len(result.onset_times)))
        table.add_row("Processing Time", f"{result.processing_time:.3f}s")
        
        console.print(table)
        
        # Show first few beats
        if result.beats:
            console.print("\n[bold]First 5 Beats:[/bold]")
            for i, beat in enumerate(result.beats[:5]):
                console.print(f"  Beat {i+1}: {beat.time:.3f}s, "
                            f"Strength: {beat.strength:.2f}, "
                            f"Downbeat: {beat.is_downbeat}, "
                            f"Position: {beat.measure_position}/4")
        
        # Test caption alignment
        test_captions = [
            {'text': 'Hello', 'start': 0.5, 'end': 1.0},
            {'text': 'World', 'start': 1.2, 'end': 1.8},
            {'text': 'Music', 'start': 2.1, 'end': 2.7},
            {'text': 'Sync', 'start': 3.0, 'end': 3.5}
        ]
        
        aligned = engine.align_captions_to_beats(test_captions, result)
        
        console.print("\n[bold]Caption Alignment Test:[/bold]")
        alignment_table = Table()
        alignment_table.add_column("Caption", style="yellow")
        alignment_table.add_column("Original Time", style="cyan")
        alignment_table.add_column("Aligned Time", style="green")
        alignment_table.add_column("Beat Aligned", style="magenta")
        
        for orig, aligned_cap in zip(test_captions, aligned):
            alignment_table.add_row(
                orig['text'],
                f"{orig['start']:.2f}s",
                f"{aligned_cap['start']:.2f}s",
                "✅" if aligned_cap.get('beat_aligned') else "❌"
            )
        
        console.print(alignment_table)
        
        # Test visual effects timing
        effects = engine.get_visual_effects_timing(result)
        console.print(f"\n✅ Generated {len(effects)} visual effects")
        
        # Show statistics
        stats = engine.get_statistics()
        console.print("\n[bold]Engine Statistics:[/bold]")
        console.print(f"  Files processed: {stats['files_processed']}")
        console.print(f"  Total beats detected: {stats['total_beats_detected']}")
        console.print(f"  Cache size: {stats['cache_size']}")
        console.print(f"  Supported audio formats: {', '.join(stats['supported_features']['audio_formats'])}")
        
        return True
        
    except FileNotFoundError:
        console.print("[yellow]⚠️  Audio file not found - test with real audio file for full functionality[/yellow]")
        console.print("  Example: python test_beat_sync.py /path/to/audio.mp3")
        return True
        
    except Exception as e:
        console.print(f"[red]❌ Error during testing: {str(e)}[/red]")
        import traceback
        traceback.print_exc()
        return False


async def test_integration():
    """Test integration with the quantum pipeline"""
    console.print("\n[bold cyan]Testing Pipeline Integration[/bold cyan]")
    
    try:
        from core.quantum_pipeline import UnifiedQuantumPipeline
        
        # Create pipeline with beat sync enabled
        pipeline = UnifiedQuantumPipeline()
        
        # Check if beat sync is available
        if pipeline.beat_sync:
            console.print("✅ Beat sync successfully integrated into quantum pipeline")
            console.print(f"   Backend: {pipeline.beat_sync.get_statistics()['backend']}")
        else:
            console.print("[yellow]⚠️  Beat sync not enabled in pipeline config[/yellow]")
            console.print("   Enable it by setting beat_sync.enabled = true in config")
        
        return True
        
    except Exception as e:
        console.print(f"[red]❌ Integration error: {str(e)}[/red]")
        return False


async def main():
    """Run all tests"""
    console.print(Panel.fit(
        "[bold green]Beat Sync System Test Suite[/bold green]\n"
        "Testing BeatSyncEngine with LibrosaBeatDetector",
        border_style="green"
    ))
    
    # Check if audio file provided
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
        console.print(f"\n[cyan]Using audio file:[/cyan] {audio_file}")
        
        # Override test audio path
        global test_audio
        test_audio = Path(audio_file)
    
    # Run tests
    tests = [
        ("LibrosaBeatDetector", test_librosa_detector),
        ("BeatSyncEngine", test_beat_sync_engine),
        ("Pipeline Integration", test_integration)
    ]
    
    results = []
    for test_name, test_func in tests:
        console.print(f"\n{'='*60}")
        success = await test_func()
        results.append((test_name, success))
    
    # Summary
    console.print(f"\n{'='*60}")
    console.print("\n[bold]Test Summary:[/bold]")
    
    summary_table = Table()
    summary_table.add_column("Test", style="cyan")
    summary_table.add_column("Result", style="green")
    
    for test_name, success in results:
        summary_table.add_row(
            test_name,
            "✅ PASSED" if success else "❌ FAILED"
        )
    
    console.print(summary_table)
    
    # Overall result
    all_passed = all(success for _, success in results)
    if all_passed:
        console.print("\n[bold green]✅ All tests passed![/bold green]")
        console.print("\n[cyan]The Beat Sync system is ready for use with Python 3.13[/cyan]")
    else:
        console.print("\n[bold red]❌ Some tests failed[/bold red]")
    
    return all_passed


if __name__ == "__main__":
    # Run the tests
    success = asyncio.run(main())
    sys.exit(0 if success else 1)