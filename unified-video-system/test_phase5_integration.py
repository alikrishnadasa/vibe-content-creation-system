#!/usr/bin/env python3
"""
Phase 5 Integration Test

Comprehensive test suite for the complete real content system integration.
Tests CLI commands, pipeline performance, and end-to-end functionality.
"""

import logging
import asyncio
import sys
import subprocess
from pathlib import Path
import time
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.append(str(Path(__file__).parent))

async def test_cli_integration():
    """Test CLI command integration"""
    logger.info("🧪 Testing CLI Integration...")
    
    tests = []
    
    # Test 1: CLI Help Commands
    logger.info("Testing CLI help commands...")
    try:
        result = subprocess.run([sys.executable, "main.py", "--help"], 
                              capture_output=True, text=True, timeout=30)
        tests.append(("CLI Main Help", result.returncode == 0, f"Exit code: {result.returncode}"))
        
        result = subprocess.run([sys.executable, "main.py", "real", "--help"], 
                              capture_output=True, text=True, timeout=30)
        tests.append(("CLI Real Help", result.returncode == 0, f"Exit code: {result.returncode}"))
        
        result = subprocess.run([sys.executable, "main.py", "batch-real", "--help"], 
                              capture_output=True, text=True, timeout=30)
        tests.append(("CLI Batch Help", result.returncode == 0, f"Exit code: {result.returncode}"))
        
    except Exception as e:
        tests.append(("CLI Help Commands", False, str(e)))
    
    # Test 2: Real Content Generation via CLI
    logger.info("Testing real content generation via CLI...")
    try:
        start_time = time.time()
        result = subprocess.run([
            sys.executable, "main.py", "real",
            "../11-scripts-for-tiktok/safe1.wav", 
            "-v", "1", "-s", "tiktok"
        ], capture_output=True, text=True, timeout=120)
        
        generation_time = time.time() - start_time
        
        success = result.returncode == 0
        tests.append(("CLI Real Content Generation", success, 
                     f"Exit code: {result.returncode}, Time: {generation_time:.2f}s"))
        
        if success:
            # Check if output file was created
            output_files = list(Path("output").glob("real_content_safe1_*.mp4"))
            tests.append(("CLI Generated File Exists", len(output_files) > 0, 
                         f"Found {len(output_files)} files"))
    
    except Exception as e:
        tests.append(("CLI Real Content Generation", False, str(e)))
    
    return tests

async def test_batch_processing():
    """Test batch processing capabilities"""
    logger.info("🧪 Testing Batch Processing...")
    
    tests = []
    
    # Test 1: Batch CLI Command
    logger.info("Testing batch CLI command...")
    try:
        start_time = time.time()
        result = subprocess.run([
            sys.executable, "main.py", "batch-real",
            "--scripts", "anxiety1", "safe1",
            "-v", "2", "--concurrent", "2",
            "--report", "cli_batch_test_report.json"
        ], capture_output=True, text=True, timeout=300)
        
        batch_time = time.time() - start_time
        
        success = result.returncode == 0
        tests.append(("CLI Batch Processing", success, 
                     f"Exit code: {result.returncode}, Time: {batch_time:.2f}s"))
        
        if success:
            # Check batch report
            report_file = Path("cli_batch_test_report.json")
            if report_file.exists():
                with open(report_file) as f:
                    report = json.load(f)
                
                batch_result = report["batch_result"]
                tests.append(("Batch Report Generated", True, 
                             f"Videos: {batch_result['successful_videos']}/{batch_result['total_videos']}"))
                
                # Check performance
                avg_time = batch_result["average_time_per_video"]
                tests.append(("Batch Performance", avg_time < 30.0, 
                             f"Avg time per video: {avg_time:.2f}s"))
            else:
                tests.append(("Batch Report Generated", False, "Report file not found"))
    
    except Exception as e:
        tests.append(("CLI Batch Processing", False, str(e)))
    
    return tests

async def test_system_performance():
    """Test overall system performance"""
    logger.info("🧪 Testing System Performance...")
    
    tests = []
    
    try:
        # Import real content generator directly
        from core.real_content_generator import RealContentGenerator, RealVideoRequest
        
        # Initialize generator
        generator = RealContentGenerator(
            clips_directory="../MJAnime",
            metadata_file="../MJAnime/metadata_final_clean_shots.json", 
            scripts_directory="../11-scripts-for-tiktok",
            music_file="../Beanie (Slowed).mp3",
            output_directory="output"
        )
        
        init_success = await generator.initialize()
        tests.append(("Generator Initialization", init_success, "Initialized successfully"))
        
        if init_success:
            # Test 1: Single video performance
            start_time = time.time()
            request = RealVideoRequest(
                script_path="../11-scripts-for-tiktok/miserable1.wav",
                script_name="miserable1_perf_test",
                variation_number=1,
                caption_style="tiktok",
                music_sync=True,
                target_duration=15.0
            )
            
            result = await generator.generate_video(request)
            generation_time = time.time() - start_time
            
            tests.append(("Single Video Generation", result.success, 
                         f"Time: {generation_time:.2f}s"))
            
            # Test 2: Performance target (goal: <30s for now, working toward <0.7s)
            performance_acceptable = generation_time < 30.0
            tests.append(("Performance Target (<30s)", performance_acceptable, 
                         f"Actual: {generation_time:.2f}s"))
            
            # Test 3: Audio integration
            if result.success:
                output_path = Path(result.output_path)
                file_exists = output_path.exists()
                tests.append(("Output File Created", file_exists, 
                             f"File: {output_path.name}" if file_exists else "File not found"))
                
                if file_exists:
                    file_size_mb = output_path.stat().st_size / (1024 * 1024)
                    reasonable_size = 5.0 <= file_size_mb <= 100.0
                    tests.append(("File Size Reasonable", reasonable_size, 
                                 f"Size: {file_size_mb:.1f}MB"))
    
    except Exception as e:
        tests.append(("System Performance Test", False, str(e)))
    
    return tests

async def test_component_integration():
    """Test integration between components"""
    logger.info("🧪 Testing Component Integration...")
    
    tests = []
    
    try:
        # Test content loading
        from content.mjanime_loader import MJAnimeLoader
        loader = MJAnimeLoader("../MJAnime", "../MJAnime/metadata_final_clean_shots.json")
        clips = await loader.load_clips()
        tests.append(("MJAnime Clips Loading", len(clips) > 0, f"Loaded {len(clips)} clips"))
        
        # Test music loading
        from content.music_manager import MusicManager
        music_manager = MusicManager("../Beanie (Slowed).mp3")
        music_loaded = await music_manager.initialize()
        tests.append(("Music Loading", music_loaded, "Beanie (Slowed).mp3 loaded"))
        
        # Test script analysis
        from content.script_analyzer import ScriptAnalyzer
        analyzer = ScriptAnalyzer("../11-scripts-for-tiktok")
        scripts = await analyzer.analyze_scripts()
        tests.append(("Script Analysis", len(scripts) > 0, f"Analyzed {len(scripts)} scripts"))
        
        # Test uniqueness engine
        from content.uniqueness_engine import UniquenessEngine
        uniqueness = UniquenessEngine()
        test_sequence = "test_sequence_hash"
        is_unique = uniqueness.is_unique(test_sequence, "test_script", 1)
        tests.append(("Uniqueness Engine", is_unique, "Sequence uniqueness validation"))
        
    except Exception as e:
        tests.append(("Component Integration", False, str(e)))
    
    return tests

def print_test_results(test_category: str, tests: list):
    """Print formatted test results"""
    logger.info(f"\n{'='*60}")
    logger.info(f"📊 {test_category} TEST RESULTS")
    logger.info(f"{'='*60}")
    
    passed = 0
    total = len(tests)
    
    for test_name, success, details in tests:
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"   {test_name}: {status} - {details}")
        if success:
            passed += 1
    
    logger.info(f"{'─'*60}")
    logger.info(f"📈 Result: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    return passed == total

async def main():
    """Main integration test suite"""
    logger.info("🚀 Starting Phase 5 Integration Test Suite")
    logger.info("Testing complete real content system integration...\n")
    
    all_results = []
    
    # Run all test categories
    test_categories = [
        ("CLI Integration", test_cli_integration),
        ("Batch Processing", test_batch_processing),
        ("System Performance", test_system_performance),
        ("Component Integration", test_component_integration)
    ]
    
    for category_name, test_function in test_categories:
        try:
            logger.info(f"🔍 Running {category_name} tests...")
            results = await test_function()
            success = print_test_results(category_name, results)
            all_results.append((category_name, success, results))
        except Exception as e:
            logger.error(f"❌ {category_name} test failed with exception: {e}")
            all_results.append((category_name, False, [(f"{category_name} Exception", False, str(e))]))
    
    # Final summary
    logger.info(f"\n{'='*60}")
    logger.info("🎯 PHASE 5 INTEGRATION TEST SUMMARY")
    logger.info(f"{'='*60}")
    
    total_categories = len(all_results)
    passed_categories = sum(1 for _, success, _ in all_results if success)
    
    for category_name, success, _ in all_results:
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"   {category_name}: {status}")
    
    logger.info(f"{'─'*60}")
    logger.info(f"📊 Overall Result: {passed_categories}/{total_categories} test categories passed")
    
    if passed_categories == total_categories:
        logger.info("🎉 ALL INTEGRATION TESTS PASSED!")
        logger.info("✅ Phase 5 Integration and Testing COMPLETE")
        logger.info("🚀 Real content system is ready for production!")
        return True
    else:
        logger.error("💥 SOME INTEGRATION TESTS FAILED")
        logger.error("⚠️  System needs attention before production use")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    if success:
        print("\n🎉 SUCCESS! All integration tests passed!")
        sys.exit(0)
    else:
        print("\n💥 FAILED! Integration tests need attention")
        sys.exit(1)