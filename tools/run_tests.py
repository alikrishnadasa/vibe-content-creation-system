#!/usr/bin/env python3
"""
Test Runner for Vibe Content Creation System
Runs all tests in the new structured format
"""

import sys
import subprocess
import os
from pathlib import Path

def run_command(cmd, description):
    """Run a command and return success status"""
    print(f"\n{'='*60}")
    print(f"🧪 {description}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(f"STDERR: {result.stderr}")
            
        success = result.returncode == 0
        print(f"Status: {'✅ PASSED' if success else '❌ FAILED'}")
        return success
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False

def main():
    """Main test runner"""
    print("🚀 Vibe Content Creation - Test Runner")
    print("Running comprehensive test suite for restructured system")
    
    # Change to project directory
    os.chdir(Path(__file__).parent)
    
    test_results = []
    
    # Test 1: Basic system functionality
    test_results.append(run_command(
        "python3 vibe.py status",
        "Basic System Status Test"
    ))
    
    # Test 2: Legacy compatibility
    test_results.append(run_command(
        "python3 vibe_generator.py status",
        "Legacy System Compatibility Test"
    ))
    
    # Test 3: Unit tests
    test_results.append(run_command(
        "python3 tests/unit/test_vibe_generator.py",
        "Unit Tests - Vibe Generator"
    ))
    
    # Test 4: Import tests
    test_results.append(run_command(
        "python3 -c \"from src.generation.vibe_generator import VibeGenerator; print('✅ Import successful')\"",
        "Import Test - New Structure"
    ))
    
    # Test 5: Directory structure
    test_results.append(run_command(
        "python3 -c \"from pathlib import Path; paths = ['src/core', 'src/content', 'src/captions', 'tests/unit', 'tools/metadata']; all_exist = all(Path(p).exists() for p in paths); print('✅ Directory structure OK' if all_exist else '❌ Missing directories')\"",
        "Directory Structure Test"
    ))
    
    # Test 6: File accessibility
    test_results.append(run_command(
        "python3 -c \"from pathlib import Path; files = ['src/core/quantum_pipeline.py', 'src/content/content_database.py', 'src/captions/unified_caption_engine.py']; all_exist = all(Path(f).exists() for f in files); print('✅ Core files accessible' if all_exist else '❌ Missing core files')\"",
        "File Accessibility Test"
    ))
    
    # Test 7: Configuration access
    test_results.append(run_command(
        "python3 -c \"from pathlib import Path; config_exists = Path('config/system_config.yaml').exists(); print('✅ Configuration accessible' if config_exists else '❌ Configuration missing')\"",
        "Configuration Access Test"
    ))
    
    # Test 8: Data access
    test_results.append(run_command(
        "python3 -c \"from pathlib import Path; data_exists = Path('data/scripts/11-scripts-for-tiktok').exists(); print('✅ Data accessible' if data_exists else '❌ Data missing')\"",
        "Data Access Test"
    ))
    
    # Test 9: Tools access
    test_results.append(run_command(
        "python3 -c \"from pathlib import Path; tools_exist = Path('tools/metadata').exists() and Path('tools/analysis').exists(); print('✅ Tools accessible' if tools_exist else '❌ Tools missing')\"",
        "Tools Access Test"
    ))
    
    # Test 10: Backward compatibility
    test_results.append(run_command(
        "python3 -c \"from pathlib import Path; legacy_exists = Path('unified-video-system-main/core/quantum_pipeline.py').exists(); print('✅ Legacy system preserved' if legacy_exists else '❌ Legacy system missing')\"",
        "Backward Compatibility Test"
    ))
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 TEST SUMMARY")
    print(f"{'='*60}")
    
    passed = sum(test_results)
    total = len(test_results)
    
    print(f"Tests Passed: {passed}/{total}")
    print(f"Success Rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! System restructure successful!")
        print("\n✅ Ready to use:")
        print("   python3 vibe.py single anxiety1")
        print("   python3 vibe.py batch anxiety1 safe1")
        print("   python3 vibe.py status")
    else:
        print("⚠️  Some tests failed. Please check the output above.")
        print("\n🔄 Fallback options:")
        print("   python3 vibe_generator.py single anxiety1  # Use legacy")
        print("   python3 unified-video-system-main/main.py status  # Use original")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)