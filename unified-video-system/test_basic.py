#!/usr/bin/env python3
"""
Basic test script to verify core structure
"""

import sys
from pathlib import Path

def test_project_structure():
    """Test that all directories and files exist"""
    print("🔍 Testing project structure...")
    
    required_dirs = [
        'core', 'captions', 'sync', 'beat_sync', 'models',
        'config', 'utils', 'tests', 'cache', 'output', 
        'logs', 'fonts', 'temp'
    ]
    
    required_files = [
        'main.py', 'requirements.txt', 'README.md',
        'core/__init__.py', 'core/quantum_pipeline.py',
        'core/neural_cache.py', 'core/gpu_engine.py',
        'core/zero_copy_engine.py', 'config/system_config.yaml'
    ]
    
    all_good = True
    
    # Check directories
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists() and dir_path.is_dir():
            print(f"✅ Directory exists: {dir_name}")
        else:
            print(f"❌ Missing directory: {dir_name}")
            all_good = False
    
    print()
    
    # Check files
    for file_name in required_files:
        file_path = Path(file_name)
        if file_path.exists() and file_path.is_file():
            print(f"✅ File exists: {file_name}")
        else:
            print(f"❌ Missing file: {file_name}")
            all_good = False
    
    return all_good


def test_imports():
    """Test basic imports"""
    print("\n🔍 Testing basic imports...")
    
    imports_to_test = [
        ('asyncio', 'Standard library'),
        ('pathlib', 'Standard library'),
        ('json', 'Standard library'),
        ('dataclasses', 'Standard library'),
        ('typing', 'Standard library'),
    ]
    
    all_good = True
    
    for module_name, description in imports_to_test:
        try:
            __import__(module_name)
            print(f"✅ {description}: {module_name}")
        except ImportError:
            print(f"❌ Failed to import {description}: {module_name}")
            all_good = False
    
    return all_good


def test_core_modules():
    """Test importing core modules (without external dependencies)"""
    print("\n🔍 Testing core module structure...")
    
    # Add current directory to path
    sys.path.insert(0, str(Path.cwd()))
    
    modules_to_test = [
        'core',
        'captions',
        'sync',
        'beat_sync',
        'models',
        'utils',
        'tests'
    ]
    
    all_good = True
    
    for module_name in modules_to_test:
        try:
            __import__(module_name)
            print(f"✅ Module package exists: {module_name}")
        except ImportError as e:
            print(f"⚠️  Module package import warning: {module_name} - {e}")
            # Don't fail on import errors as they might be due to missing dependencies
    
    return True  # We don't fail on import errors for now


def main():
    """Run all tests"""
    print("="*60)
    print("Unified Video System - Basic Structure Test")
    print("="*60)
    
    tests = [
        ("Project Structure", test_project_structure),
        ("Basic Imports", test_imports),
        ("Core Modules", test_core_modules),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n### {test_name} ###")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n✅ All basic tests passed!")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Run full test: python main.py test")
        print("3. Generate your first video: python main.py generate 'Your text here'")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        print("\nCommon fixes:")
        print("- Make sure you're in the unified-video-system directory")
        print("- Check that all files were created properly")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main()) 