#!/usr/bin/env python3
"""
Vibe Content Creation - Structure Maintenance Script
Helps maintain clean directory structure by organizing files
"""

import os
import shutil
from pathlib import Path
from typing import Dict, List

def get_file_categories() -> Dict[str, List[str]]:
    """Define file categories and their target directories"""
    return {
        "docs/specs/": [
            "*_PRD.md",
            "*_spec.md",
            "*_requirements.md"
        ],
        "docs/implementation/": [
            "*_IMPLEMENTATION.md",
            "*_INTEGRATION.md",
            "REFACTORING_*.md",
            "*-system.md",
            "*-pipeline.md"
        ],
        "docs/guides/": [
            "*_README.md",
            "*_GUIDE.md",
            "*_INSTRUCTIONS.md"
        ],
        "tools/analysis/": [
            "analyze_*.py",
            "*_analysis.py",
            "simple_*.py"
        ],
        "tools/metadata/": [
            "create_*_metadata.py",
            "*_metadata_*.py"
        ],
        "tools/utils/": [
            "*_utils.py",
            "fix_*.py",
            "normalize_*.py"
        ],
        "tools/": [
            "*_content_*.py",
            "*_script_*.py",
            "integrate_*.py",
            "improve_*.py",
            "enhance_*.py",
            "cleanup_*.py",
            "run_*.py",
            "start_*.sh"
        ],
        "tests/unit/": [
            "test_*_generator.py",
            "test_*_pipeline.py",
            "test_*_engine.py"
        ],
        "tests/integration/": [
            "test_*_integration.py",
            "test_*_fix.py"
        ],
        "data/metadata/": [
            "*_metadata.json",
            "*_analysis_results*.json"
        ],
        "output/test_outputs/": [
            "real_content_*.mp4",
            "test_*.mp4"
        ],
        "output/caption_tests/": [
            "*_caption_*.mp4"
        ]
    }

def should_preserve_in_root(file_path: Path) -> bool:
    """Check if file should be preserved in root directory"""
    preserve_files = {
        "vibe.py",
        "vibe_generator.py", 
        "README.md",
        "MIGRATION_GUIDE.md",
        "requirements.txt",
        "vercel.json",
        "unified_enhanced_metadata.json",
        "unified_clips_metadata.json",
        "mjanime_metadata.json",
        "enhanced_midjourney_metadata.json"
    }
    
    preserve_dirs = {
        "11-scripts-for-tiktok"
    }
    
    return file_path.name in preserve_files or file_path.name in preserve_dirs

def organize_files(root_dir: Path, dry_run: bool = True) -> Dict[str, List[str]]:
    """Organize files according to categories"""
    categories = get_file_categories()
    moves = {}
    
    for item in root_dir.iterdir():
        if item.is_dir():
            # Skip organized directories and preserve directories
            if item.name in ["src", "tests", "tools", "docs", "data", "config", "cache", "output", "deprecated", "MJAnime", "midjourney_composite_2025-7-15", "unified-video-system-main", "frontend", "backend", "api", "video-clip-contextualizer"]:
                continue
            if should_preserve_in_root(item):
                continue
                
        elif item.is_file():
            if should_preserve_in_root(item):
                continue
                
        # Find matching category
        for target_dir, patterns in categories.items():
            for pattern in patterns:
                if item.match(pattern):
                    if target_dir not in moves:
                        moves[target_dir] = []
                    moves[target_dir].append(str(item))
                    break
            else:
                continue
            break
    
    # Execute moves if not dry run
    if not dry_run:
        for target_dir, files in moves.items():
            target_path = root_dir / target_dir
            target_path.mkdir(parents=True, exist_ok=True)
            
            for file_path in files:
                source = Path(file_path)
                dest = target_path / source.name
                
                if source.exists():
                    print(f"Moving {source.name} → {target_dir}")
                    shutil.move(str(source), str(dest))
    
    return moves

def cleanup_empty_directories(root_dir: Path, dry_run: bool = True):
    """Remove empty directories (except preserved ones)"""
    preserve_dirs = {"src", "tests", "tools", "docs", "data", "config", "cache", "output", "deprecated"}
    
    for item in root_dir.iterdir():
        if item.is_dir() and item.name not in preserve_dirs:
            try:
                if not any(item.iterdir()):  # Directory is empty
                    if not dry_run:
                        print(f"Removing empty directory: {item.name}")
                        item.rmdir()
                    else:
                        print(f"Would remove empty directory: {item.name}")
            except OSError:
                pass  # Directory not empty or can't be removed

def main():
    """Main cleanup function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Maintain clean directory structure")
    parser.add_argument("--dry-run", action="store_true", default=True, help="Show what would be done without making changes")
    parser.add_argument("--execute", action="store_true", help="Actually perform the cleanup")
    
    args = parser.parse_args()
    dry_run = not args.execute
    
    root_dir = Path(__file__).parent.parent
    
    print("🧹 Vibe Content Creation - Structure Maintenance")
    print(f"📁 Root directory: {root_dir}")
    print(f"🔄 Mode: {'DRY RUN' if dry_run else 'EXECUTE'}")
    print()
    
    # Organize files
    moves = organize_files(root_dir, dry_run)
    
    if moves:
        print("📋 Files to organize:")
        for target_dir, files in moves.items():
            print(f"  {target_dir}:")
            for file in files:
                print(f"    - {Path(file).name}")
        print()
    else:
        print("✅ No files need organizing")
    
    # Cleanup empty directories
    cleanup_empty_directories(root_dir, dry_run)
    
    if dry_run:
        print("\n💡 To execute these changes, run:")
        print("   python3 tools/maintain_clean_structure.py --execute")
    else:
        print("\n✅ Structure maintenance completed!")

if __name__ == "__main__":
    main()