#!/usr/bin/env python3
"""
Cleanup Deprecated Files
Organize deprecated files after restructuring
"""

import os
import shutil
from pathlib import Path

def main():
    """Clean up deprecated files"""
    print("🧹 Cleaning up deprecated files...")
    
    # Create deprecated directory
    deprecated_dir = Path("deprecated")
    deprecated_dir.mkdir(exist_ok=True)
    
    # Files to move to deprecated (keep as backup)
    deprecated_files = [
        "generate_test_videos.py",
        "generate_enhanced_test_videos.py",
        "generate_unified_videos.py",
        "create_unified_test_video.py",
        "simple_pipeline_test.py",
        "use_mjanime_metadata.py",
        "use_mjclip_metadata.py",
        "use_unified_quantum_pipeline.py",
        "integrate_mjanime_metadata.py"
    ]
    
    # Move deprecated files
    moved_count = 0
    for file in deprecated_files:
        file_path = Path(file)
        if file_path.exists():
            dest_path = deprecated_dir / file
            shutil.move(str(file_path), str(dest_path))
            print(f"✅ Moved {file} to deprecated/")
            moved_count += 1
    
    # Create deprecated README
    deprecated_readme = deprecated_dir / "README.md"
    with open(deprecated_readme, 'w') as f:
        f.write("""# Deprecated Files

This directory contains files that have been replaced by the new unified system.

## Replacement Commands

### Old Scripts → New Commands

| Deprecated File | New Command |
|----------------|-------------|
| `generate_enhanced_test_videos.py` | `python3 vibe.py single <script>` |
| `generate_unified_videos.py` | `python3 vibe.py single <script>` |
| `generate_test_videos.py` | `python3 vibe.py single <script>` |
| `create_unified_test_video.py` | `python3 vibe.py single <script>` |
| `simple_pipeline_test.py` | `python3 vibe.py single <script> --no-enhanced --no-quantum` |
| `use_mjanime_metadata.py` | Functionality integrated into `vibe.py` |
| `use_mjclip_metadata.py` | Functionality integrated into `vibe.py` |
| `use_unified_quantum_pipeline.py` | `python3 vibe.py` (default behavior) |
| `integrate_mjanime_metadata.py` | `tools/metadata/` utilities |

## Why These Files Are Deprecated

- **Functionality Consolidated**: All features moved to unified interface
- **Better Maintainability**: Single codebase easier to maintain
- **Improved User Experience**: One command instead of many scripts
- **Enhanced Features**: New system has additional capabilities

## Recovery

These files are preserved for reference. If you need to restore any functionality:

1. Check if it's available in the new system first
2. Refer to MIGRATION_GUIDE.md for equivalent commands
3. If absolutely necessary, these files can be restored from this directory

## Safe to Delete

These files are safe to delete after confirming the new system works correctly.
""")
    
    print(f"\n📊 Summary:")
    print(f"✅ Moved {moved_count} deprecated files")
    print(f"📁 Created deprecated/ directory with README")
    print(f"🧹 Cleanup completed successfully!")
    
    return moved_count > 0

if __name__ == "__main__":
    main()