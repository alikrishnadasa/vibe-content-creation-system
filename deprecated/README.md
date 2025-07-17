# Deprecated Files

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
