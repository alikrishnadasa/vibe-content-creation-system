# Migration Guide: Old Scripts → Unified Vibe Generator

## Overview

The Vibe Content Creation system has been consolidated from **17 generation scripts** into **1 unified interface**. This guide helps you migrate from the old scripts to the new `vibe_generator.py`.

## Script Migration Map

### Single Video Generation
| Old Script | New Command | Notes |
|------------|-------------|-------|
| `generate_enhanced_test_videos.py` | `python3 vibe_generator.py single <script>` | Enhanced system enabled by default |
| `generate_unified_videos.py` | `python3 vibe_generator.py single <script>` | Unified metadata used automatically |
| `generate_test_videos.py` | `python3 vibe_generator.py single <script>` | Basic generation mode |
| `create_unified_test_video.py` | `python3 vibe_generator.py single <script>` | Unified clips used automatically |
| `simple_pipeline_test.py` | `python3 vibe_generator.py single <script> --no-enhanced --no-quantum` | Simple mode |

### Batch Generation
| Old Script | New Command | Notes |
|------------|-------------|-------|
| `unified-video-system-main/main.py batch-real` | `python3 vibe_generator.py batch <scripts>` | Real content batch processing |
| `unified-video-system-main/generate_batch_videos.py` | `python3 vibe_generator.py batch <scripts>` | Batch generation |
| Multiple test scripts | `python3 vibe_generator.py batch <scripts>` | Consolidated batch testing |

### CLI Interface
| Old Script | New Command | Notes |
|------------|-------------|-------|
| `unified-video-system-main/main.py` | `python3 vibe_generator.py` | Full CLI interface |
| `unified-video-system-main/main.py real` | `python3 vibe_generator.py single <script>` | Real content generation |
| `unified-video-system-main/main.py status` | `python3 vibe_generator.py status` | System status |

## Migration Examples

### Before (Old Scripts)
```bash
# Generate single video with enhanced system
python3 generate_enhanced_test_videos.py

# Generate unified video
python3 generate_unified_videos.py

# Generate batch with CLI
python3 unified-video-system-main/main.py batch-real -v 3 --scripts anxiety1 safe1

# Check system status
python3 unified-video-system-main/main.py status
```

### After (New Unified System)
```bash
# Generate single video with enhanced system (default)
python3 vibe_generator.py single anxiety1

# Generate unified video (same command)
python3 vibe_generator.py single anxiety1

# Generate batch with variations
python3 vibe_generator.py batch anxiety1 safe1 -v 3

# Check system status
python3 vibe_generator.py status
```

## Feature Mapping

### Enhanced Semantic System
- **Before**: `integrate_enhanced_system.py` + `generate_enhanced_test_videos.py`
- **After**: `python3 vibe_generator.py single <script>` (enabled by default)
- **Disable**: `python3 vibe_generator.py single <script> --no-enhanced`

### Quantum Pipeline
- **Before**: `use_unified_quantum_pipeline.py` + complex setup
- **After**: `python3 vibe_generator.py single <script>` (enabled by default)
- **Disable**: `python3 vibe_generator.py single <script> --no-quantum`

### Caption Styles
- **Before**: Hardcoded in scripts or CLI args
- **After**: `python3 vibe_generator.py single <script> -s <style>`
- **Styles**: `tiktok`, `youtube`, `cinematic`, `minimal`, `karaoke`

### Output Directory
- **Before**: Hardcoded `output/` directory
- **After**: `python3 vibe_generator.py single <script> -o <directory>`

### Music Sync
- **Before**: Various flags across different scripts
- **After**: `--no-music` to disable (enabled by default)

### Variations
- **Before**: Multiple script runs or batch parameters
- **After**: `python3 vibe_generator.py single <script> -v <count>`

## System Architecture Changes

### Old System (17 Scripts)
```
generate_enhanced_test_videos.py
├── integrate_enhanced_system.py
├── enhanced_script_analyzer.py
└── enhanced_content_selector.py

generate_unified_videos.py
├── real_content_generator.py
└── unified_clips_metadata.json

unified-video-system-main/main.py
├── quantum_pipeline.py
├── real_content_generator.py
└── various CLI commands
```

### New System (1 Script)
```
vibe_generator.py
├── Enhanced Semantic System (optional)
├── Quantum Pipeline (optional)
├── Real Content Generator (always)
└── Unified CLI Interface
```

## Configuration Changes

### Old Configuration
Multiple config files and hardcoded settings:
- `unified-video-system-main/config/system_config.yaml`
- Hardcoded paths in scripts
- Various CLI argument patterns

### New Configuration
Single configuration object:
```python
GenerationConfig(
    script_name="anxiety1",
    variation_number=3,
    caption_style="tiktok",
    use_enhanced_system=True,
    use_quantum_pipeline=True,
    music_sync=True,
    burn_in_captions=True,
    output_directory="output"
)
```

## Testing Migration

### Test Old System (Before Migration)
```bash
# Test each old script individually
python3 generate_enhanced_test_videos.py  # Should work
python3 generate_unified_videos.py        # Should work
python3 unified-video-system-main/main.py status  # Should work
```

### Test New System (After Migration)
```bash
# Test unified system
python3 vibe_generator.py status          # Should work
python3 vibe_generator.py single anxiety1 # Should work with all systems
python3 vibe_generator.py batch anxiety1 safe1 # Should work for batch
```

## Backward Compatibility

### Preserved Features
- ✅ All video generation functionality
- ✅ Enhanced semantic system
- ✅ Quantum pipeline
- ✅ Real content generation
- ✅ Caption styles
- ✅ Music synchronization
- ✅ Batch processing
- ✅ CLI interface

### Deprecated Features
- ❌ Individual script execution
- ❌ Multiple CLI interfaces
- ❌ Hardcoded configuration
- ❌ Scattered test files

## Rollback Plan

If you need to rollback to the old system:

1. **Restore from git**:
   ```bash
   git checkout HEAD~1  # Go back to pre-refactoring state
   ```

2. **Use backup files**:
   - Old scripts are preserved in git history
   - Core system files remain unchanged
   - Metadata files are backward compatible

3. **Manual restoration**:
   ```bash
   # The old scripts still exist in the system
   python3 generate_enhanced_test_videos.py  # Still works
   python3 generate_unified_videos.py        # Still works
   ```

## Benefits of Migration

### For Users
- **Simplified**: 1 command instead of 17 scripts
- **Consistent**: Same interface for all operations
- **Powerful**: All features available in one place
- **Flexible**: Easy to enable/disable features
- **Documented**: Clear help and examples

### For Developers
- **Maintainable**: Single codebase to update
- **Testable**: Unified test suite
- **Extensible**: Easy to add new features
- **Debuggable**: Centralized error handling

## Support

If you encounter issues during migration:

1. **Check system status**: `python3 vibe_generator.py status`
2. **Use verbose mode**: `python3 vibe_generator.py -V single <script>`
3. **Check logs**: Look at `vibe_content_creation.log`
4. **Fallback options**: Use `--no-enhanced` or `--no-quantum` if needed
5. **Test basic functionality**: Try `python3 vibe_generator.py single anxiety1`

## Next Steps

1. **Try the new system**: `python3 vibe_generator.py single anxiety1`
2. **Test your workflows**: Migrate your common commands
3. **Update documentation**: Update any personal notes or scripts
4. **Provide feedback**: Report any issues or missing features

The new unified system maintains 100% functionality while providing a much cleaner, more maintainable interface.