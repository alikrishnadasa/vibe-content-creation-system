# Directory Structure Guide

## Overview

The Vibe Content Creation system has been restructured for better organization and maintainability. This document explains the new directory structure and how to use it.

## New Directory Structure

```
vibe-content-creation/
├── vibe.py                    # 🎯 NEW MAIN ENTRY POINT
├── vibe_generator.py          # 🔄 LEGACY ENTRY POINT (still works)
├── src/                       # 📁 SOURCE CODE
│   ├── __init__.py
│   ├── core/                  # Core processing engines
│   │   ├── quantum_pipeline.py
│   │   ├── real_content_generator.py
│   │   ├── ffmpeg_video_processor.py
│   │   ├── gpu_engine.py
│   │   ├── neural_cache.py
│   │   └── performance_optimizer.py
│   ├── content/               # Content management
│   │   ├── content_database.py
│   │   ├── content_selector.py
│   │   ├── mjanime_loader.py
│   │   ├── music_manager.py
│   │   ├── script_analyzer.py
│   │   └── uniqueness_engine.py
│   ├── captions/              # Caption system
│   │   ├── unified_caption_engine.py
│   │   ├── preset_manager.py
│   │   ├── script_caption_cache.py
│   │   └── whisper_transcriber.py
│   ├── generation/            # Video generation
│   │   ├── vibe_generator.py
│   │   ├── enhanced_script_analyzer.py
│   │   ├── enhanced_content_selector.py
│   │   └── enhanced_metadata_creator.py
│   └── utils/                 # Utility functions
├── tests/                     # 🧪 TESTS
│   ├── unit/                  # Unit tests
│   │   ├── test_vibe_generator.py
│   │   ├── test_unified_pipeline.py
│   │   ├── test_quantum_pipeline.py
│   │   ├── test_quantum_captions.py
│   │   └── test_local.py
│   └── integration/           # Integration tests
│       ├── test_integration.py
│       ├── test_mjanime_integration.py
│       ├── test_unified_integration.py
│       └── test_caption_fix.py
├── tools/                     # 🔧 DEVELOPMENT TOOLS
│   ├── metadata/              # Metadata utilities
│   │   ├── create_unified_metadata.py
│   │   ├── create_mjanime_metadata.py
│   │   └── create_mjclip_metadata.py
│   ├── analysis/              # Analysis utilities
│   │   ├── analyze_mjanime_clips.py
│   │   ├── analyze_mjclips.py
│   │   └── simple_mjanime_analysis.py
│   ├── improve_content_matching.py
│   └── integrate_enhanced_system.py
├── config/                    # ⚙️ CONFIGURATION
│   ├── system_config.yaml
│   └── __init__.py
├── data/                      # 📊 DATA FILES
│   ├── scripts/               # Audio scripts
│   │   └── 11-scripts-for-tiktok/
│   ├── clips/                 # Video clips (symlinked)
│   │   ├── MJAnime -> ../../MJAnime
│   │   └── midjourney_composite -> ../../midjourney_composite_2025-7-15
│   └── metadata/              # Metadata files
│       ├── unified_clips_metadata.json
│       ├── unified_enhanced_metadata.json
│       ├── mjanime_metadata.json
│       └── enhanced_midjourney_metadata.json
├── docs/                      # 📚 DOCUMENTATION
│   └── guides/
│       └── DIRECTORY_STRUCTURE.md (this file)
├── cache/                     # 🚀 CACHE FILES
├── output/                    # 📹 GENERATED VIDEOS
└── unified-video-system-main/ # 🔄 LEGACY SYSTEM (preserved)
```

## Usage Examples

### New Entry Point (Recommended)
```bash
# Use the new structured entry point
python3 vibe.py single anxiety1
python3 vibe.py batch anxiety1 safe1
python3 vibe.py status
```

### Legacy Entry Point (Still Works)
```bash
# Original entry point still works
python3 vibe_generator.py single anxiety1
python3 vibe_generator.py batch anxiety1 safe1
python3 vibe_generator.py status
```

## File Locations

### Core Components
- **Main Interface**: `vibe.py` (new) or `vibe_generator.py` (legacy)
- **Core Processing**: `src/core/`
- **Content Management**: `src/content/`
- **Caption System**: `src/captions/`
- **Enhanced System**: `src/generation/`

### Development Tools
- **Metadata Tools**: `tools/metadata/`
- **Analysis Tools**: `tools/analysis/`
- **General Tools**: `tools/`

### Testing
- **Unit Tests**: `tests/unit/`
- **Integration Tests**: `tests/integration/`

### Configuration & Data
- **Configuration**: `config/`
- **Scripts**: `data/scripts/`
- **Metadata**: `data/metadata/`
- **Generated Videos**: `output/`

## Benefits of New Structure

### For Users
- **Clearer Organization**: Logical grouping of related files
- **Better Documentation**: Dedicated docs directory
- **Easier Navigation**: Intuitive directory names
- **Backward Compatibility**: Old commands still work

### For Developers
- **Maintainability**: Separated concerns
- **Testability**: Organized test structure
- **Extensibility**: Clear places for new features
- **Debuggability**: Logical file organization

## Migration Notes

### Import Changes
If you were importing modules directly:
```python
# Old way (still works)
from vibe_generator import VibeGenerator

# New way (recommended)
from src.generation.vibe_generator import VibeGenerator
```

### Path Changes
- **Scripts**: `11-scripts-for-tiktok/` → `data/scripts/11-scripts-for-tiktok/`
- **Metadata**: Root directory → `data/metadata/`
- **Config**: `unified-video-system-main/config/` → `config/`
- **Tests**: Scattered → `tests/unit/` and `tests/integration/`

### Backward Compatibility
- All original commands work unchanged
- Legacy entry points preserved
- Original file locations still accessible
- No breaking changes to existing workflows

## Running Tests

### Unit Tests
```bash
# Run all unit tests
python3 -m pytest tests/unit/

# Run specific test
python3 tests/unit/test_vibe_generator.py
```

### Integration Tests
```bash
# Run all integration tests
python3 -m pytest tests/integration/

# Run specific integration test
python3 tests/integration/test_unified_integration.py
```

### All Tests
```bash
# Run all tests
python3 -m pytest tests/
```

## Development Workflow

### Adding New Features
1. **Core functionality**: Add to `src/core/`
2. **Content management**: Add to `src/content/`
3. **Caption features**: Add to `src/captions/`
4. **Generation features**: Add to `src/generation/`
5. **Utilities**: Add to `src/utils/`

### Adding Tools
1. **Metadata tools**: Add to `tools/metadata/`
2. **Analysis tools**: Add to `tools/analysis/`
3. **General tools**: Add to `tools/`

### Adding Tests
1. **Unit tests**: Add to `tests/unit/`
2. **Integration tests**: Add to `tests/integration/`

### Adding Documentation
1. **Guides**: Add to `docs/guides/`
2. **API docs**: Add to `docs/api/`

## Support

If you encounter issues with the new structure:

1. **Check paths**: Ensure you're using the correct paths
2. **Use legacy**: Fall back to `vibe_generator.py` if needed
3. **Check imports**: Verify import paths are correct
4. **Test functionality**: Run tests to verify everything works

The new structure maintains 100% backward compatibility while providing better organization for future development.