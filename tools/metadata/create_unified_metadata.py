#!/usr/bin/env python3
"""
Unified Metadata Creator
Merges MJAnime and midjourney_composite metadata into single database for quantum pipeline
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

def load_metadata(file_path: str) -> Dict[str, Any]:
    """Load metadata from JSON file"""
    with open(file_path, 'r') as f:
        return json.load(f)

def normalize_clip_entry(clip: Dict[str, Any], source_type: str, base_directory: str) -> Dict[str, Any]:
    """Normalize clip entry to unified format"""
    # Get the base path for the source
    if source_type == "mjanime":
        # MJAnime clips are in MJAnime/fixed/ subdirectory
        file_path = f"MJAnime/fixed/{clip['filename']}"
    else:
        # Midjourney composite clips are directly in the directory
        file_path = f"midjourney_composite_2025-7-15/{clip['filename']}"
    
    # Create unified entry
    unified_clip = {
        "id": clip["id"],
        "filename": clip["filename"],
        "file_path": file_path,
        "source_type": source_type,
        "tags": clip["tags"],
        "duration": clip["duration"],
        "resolution": clip["resolution"],
        "fps": clip["fps"],
        "file_size_mb": clip["file_size_mb"],
        "shot_analysis": clip["shot_analysis"],
        "created_at": clip["created_at"]
    }
    
    return unified_clip

def create_unified_metadata():
    """Create unified metadata file combining both sources"""
    
    print("🔄 Creating unified metadata database...")
    
    # Load both metadata files
    print("📥 Loading MJAnime metadata...")
    mjanime_metadata = load_metadata("MJAnime/mjanime_metadata.json")
    
    print("📥 Loading midjourney composite metadata...")
    mjclip_metadata = load_metadata("midjourney_composite_2025-7-15/mjclip_metadata.json")
    
    # Create unified structure
    unified_metadata = {
        "metadata_info": {
            "created_at": datetime.now().isoformat(),
            "sources": [
                {
                    "name": "MJAnime",
                    "directory": "MJAnime/fixed",
                    "total_clips": mjanime_metadata["metadata_info"]["total_clips"],
                    "original_created": mjanime_metadata["metadata_info"]["created_at"]
                },
                {
                    "name": "midjourney_composite",
                    "directory": "midjourney_composite_2025-7-15",
                    "total_clips": mjclip_metadata["metadata_info"]["total_clips"],
                    "original_created": mjclip_metadata["metadata_info"]["created_at"]
                }
            ],
            "total_clips": len(mjanime_metadata["clips"]) + len(mjclip_metadata["clips"]),
            "generator": "create_unified_metadata.py",
            "version": "1.0.0"
        },
        "clips": []
    }
    
    # Add MJAnime clips
    print(f"🎬 Processing {len(mjanime_metadata['clips'])} MJAnime clips...")
    for clip in mjanime_metadata["clips"]:
        unified_clip = normalize_clip_entry(clip, "mjanime", "MJAnime/fixed")
        unified_metadata["clips"].append(unified_clip)
    
    # Add midjourney composite clips
    print(f"🎬 Processing {len(mjclip_metadata['clips'])} midjourney composite clips...")
    for clip in mjclip_metadata["clips"]:
        unified_clip = normalize_clip_entry(clip, "midjourney_composite", "midjourney_composite_2025-7-15")
        unified_metadata["clips"].append(unified_clip)
    
    # Save unified metadata
    output_file = "unified_clips_metadata.json"
    print(f"💾 Saving unified metadata to {output_file}...")
    
    with open(output_file, 'w') as f:
        json.dump(unified_metadata, f, indent=2)
    
    print(f"✅ Unified metadata created successfully!")
    print(f"📊 Total clips: {unified_metadata['metadata_info']['total_clips']}")
    print(f"   - MJAnime: {mjanime_metadata['metadata_info']['total_clips']} clips")
    print(f"   - Midjourney Composite: {mjclip_metadata['metadata_info']['total_clips']} clips")
    print(f"📁 Output file: {output_file}")

if __name__ == "__main__":
    create_unified_metadata()