#!/usr/bin/env python3
"""
Vibe Content Creation - Main Entry Point
New structured interface for the unified video generation system
"""

import sys
import os
from pathlib import Path

# Add src directory to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Add unified-video-system-main to path for backward compatibility
unified_path = Path(__file__).parent / "unified-video-system-main"
sys.path.insert(0, str(unified_path))

# Import and run the main generator
try:
    from generation.vibe_generator import main
    
    if __name__ == "__main__":
        import asyncio
        asyncio.run(main())
        
except ImportError as e:
    print(f"Error importing vibe generator: {e}")
    print("Falling back to original vibe_generator.py")
    
    # Fallback to original generator
    try:
        from vibe_generator import main
        if __name__ == "__main__":
            import asyncio
            asyncio.run(main())
    except ImportError:
        print("Error: Could not import vibe generator from any location")
        print("Please ensure the system is properly installed")
        sys.exit(1)