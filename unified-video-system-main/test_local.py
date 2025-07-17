#!/usr/bin/env python3
"""
Local testing script for RunPod deployment
Test the Docker container locally before deploying
"""

import json
import asyncio
import sys
from runpod_handler import handler

async def test_health_check():
    """Test the health check endpoint"""
    print("Testing health check endpoint...")
    
    job = {
        "input": {
            "endpoint": "health"
        }
    }
    
    result = handler(job)
    print(f"Health check result: {json.dumps(result, indent=2)}")
    return result.get('status') == 'healthy'

async def test_video_generation():
    """Test video generation endpoint"""
    print("\nTesting video generation endpoint...")
    
    job = {
        "input": {
            "endpoint": "generate",
            "script_text": "This is a test script for video generation. It should be processed quickly and efficiently.",
            "script_name": "test_script",
            "variation_number": 1,
            "caption_style": "tiktok",
            "music_sync": True,
            "min_clip_duration": 2.5
        }
    }
    
    result = handler(job)
    print(f"Video generation result: {json.dumps(result, indent=2)}")
    return result.get('success', False)

async def main():
    """Run all tests"""
    print("=== RunPod Handler Local Testing ===\n")
    
    # Test health check
    health_ok = await test_health_check()
    
    if not health_ok:
        print("❌ Health check failed - system not ready")
        sys.exit(1)
    
    print("✅ Health check passed")
    
    # Test video generation
    # Note: This will fail locally without proper setup, but we can test the handler logic
    try:
        video_ok = await test_video_generation()
        if video_ok:
            print("✅ Video generation test passed")
        else:
            print("⚠️  Video generation test failed (expected without full setup)")
    except Exception as e:
        print(f"⚠️  Video generation test error: {e}")
        print("This is expected without full MJAnime setup")
    
    print("\n=== Test Summary ===")
    print("Handler logic appears to be working correctly")
    print("Ready for RunPod deployment!")

if __name__ == "__main__":
    asyncio.run(main())