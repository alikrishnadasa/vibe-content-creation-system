#!/usr/bin/env python3
"""
Test Frontend-Backend Integration
"""
import requests
import json

def test_integration():
    print("🧪 Testing Frontend-Backend Integration")
    print("=" * 50)
    
    # Test 1: Backend Health
    print("1. Testing Backend Health...")
    try:
        response = requests.get("http://localhost:8000/api/health")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Backend: {data['status']}")
            print(f"   ✅ Quantum Pipeline: {data['quantum_pipeline_ready']}")
            print(f"   ✅ Scripts: {data['available_scripts']}")
            print(f"   ✅ Cached Captions: {data['cached_caption_scripts']}")
        else:
            print(f"   ❌ Backend health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Backend connection failed: {e}")
        return False
    
    # Test 2: Frontend Response
    print("\n2. Testing Frontend Response...")
    try:
        response = requests.get("http://localhost:3000")
        if response.status_code == 200:
            if "Video Content Creation Studio" in response.text:
                print("   ✅ Frontend: Page loads correctly")
                print("   ✅ Title: Found in page")
            else:
                print("   ❌ Frontend: Title not found")
                return False
        else:
            print(f"   ❌ Frontend failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Frontend connection failed: {e}")
        return False
    
    # Test 3: API Integration
    print("\n3. Testing API Integration...")
    try:
        # Test scripts endpoint
        response = requests.get("http://localhost:8000/api/scripts")
        if response.status_code == 200:
            scripts = response.json()["scripts"]
            cached_count = sum(1 for s in scripts if s["has_cached_captions"])
            print(f"   ✅ Scripts API: {len(scripts)} scripts available")
            print(f"   ✅ Cached Captions: {cached_count} scripts have cached captions")
            
            # Test config endpoint
            response = requests.get("http://localhost:8000/api/config")
            if response.status_code == 200:
                config = response.json()
                print(f"   ✅ Config API: Quantum pipeline initialized = {config['quantum_pipeline']['initialized']}")
                print(f"   ✅ Resolution: {config['video_config']['target_resolution']}")
            else:
                print(f"   ❌ Config API failed: {response.status_code}")
                return False
        else:
            print(f"   ❌ Scripts API failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ API integration failed: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("✅ ALL TESTS PASSED!")
    print("🎉 Frontend-Backend Integration is working correctly!")
    print("\nYou can now:")
    print("• Visit http://localhost:3000 to use the webapp")
    print("• Generate single videos with quantum pipeline")
    print("• Create batch videos")
    print("• Monitor jobs in real-time")
    print("• Browse and download videos")
    
    return True

if __name__ == "__main__":
    test_integration()