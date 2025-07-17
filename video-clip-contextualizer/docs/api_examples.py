#!/usr/bin/env python3
"""
Video Clip Contextualizer API Examples

This file contains practical examples of using the Video Clip Contextualizer API
for various use cases and scenarios.
"""

import requests
import json
import time
from typing import Dict, Any, List


class VideoContextualizerAPI:
    """Client wrapper for the Video Clip Contextualizer API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
    
    def analyze_video_file(self, video_path: str, script: str, **kwargs) -> Dict[str, Any]:
        """
        Analyze a local video file with a script.
        
        Args:
            video_path: Path to video file
            script: Text script to match
            **kwargs: Additional parameters (clip_duration, overlap, etc.)
        
        Returns:
            Analysis results
        """
        url = f"{self.base_url}/api/v1/analyze"
        
        # Prepare form data
        data = {
            "script": script,
            "clip_duration": kwargs.get("clip_duration", 5.0),
            "overlap": kwargs.get("overlap", 0.5),
            "language": kwargs.get("language", "en"),
            "matching_strategy": kwargs.get("matching_strategy", "optimal")
        }
        
        # Upload file
        with open(video_path, 'rb') as video_file:
            files = {"video_file": video_file}
            response = self.session.post(url, data=data, files=files)
        
        response.raise_for_status()
        return response.json()
    
    def analyze_video_url(self, video_url: str, script: str, **kwargs) -> Dict[str, Any]:
        """
        Analyze a video from URL with a script.
        
        Args:
            video_url: URL to video file
            script: Text script to match
            **kwargs: Additional parameters
        
        Returns:
            Analysis results
        """
        url = f"{self.base_url}/api/v1/analyze"
        
        payload = {
            "video_url": video_url,
            "script": script,
            "clip_duration": kwargs.get("clip_duration", 5.0),
            "overlap": kwargs.get("overlap", 0.5),
            "language": kwargs.get("language", "en"),
            "matching_strategy": kwargs.get("matching_strategy", "optimal")
        }
        
        response = self.session.post(url, json=payload)
        response.raise_for_status()
        return response.json()
    
    def batch_analyze(self, requests_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Submit multiple analysis requests as a batch.
        
        Args:
            requests_list: List of analysis request dictionaries
        
        Returns:
            Batch job information
        """
        url = f"{self.base_url}/api/v1/batch"
        
        payload = {"requests": requests_list}
        response = self.session.post(url, json=payload)
        response.raise_for_status()
        return response.json()
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get status of a processing job."""
        url = f"{self.base_url}/api/v1/job/{job_id}"
        response = self.session.get(url)
        response.raise_for_status()
        return response.json()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models."""
        url = f"{self.base_url}/api/v1/models"
        response = self.session.get(url)
        response.raise_for_status()
        return response.json()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get system performance metrics."""
        url = f"{self.base_url}/api/v1/metrics"
        response = self.session.get(url)
        response.raise_for_status()
        return response.json()
    
    def health_check(self) -> bool:
        """Check if the API is healthy."""
        try:
            url = f"{self.base_url}/api/v1/health"
            response = self.session.get(url, timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False


def example_basic_analysis():
    """Example: Basic video analysis with a simple script."""
    
    api = VideoContextualizerAPI()
    
    script = """
    A person walks into a coffee shop and orders a drink.
    The barista prepares the coffee behind the counter.
    The customer pays and receives their order.
    """
    
    try:
        # Analyze video (replace with actual video path)
        result = api.analyze_video_file(
            video_path="examples/coffee_shop.mp4",
            script=script,
            clip_duration=3.0,
            overlap=0.5
        )
        
        print("Analysis Results:")
        print(f"Processing time: {result['processing_time']:.2f}s")
        print(f"Found {len(result['matches'])} matches")
        
        for i, match in enumerate(result['matches']):
            print(f"\nMatch {i+1}:")
            print(f"  Video: {match['video_segment']['start']:.1f}s - {match['video_segment']['end']:.1f}s")
            print(f"  Text: chars {match['script_segment']['start']}-{match['script_segment']['end']}")
            print(f"  Confidence: {match['confidence']:.3f}")
            print(f"  Match type: {match['explanation']['match_type']}")
        
    except requests.RequestException as e:
        print(f"API request failed: {e}")
    except FileNotFoundError:
        print("Video file not found. Please provide a valid video path.")


def example_batch_processing():
    """Example: Process multiple videos in batch."""
    
    api = VideoContextualizerAPI()
    
    # Prepare batch requests
    batch_requests = [
        {
            "video_url": "https://example.com/video1.mp4",
            "script": "A person cooking in the kitchen",
            "clip_duration": 4.0,
            "matching_strategy": "optimal"
        },
        {
            "video_url": "https://example.com/video2.mp4", 
            "script": "Children playing in the park",
            "clip_duration": 6.0,
            "matching_strategy": "greedy"
        },
        {
            "video_url": "https://example.com/video3.mp4",
            "script": "Business meeting in conference room",
            "clip_duration": 5.0,
            "matching_strategy": "threshold"
        }
    ]
    
    try:
        # Submit batch
        batch_result = api.batch_analyze(batch_requests)
        job_id = batch_result["job_id"]
        
        print(f"Batch submitted with job ID: {job_id}")
        print(f"Batch size: {batch_result['batch_size']}")
        
        # Monitor progress
        while True:
            status = api.get_job_status(job_id)
            print(f"Status: {status['status']}, Progress: {status['progress']:.1%}")
            
            if status['status'] in ['completed', 'failed']:
                break
            
            time.sleep(2)
        
        if status['status'] == 'completed':
            print("Batch processing completed successfully!")
            if 'result' in status:
                print(f"Results available in job status response")
        else:
            print(f"Batch processing failed: {status.get('error', 'Unknown error')}")
    
    except requests.RequestException as e:
        print(f"API request failed: {e}")


def example_educational_content():
    """Example: Analyze educational content with detailed matching."""
    
    api = VideoContextualizerAPI()
    
    script = """
    Welcome to our lesson on photosynthesis.
    First, let's examine the structure of a leaf under the microscope.
    Notice the chloroplasts, which contain chlorophyll.
    Light energy is captured by these green pigments.
    Carbon dioxide enters through tiny pores called stomata.
    Water is absorbed by the roots and transported to the leaves.
    The chemical reaction combines CO2 and water using light energy.
    This process produces glucose and releases oxygen.
    The glucose provides energy for the plant's growth.
    """
    
    try:
        result = api.analyze_video_file(
            video_path="examples/photosynthesis_lesson.mp4",
            script=script,
            clip_duration=8.0,  # Longer clips for educational content
            overlap=1.0,        # More overlap for better continuity
            matching_strategy="optimal"
        )
        
        print("Educational Content Analysis:")
        print(f"Total video segments: {result['metadata']['video_segments_count']}")
        print(f"Total text segments: {result['metadata']['text_segments_count']}")
        print(f"Average confidence: {result['metadata']['avg_confidence']:.3f}")
        
        # Group matches by confidence level
        high_confidence = [m for m in result['matches'] if m['confidence'] >= 0.8]
        medium_confidence = [m for m in result['matches'] if 0.5 <= m['confidence'] < 0.8]
        low_confidence = [m for m in result['matches'] if m['confidence'] < 0.5]
        
        print(f"\nConfidence Distribution:")
        print(f"  High confidence (≥0.8): {len(high_confidence)} matches")
        print(f"  Medium confidence (0.5-0.8): {len(medium_confidence)} matches")
        print(f"  Low confidence (<0.5): {len(low_confidence)} matches")
        
        # Show best matches
        print(f"\nTop 3 Matches:")
        sorted_matches = sorted(result['matches'], key=lambda x: x['confidence'], reverse=True)
        for i, match in enumerate(sorted_matches[:3]):
            explanation = match['explanation']
            print(f"\n{i+1}. Video {match['video_segment']['start']:.1f}s-{match['video_segment']['end']:.1f}s")
            print(f"   Text: \"{explanation['text_preview']}\"")
            print(f"   Confidence: {match['confidence']:.3f}")
            print(f"   Keywords: {', '.join(explanation['keywords'][:5])}")
    
    except requests.RequestException as e:
        print(f"API request failed: {e}")
    except FileNotFoundError:
        print("Video file not found. Please provide a valid video path.")


def example_marketing_content():
    """Example: Analyze marketing video with product demonstrations."""
    
    api = VideoContextualizerAPI()
    
    script = """
    Introducing the revolutionary SmartWatch Pro.
    Check the time with a simple glance at your wrist.
    Track your fitness goals with built-in heart rate monitoring.
    Receive notifications from your smartphone instantly.
    The waterproof design means you can wear it anywhere.
    Swimming, running, or just daily activities - it handles everything.
    The battery lasts up to 7 days on a single charge.
    Choose from five stunning colors to match your style.
    Order now and get free shipping worldwide.
    """
    
    try:
        result = api.analyze_video_file(
            video_path="examples/smartwatch_ad.mp4",
            script=script,
            clip_duration=4.0,  # Shorter clips for dynamic marketing content
            overlap=0.3,
            matching_strategy="greedy"  # Capture multiple product shots
        )
        
        print("Marketing Content Analysis:")
        
        # Analyze product features mentioned
        feature_keywords = ['time', 'fitness', 'heart rate', 'notifications', 'waterproof', 'battery', 'colors']
        feature_matches = {}
        
        for match in result['matches']:
            explanation = match['explanation']
            for keyword in feature_keywords:
                if keyword in explanation.get('keywords', []):
                    if keyword not in feature_matches:
                        feature_matches[keyword] = []
                    feature_matches[keyword].append({
                        'video_time': f"{match['video_segment']['start']:.1f}s-{match['video_segment']['end']:.1f}s",
                        'confidence': match['confidence']
                    })
        
        print("\nProduct Features Coverage:")
        for feature, matches in feature_matches.items():
            print(f"  {feature.title()}: {len(matches)} clips")
            for match in matches:
                print(f"    - {match['video_time']} (confidence: {match['confidence']:.3f})")
        
        # Identify potential call-to-action moments
        cta_keywords = ['order', 'buy', 'purchase', 'free shipping']
        cta_matches = []
        
        for match in result['matches']:
            explanation = match['explanation']
            if any(keyword in explanation.get('keywords', []) for keyword in cta_keywords):
                cta_matches.append(match)
        
        print(f"\nCall-to-Action Moments: {len(cta_matches)}")
        for match in cta_matches:
            print(f"  - {match['video_segment']['start']:.1f}s-{match['video_segment']['end']:.1f}s")
            print(f"    \"{match['explanation']['text_preview']}\"")
    
    except requests.RequestException as e:
        print(f"API request failed: {e}")
    except FileNotFoundError:
        print("Video file not found. Please provide a valid video path.")


def example_performance_monitoring():
    """Example: Monitor system performance and optimization."""
    
    api = VideoContextualizerAPI()
    
    try:
        # Check system health
        if not api.health_check():
            print("API is not healthy!")
            return
        
        # Get model information
        model_info = api.get_model_info()
        print("Loaded Models:")
        print(f"  Video encoder: {model_info['clip_encoder']['model_name']}")
        print(f"  Device: {model_info['clip_encoder']['device']}")
        print(f"  Embedding dimension: {model_info['clip_encoder']['embedding_dim']}")
        
        # Get current metrics
        metrics = api.get_metrics()
        system_metrics = metrics['system']
        
        print(f"\nSystem Performance:")
        print(f"  CPU usage: {system_metrics['cpu_usage']:.1f}%")
        print(f"  Memory usage: {system_metrics['memory_usage']:.1f}%")
        print(f"  Disk usage: {system_metrics['disk_usage']:.1f}%")
        print(f"  Active jobs: {metrics['api']['active_jobs']}")
        
        # Performance recommendations
        if system_metrics['memory_usage'] > 80:
            print("\n⚠️  Warning: High memory usage detected")
            print("   Consider reducing batch size or clip duration")
        
        if system_metrics['cpu_usage'] > 90:
            print("\n⚠️  Warning: High CPU usage detected")
            print("   Consider scaling horizontally or optimizing processing")
        
        print(f"\nAPI Status: ✅ Healthy")
    
    except requests.RequestException as e:
        print(f"API request failed: {e}")


def example_custom_matching_strategies():
    """Example: Compare different matching strategies."""
    
    api = VideoContextualizerAPI()
    
    script = "A chef prepares a delicious pasta dish in a modern kitchen."
    strategies = ["optimal", "greedy", "threshold"]
    
    try:
        results = {}
        
        for strategy in strategies:
            print(f"\nTesting {strategy} matching strategy...")
            
            result = api.analyze_video_file(
                video_path="examples/cooking_demo.mp4",
                script=script,
                clip_duration=5.0,
                matching_strategy=strategy
            )
            
            results[strategy] = result
            
            print(f"  Matches found: {len(result['matches'])}")
            print(f"  Processing time: {result['processing_time']:.3f}s")
            print(f"  Average confidence: {result['metadata']['avg_confidence']:.3f}")
        
        # Compare strategies
        print(f"\nStrategy Comparison:")
        print(f"{'Strategy':<12} {'Matches':<8} {'Avg Conf':<10} {'Time (s)':<10}")
        print("-" * 42)
        
        for strategy in strategies:
            result = results[strategy]
            print(f"{strategy:<12} {len(result['matches']):<8} "
                  f"{result['metadata']['avg_confidence']:<10.3f} "
                  f"{result['processing_time']:<10.3f}")
        
        # Recommend best strategy
        best_strategy = max(strategies, key=lambda s: results[s]['metadata']['avg_confidence'])
        print(f"\n🏆 Best strategy for this content: {best_strategy}")
    
    except requests.RequestException as e:
        print(f"API request failed: {e}")
    except FileNotFoundError:
        print("Video file not found. Please provide a valid video path.")


if __name__ == "__main__":
    print("Video Clip Contextualizer API Examples")
    print("=" * 50)
    
    # Test API availability
    api = VideoContextualizerAPI()
    if not api.health_check():
        print("❌ API is not available. Please start the server first:")
        print("   python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000")
        exit(1)
    
    print("✅ API is available\n")
    
    # Run examples (comment out as needed)
    
    # Basic usage
    print("1. Basic Analysis Example:")
    example_basic_analysis()
    
    # Batch processing
    print("\n2. Batch Processing Example:")
    example_batch_processing()
    
    # Educational content
    print("\n3. Educational Content Example:")
    example_educational_content()
    
    # Marketing content
    print("\n4. Marketing Content Example:")
    example_marketing_content()
    
    # Performance monitoring
    print("\n5. Performance Monitoring Example:")
    example_performance_monitoring()
    
    # Strategy comparison
    print("\n6. Matching Strategies Example:")
    example_custom_matching_strategies()
    
    print("\n" + "=" * 50)
    print("Examples completed! Check the output above for results.")