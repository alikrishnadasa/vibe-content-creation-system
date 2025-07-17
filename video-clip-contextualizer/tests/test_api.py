import pytest
import asyncio
import tempfile
import os
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import json

from src.api.main import app
from src.processing.video_processor import VideoSegment
from src.processing.text_processor import TextSegment
from src.matching.semantic_matcher import VideoTextMatch


class TestAPI:
    
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    @pytest.fixture
    def sample_video_file(self):
        """Create a dummy video file for testing."""
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
            tmp_file.write(b"fake video content")
            yield tmp_file.name
        
        if os.path.exists(tmp_file.name):
            os.unlink(tmp_file.name)
    
    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "name" in data
        assert "version" in data
        assert "description" in data
        assert "endpoints" in data
    
    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/api/v1/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data
    
    def test_get_model_info(self, client):
        """Test model info endpoint."""
        with patch('src.api.main.semantic_matcher') as mock_matcher:
            mock_matcher.clip_encoder.get_model_info.return_value = {
                "model_name": "test_model",
                "device": "cpu",
                "embedding_dim": 512
            }
            
            response = client.get("/api/v1/models")
            
            assert response.status_code == 200
            data = response.json()
            
            assert "clip_encoder" in data
            assert "text_processor" in data
            assert "video_processor" in data
            assert "system_config" in data
    
    def test_get_metrics(self, client):
        """Test metrics endpoint."""
        response = client.get("/api/v1/metrics")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "system" in data
        assert "api" in data
        assert "cpu_usage" in data["system"]
        assert "memory_usage" in data["system"]
        assert "disk_usage" in data["system"]
    
    @patch('src.api.main.video_processor')
    @patch('src.api.main.text_processor')
    @patch('src.api.main.semantic_matcher')
    def test_analyze_video_with_file(self, mock_matcher, mock_text_processor, 
                                   mock_video_processor, client, sample_video_file):
        """Test video analysis with file upload."""
        # Mock video processing
        mock_video_segments = [
            VideoSegment(0.0, 2.0, None, 30.0, 2.0),
            VideoSegment(1.5, 3.5, None, 30.0, 2.0)
        ]
        mock_video_processor.process_video_file.return_value = mock_video_segments
        
        # Mock text processing
        mock_text_segments = [
            TextSegment("Sample text", 0, 11, None, ["sample", "text"])
        ]
        mock_text_processor.segment_text.return_value = mock_text_segments
        mock_text_processor.generate_embeddings.return_value = mock_text_segments
        
        # Mock matching
        mock_matches = [
            VideoTextMatch(
                video_segment_idx=0,
                text_segment_idx=0,
                video_start=0.0,
                video_end=2.0,
                text_start=0,
                text_end=11,
                confidence=0.85,
                explanation={
                    "match_type": "high_confidence",
                    "video_duration": 2.0,
                    "text_length": 11,
                    "text_preview": "Sample text",
                    "keywords": ["sample", "text"],
                    "temporal_alignment": {"video_position": 0.0, "text_position": 0.0}
                }
            )
        ]
        mock_matcher.match_video_to_text.return_value = mock_matches
        
        # Test request
        with open(sample_video_file, 'rb') as f:
            response = client.post(
                "/api/v1/analyze",
                files={"video_file": ("test.mp4", f, "video/mp4")},
                data={
                    "script": "Sample text for analysis",
                    "clip_duration": "2.0",
                    "overlap": "0.5",
                    "language": "en",
                    "matching_strategy": "optimal"
                }
            )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "matches" in data
        assert "processing_time" in data
        assert "job_id" in data
        assert "metadata" in data
        
        assert len(data["matches"]) == 1
        assert data["matches"][0]["confidence"] == 0.85
    
    def test_analyze_video_without_file_or_url(self, client):
        """Test video analysis without file or URL."""
        response = client.post(
            "/api/v1/analyze",
            json={
                "script": "Sample text for analysis",
                "clip_duration": 2.0,
                "overlap": 0.5,
                "language": "en"
            }
        )
        
        assert response.status_code == 400
        assert "Either video_file or video_url must be provided" in response.json()["detail"]
    
    def test_analyze_video_with_url_not_implemented(self, client):
        """Test video analysis with URL (not implemented)."""
        response = client.post(
            "/api/v1/analyze",
            json={
                "video_url": "https://example.com/video.mp4",
                "script": "Sample text for analysis",
                "clip_duration": 2.0,
                "overlap": 0.5,
                "language": "en"
            }
        )
        
        assert response.status_code == 501
        assert "Video URL download not implemented" in response.json()["detail"]
    
    @patch('src.api.main.video_processor')
    def test_analyze_video_processing_error(self, mock_video_processor, client, sample_video_file):
        """Test video analysis with processing error."""
        mock_video_processor.process_video_file.side_effect = Exception("Processing failed")
        
        with open(sample_video_file, 'rb') as f:
            response = client.post(
                "/api/v1/analyze",
                files={"video_file": ("test.mp4", f, "video/mp4")},
                data={"script": "Sample text"}
            )
        
        assert response.status_code == 500
        assert "Processing error" in response.json()["detail"]
    
    def test_batch_analyze(self, client):
        """Test batch analysis endpoint."""
        batch_request = {
            "requests": [
                {
                    "video_url": "https://example.com/video1.mp4",
                    "script": "First video script",
                    "clip_duration": 2.0,
                    "overlap": 0.5,
                    "language": "en"
                },
                {
                    "video_url": "https://example.com/video2.mp4",
                    "script": "Second video script",
                    "clip_duration": 3.0,
                    "overlap": 0.5,
                    "language": "en"
                }
            ]
        }
        
        response = client.post("/api/v1/batch", json=batch_request)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "job_id" in data
        assert "status" in data
        assert "batch_size" in data
        assert data["batch_size"] == 2
        assert data["status"] == "queued"
    
    def test_get_job_status_not_found(self, client):
        """Test getting status of non-existent job."""
        response = client.get("/api/v1/job/nonexistent-job-id")
        
        assert response.status_code == 404
        assert "Job not found" in response.json()["detail"]
    
    def test_get_job_status_existing(self, client):
        """Test getting status of existing job."""
        # First create a job
        batch_request = {
            "requests": [
                {
                    "video_url": "https://example.com/video.mp4",
                    "script": "Test script",
                    "language": "en"
                }
            ]
        }
        
        batch_response = client.post("/api/v1/batch", json=batch_request)
        job_id = batch_response.json()["job_id"]
        
        # Then get its status
        response = client.get(f"/api/v1/job/{job_id}")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "job_id" in data
        assert "status" in data
        assert "progress" in data
        assert "created_at" in data
        assert "updated_at" in data
    
    def test_cors_headers(self, client):
        """Test CORS headers are present."""
        response = client.options("/api/v1/health")
        
        # FastAPI handles CORS automatically with the middleware
        assert response.status_code == 200
    
    def test_request_validation(self, client):
        """Test request validation for analyze endpoint."""
        # Missing required script field
        response = client.post(
            "/api/v1/analyze",
            json={
                "clip_duration": 2.0,
                "overlap": 0.5,
                "language": "en"
            }
        )
        
        assert response.status_code == 422  # Validation error
    
    def test_analyze_video_with_custom_config(self, client, sample_video_file):
        """Test video analysis with custom configuration."""
        with patch('src.api.main.video_processor') as mock_video_processor, \
             patch('src.api.main.text_processor') as mock_text_processor, \
             patch('src.api.main.semantic_matcher') as mock_matcher:
            
            # Mock successful processing
            mock_video_processor.process_video_file.return_value = [
                VideoSegment(0.0, 3.0, None, 30.0, 3.0)
            ]
            mock_text_processor.segment_text.return_value = [
                TextSegment("Test", 0, 4, None, ["test"])
            ]
            mock_text_processor.generate_embeddings.return_value = [
                TextSegment("Test", 0, 4, None, ["test"])
            ]
            mock_matcher.match_video_to_text.return_value = []
            
            with open(sample_video_file, 'rb') as f:
                response = client.post(
                    "/api/v1/analyze",
                    files={"video_file": ("test.mp4", f, "video/mp4")},
                    data={
                        "script": "Test script",
                        "clip_duration": "3.0",
                        "overlap": "1.0",
                        "matching_strategy": "greedy"
                    }
                )
            
            assert response.status_code == 200
            
            # Verify custom configuration was applied
            assert mock_video_processor.clip_duration == 3.0
            assert mock_video_processor.overlap == 1.0
            
            mock_matcher.match_video_to_text.assert_called_with(
                mock_video_processor.process_video_file.return_value,
                mock_text_processor.generate_embeddings.return_value,
                "greedy"
            )
    
    def test_response_format(self, client, sample_video_file):
        """Test response format matches API specification."""
        with patch('src.api.main.video_processor') as mock_video_processor, \
             patch('src.api.main.text_processor') as mock_text_processor, \
             patch('src.api.main.semantic_matcher') as mock_matcher:
            
            # Mock processing with specific return values
            mock_video_processor.process_video_file.return_value = [
                VideoSegment(0.0, 2.0, None, 30.0, 2.0)
            ]
            mock_text_processor.segment_text.return_value = [
                TextSegment("Test text", 0, 9, None, ["test", "text"])
            ]
            mock_text_processor.generate_embeddings.return_value = [
                TextSegment("Test text", 0, 9, None, ["test", "text"])
            ]
            mock_matcher.match_video_to_text.return_value = [
                VideoTextMatch(
                    video_segment_idx=0,
                    text_segment_idx=0,
                    video_start=0.0,
                    video_end=2.0,
                    text_start=0,
                    text_end=9,
                    confidence=0.75,
                    explanation={
                        "match_type": "medium_confidence",
                        "video_duration": 2.0,
                        "text_length": 9,
                        "text_preview": "Test text",
                        "keywords": ["test", "text"],
                        "temporal_alignment": {"video_position": 0.0, "text_position": 0.0}
                    }
                )
            ]
            
            with open(sample_video_file, 'rb') as f:
                response = client.post(
                    "/api/v1/analyze",
                    files={"video_file": ("test.mp4", f, "video/mp4")},
                    data={"script": "Test text"}
                )
            
            assert response.status_code == 200
            data = response.json()
            
            # Check response structure matches API spec
            assert "matches" in data
            assert "processing_time" in data
            assert "job_id" in data
            assert "metadata" in data
            
            # Check match structure
            match = data["matches"][0]
            assert "video_segment" in match
            assert "script_segment" in match
            assert "confidence" in match
            assert "explanation" in match
            
            # Check video segment structure
            video_segment = match["video_segment"]
            assert "start" in video_segment
            assert "end" in video_segment
            
            # Check script segment structure
            script_segment = match["script_segment"]
            assert "start" in script_segment
            assert "end" in script_segment
            
            # Check explanation structure
            explanation = match["explanation"]
            assert "match_type" in explanation
            assert "video_duration" in explanation
            assert "text_length" in explanation
            assert "text_preview" in explanation
            assert "keywords" in explanation
            assert "temporal_alignment" in explanation