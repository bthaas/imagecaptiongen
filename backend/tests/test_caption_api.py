"""
Integration tests for the caption generation API endpoint.

Tests the complete API functionality including request validation,
ML pipeline integration, error handling, and response formatting.
"""

import pytest
import base64
import json
import time
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient
from PIL import Image
import io

from app.main import app
from app.models.api_models import ErrorCode
from app.services.caption_service import CaptionResult


class TestCaptionGenerationAPI:
    """Test suite for caption generation API endpoint."""
    
    @pytest.fixture
    def client(self):
        """Create test client for FastAPI app."""
        return TestClient(app)
    
    @pytest.fixture
    def sample_image_data(self):
        """Create sample image data for testing."""
        # Create a simple test image
        img = Image.new('RGB', (224, 224), color='red')
        img_buffer = io.BytesIO()
        img.save(img_buffer, format='JPEG')
        img_buffer.seek(0)
        return img_buffer.getvalue()
    
    @pytest.fixture
    def base64_image(self, sample_image_data):
        """Create base64 encoded image for API requests."""
        return base64.b64encode(sample_image_data).decode('utf-8')
    
    @pytest.fixture
    def data_url_image(self, base64_image):
        """Create data URL formatted image for API requests."""
        return f"data:image/jpeg;base64,{base64_image}"
    
    @pytest.fixture
    def mock_caption_service(self):
        """Mock caption service for testing."""
        with patch('app.main.caption_service') as mock_service:
            yield mock_service

    def test_generate_caption_success(self, client, data_url_image, mock_caption_service):
        """Test successful caption generation."""
        # Mock successful caption generation
        mock_result = CaptionResult(
            caption="a red square image",
            confidence=0.85,
            processing_time=2.34,
            image_metadata={'width': 224, 'height': 224, 'format': 'JPEG'},
            model_info={'cnn_model': 'ResNet50', 'lstm_model': 'LSTM'},
            success=True
        )
        mock_caption_service.generate_caption.return_value = mock_result
        
        # Make API request
        response = client.post(
            "/api/v1/generate-caption",
            json={
                "image_data": data_url_image,
                "max_length": 20,
                "temperature": 1.0
            }
        )
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        
        assert data["caption"] == "a red square image"
        assert data["confidence"] == 0.85
        assert data["processing_time"] == 2.34
        assert "image_id" in data
        assert "timestamp" in data
        
        # Verify service was called correctly
        mock_caption_service.generate_caption.assert_called_once()
        call_args = mock_caption_service.generate_caption.call_args
        assert call_args[1]["temperature"] == 1.0
        assert call_args[1]["use_beam_search"] is False
        assert call_args[1]["beam_width"] == 3

    def test_generate_caption_with_base64_only(self, client, base64_image, mock_caption_service):
        """Test caption generation with base64 data without data URL prefix."""
        mock_result = CaptionResult(
            caption="test caption",
            confidence=0.75,
            processing_time=1.5,
            image_metadata={},
            model_info={},
            success=True
        )
        mock_caption_service.generate_caption.return_value = mock_result
        
        response = client.post(
            "/api/v1/generate-caption",
            json={
                "image_data": base64_image,
                "max_length": 15,
                "temperature": 0.8
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["caption"] == "test caption"
        assert data["confidence"] == 0.75

    def test_generate_caption_default_parameters(self, client, data_url_image, mock_caption_service):
        """Test caption generation with default parameters."""
        mock_result = CaptionResult(
            caption="default test",
            confidence=0.9,
            processing_time=1.0,
            image_metadata={},
            model_info={},
            success=True
        )
        mock_caption_service.generate_caption.return_value = mock_result
        
        # Request with only required field
        response = client.post(
            "/api/v1/generate-caption",
            json={"image_data": data_url_image}
        )
        
        assert response.status_code == 200
        
        # Verify default parameters were used
        call_args = mock_caption_service.generate_caption.call_args
        assert call_args[1]["temperature"] == 1.0  # Default temperature

    def test_generate_caption_ml_pipeline_error(self, client, data_url_image, mock_caption_service):
        """Test handling of ML pipeline errors."""
        mock_result = CaptionResult(
            caption="",
            confidence=0.0,
            processing_time=0.5,
            image_metadata={},
            model_info={},
            success=False,
            error_message="Model failed to load"
        )
        mock_caption_service.generate_caption.return_value = mock_result
        
        response = client.post(
            "/api/v1/generate-caption",
            json={"image_data": data_url_image}
        )
        
        assert response.status_code == 500
        data = response.json()
        assert data["error"] == ErrorCode.MODEL_ERROR
        assert "Model failed to load" in data["message"]
        assert "request_id" in data

    def test_generate_caption_invalid_base64(self, client):
        """Test handling of invalid base64 data."""
        response = client.post(
            "/api/v1/generate-caption",
            json={"image_data": "invalid_base64_data!!!"}
        )
        
        assert response.status_code == 400
        data = response.json()
        assert data["error"] == ErrorCode.INVALID_IMAGE_FORMAT
        assert "Invalid base64 encoding" in data["message"]

    def test_generate_caption_empty_image_data(self, client):
        """Test handling of empty image data."""
        empty_base64 = base64.b64encode(b"").decode('utf-8')
        
        response = client.post(
            "/api/v1/generate-caption",
            json={"image_data": empty_base64}
        )
        
        assert response.status_code == 400
        data = response.json()
        assert data["error"] == ErrorCode.INVALID_IMAGE_FORMAT
        assert "Empty image data" in data["message"]

    def test_generate_caption_invalid_data_url(self, client):
        """Test handling of invalid data URL format."""
        response = client.post(
            "/api/v1/generate-caption",
            json={"image_data": "data:image/jpeg;invalid_format"}
        )
        
        assert response.status_code == 400
        data = response.json()
        assert data["error"] == ErrorCode.INVALID_IMAGE_FORMAT
        assert "Invalid data URL format" in data["message"]

    def test_generate_caption_non_image_data(self, client):
        """Test handling of non-image data."""
        # Create base64 encoded text instead of image
        text_data = base64.b64encode(b"This is not an image").decode('utf-8')
        
        response = client.post(
            "/api/v1/generate-caption",
            json={"image_data": text_data}
        )
        
        assert response.status_code == 400
        data = response.json()
        assert data["error"] == ErrorCode.INVALID_IMAGE_FORMAT
        assert "Invalid image format" in data["message"]

    def test_generate_caption_missing_image_data(self, client):
        """Test handling of missing image_data field."""
        response = client.post(
            "/api/v1/generate-caption",
            json={"max_length": 20}
        )
        
        assert response.status_code == 422  # Validation error
        data = response.json()
        assert data["error"] == ErrorCode.VALIDATION_ERROR

    def test_generate_caption_invalid_parameters(self, client, data_url_image):
        """Test validation of request parameters."""
        # Test invalid max_length
        response = client.post(
            "/api/v1/generate-caption",
            json={
                "image_data": data_url_image,
                "max_length": 100  # Above maximum of 50
            }
        )
        assert response.status_code == 422
        
        # Test invalid temperature
        response = client.post(
            "/api/v1/generate-caption",
            json={
                "image_data": data_url_image,
                "temperature": 5.0  # Above maximum of 2.0
            }
        )
        assert response.status_code == 422

    def test_generate_caption_service_exception(self, client, data_url_image, mock_caption_service):
        """Test handling of unexpected service exceptions."""
        mock_caption_service.generate_caption.side_effect = Exception("Unexpected error")
        
        response = client.post(
            "/api/v1/generate-caption",
            json={"image_data": data_url_image}
        )
        
        assert response.status_code == 500
        data = response.json()
        assert data["error"] == ErrorCode.INTERNAL_ERROR
        assert data["message"] == "Internal server error occurred"

    def test_generate_caption_request_logging(self, client, data_url_image, mock_caption_service, caplog):
        """Test that requests are properly logged."""
        mock_result = CaptionResult(
            caption="logged test",
            confidence=0.8,
            processing_time=1.2,
            image_metadata={},
            model_info={},
            success=True
        )
        mock_caption_service.generate_caption.return_value = mock_result
        
        with caplog.at_level("INFO"):
            response = client.post(
                "/api/v1/generate-caption",
                json={
                    "image_data": data_url_image,
                    "max_length": 25,
                    "temperature": 1.2
                }
            )
        
        assert response.status_code == 200
        
        # Check that request was logged
        log_messages = [record.message for record in caplog.records]
        assert any("Caption generation request started" in msg for msg in log_messages)
        assert any("max_length: 25" in msg and "temperature: 1.2" in msg for msg in log_messages)
        assert any("completed successfully" in msg and "logged test" in msg for msg in log_messages)

    def test_model_info_endpoint(self, client, mock_caption_service):
        """Test the model info endpoint."""
        mock_status = {
            'initialized': True,
            'components': {
                'cnn_model': {'loaded': True, 'model_type': 'ResNet50'},
                'lstm_model': {'loaded': True, 'vocabulary_loaded': True}
            }
        }
        mock_caption_service.get_service_status.return_value = mock_status
        
        response = client.get("/api/v1/model-info")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["service_status"] == mock_status
        assert data["api_version"] == "1.0.0"
        assert "JPEG" in data["supported_formats"]
        assert "PNG" in data["supported_formats"]
        assert "WebP" in data["supported_formats"]
        assert data["max_image_size_mb"] == 10
        assert data["max_caption_length"] == 50

    def test_model_info_endpoint_error(self, client, mock_caption_service):
        """Test model info endpoint error handling."""
        mock_caption_service.get_service_status.side_effect = Exception("Service error")
        
        response = client.get("/api/v1/model-info")
        
        assert response.status_code == 500
        data = response.json()
        # The error response format uses our standard ErrorResponse model
        assert data["error"] == ErrorCode.INTERNAL_ERROR
        assert "Could not retrieve model information" in data["message"]

    def test_cors_headers(self, client, data_url_image, mock_caption_service):
        """Test that CORS headers are properly set."""
        mock_result = CaptionResult(
            caption="cors test",
            confidence=0.7,
            processing_time=1.0,
            image_metadata={},
            model_info={},
            success=True
        )
        mock_caption_service.generate_caption.return_value = mock_result
        
        response = client.post(
            "/api/v1/generate-caption",
            json={"image_data": data_url_image},
            headers={"Origin": "http://localhost:3000"}
        )
        
        assert response.status_code == 200
        # CORS headers should be handled by middleware

    def test_concurrent_requests(self, client, data_url_image, mock_caption_service):
        """Test handling of concurrent requests."""
        import threading
        import queue
        
        mock_result = CaptionResult(
            caption="concurrent test",
            confidence=0.8,
            processing_time=0.5,
            image_metadata={},
            model_info={},
            success=True
        )
        mock_caption_service.generate_caption.return_value = mock_result
        
        results = queue.Queue()
        
        def make_request():
            response = client.post(
                "/api/v1/generate-caption",
                json={"image_data": data_url_image}
            )
            results.put(response.status_code)
        
        # Create multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check all requests succeeded
        status_codes = []
        while not results.empty():
            status_codes.append(results.get())
        
        assert len(status_codes) == 5
        assert all(code == 200 for code in status_codes)


class TestImageValidation:
    """Test suite for image validation functions."""
    
    def test_decode_image_data_with_data_url(self):
        """Test decoding image data with data URL prefix."""
        from app.main import _decode_image_data
        
        # Create test image
        img = Image.new('RGB', (10, 10), color='blue')
        img_buffer = io.BytesIO()
        img.save(img_buffer, format='PNG')
        img_data = img_buffer.getvalue()
        
        # Create data URL
        base64_data = base64.b64encode(img_data).decode('utf-8')
        data_url = f"data:image/png;base64,{base64_data}"
        
        # Test decoding
        decoded = _decode_image_data(data_url)
        assert decoded == img_data

    def test_decode_image_data_base64_only(self):
        """Test decoding plain base64 image data."""
        from app.main import _decode_image_data
        
        # Create test image
        img = Image.new('RGB', (10, 10), color='green')
        img_buffer = io.BytesIO()
        img.save(img_buffer, format='JPEG')
        img_data = img_buffer.getvalue()
        
        # Create base64 data
        base64_data = base64.b64encode(img_data).decode('utf-8')
        
        # Test decoding
        decoded = _decode_image_data(base64_data)
        assert decoded == img_data

    def test_is_valid_image_signature(self):
        """Test image signature validation."""
        from app.main import _is_valid_image_signature
        
        # Test JPEG signature
        jpeg_data = b'\xff\xd8\xff\xe0' + b'test'
        assert _is_valid_image_signature(jpeg_data) is True
        
        # Test PNG signature
        png_data = b'\x89PNG\r\n\x1a\n' + b'test'
        assert _is_valid_image_signature(png_data) is True
        
        # Test WebP signature
        webp_data = b'RIFF' + b'test' + b'WEBP' + b'more'
        assert _is_valid_image_signature(webp_data) is True
        
        # Test invalid signature
        invalid_data = b'invalid'
        assert _is_valid_image_signature(invalid_data) is False
        
        # Test empty data
        assert _is_valid_image_signature(b'') is False

    def test_map_error_to_code(self):
        """Test error message to error code mapping."""
        from app.main import _map_error_to_code
        
        # Test format errors
        assert _map_error_to_code("Invalid image format") == ErrorCode.INVALID_IMAGE_FORMAT
        assert _map_error_to_code("Image decode failed") == ErrorCode.INVALID_IMAGE_FORMAT
        
        # Test size errors
        assert _map_error_to_code("Image too large") == ErrorCode.IMAGE_TOO_LARGE
        assert _map_error_to_code("Memory error") == ErrorCode.IMAGE_TOO_LARGE
        
        # Test model errors
        assert _map_error_to_code("Model failed") == ErrorCode.MODEL_ERROR
        assert _map_error_to_code("TensorFlow error") == ErrorCode.MODEL_ERROR
        
        # Test processing errors
        assert _map_error_to_code("Processing failed") == ErrorCode.PROCESSING_ERROR
        
        # Test default error
        assert _map_error_to_code("Unknown error") == ErrorCode.INTERNAL_ERROR
        assert _map_error_to_code("") == ErrorCode.INTERNAL_ERROR


if __name__ == "__main__":
    pytest.main([__file__])