"""
Unit tests for Pydantic API models
"""
import pytest
from datetime import datetime
from pydantic import ValidationError

from app.models.api_models import (
    ErrorResponse,
    ErrorCode,
    HealthResponse,
    CaptionRequest,
    CaptionResponse,
    ImageMetadata,
    ModelPrediction,
    APIResponse
)


class TestErrorResponse:
    """Test cases for ErrorResponse model"""
    
    def test_error_response_creation(self):
        """Test creating a valid ErrorResponse"""
        error = ErrorResponse(
            error=ErrorCode.INVALID_IMAGE_FORMAT,
            message="Invalid image format provided"
        )
        
        assert error.error == ErrorCode.INVALID_IMAGE_FORMAT
        assert error.message == "Invalid image format provided"
        assert isinstance(error.timestamp, datetime)
        assert error.request_id is None
        assert error.details is None
    
    def test_error_response_with_optional_fields(self):
        """Test ErrorResponse with optional fields"""
        error = ErrorResponse(
            error=ErrorCode.PROCESSING_ERROR,
            message="Processing failed",
            request_id="test-123",
            details={"step": "preprocessing", "reason": "invalid_format"}
        )
        
        assert error.request_id == "test-123"
        assert error.details["step"] == "preprocessing"
        assert error.details["reason"] == "invalid_format"
    
    def test_error_response_json_serialization(self):
        """Test ErrorResponse JSON serialization"""
        error = ErrorResponse(
            error=ErrorCode.MODEL_ERROR,
            message="Model inference failed"
        )
        
        json_data = error.model_dump()
        
        assert json_data["error"] == "MODEL_ERROR"
        assert json_data["message"] == "Model inference failed"
        assert "timestamp" in json_data


class TestHealthResponse:
    """Test cases for HealthResponse model"""
    
    def test_health_response_creation(self):
        """Test creating a valid HealthResponse"""
        health = HealthResponse(
            status="healthy",
            version="1.0.0",
            system_info={
                "memory_usage_percent": 45.2,
                "disk_free_gb": 100.0
            },
            services={
                "api": "healthy",
                "model": "available"
            }
        )
        
        assert health.status == "healthy"
        assert health.version == "1.0.0"
        assert health.system_info["memory_usage_percent"] == 45.2
        assert health.services["api"] == "healthy"
        assert isinstance(health.timestamp, datetime)
    
    def test_health_response_json_serialization(self):
        """Test HealthResponse JSON serialization"""
        health = HealthResponse(
            status="healthy",
            version="1.0.0",
            system_info={"memory": 50.0},
            services={"api": "healthy"}
        )
        
        json_data = health.model_dump()
        
        assert json_data["status"] == "healthy"
        assert "timestamp" in json_data
        assert isinstance(json_data["system_info"], dict)
        assert isinstance(json_data["services"], dict)


class TestCaptionRequest:
    """Test cases for CaptionRequest model"""
    
    def test_caption_request_creation(self):
        """Test creating a valid CaptionRequest"""
        request = CaptionRequest(
            image_data="data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ..."
        )
        
        assert request.image_data.startswith("data:image/jpeg;base64,")
        assert request.max_length == 20  # default value
        assert request.temperature == 1.0  # default value
    
    def test_caption_request_with_custom_params(self):
        """Test CaptionRequest with custom parameters"""
        request = CaptionRequest(
            image_data="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...",
            max_length=15,
            temperature=0.8
        )
        
        assert request.max_length == 15
        assert request.temperature == 0.8
    
    def test_caption_request_validation_max_length(self):
        """Test CaptionRequest validation for max_length"""
        # Test minimum boundary
        with pytest.raises(ValidationError):
            CaptionRequest(
                image_data="data:image/jpeg;base64,test",
                max_length=4  # below minimum of 5
            )
        
        # Test maximum boundary
        with pytest.raises(ValidationError):
            CaptionRequest(
                image_data="data:image/jpeg;base64,test",
                max_length=51  # above maximum of 50
            )
    
    def test_caption_request_validation_temperature(self):
        """Test CaptionRequest validation for temperature"""
        # Test minimum boundary
        with pytest.raises(ValidationError):
            CaptionRequest(
                image_data="data:image/jpeg;base64,test",
                temperature=0.05  # below minimum of 0.1
            )
        
        # Test maximum boundary
        with pytest.raises(ValidationError):
            CaptionRequest(
                image_data="data:image/jpeg;base64,test",
                temperature=2.5  # above maximum of 2.0
            )
    
    def test_caption_request_missing_image_data(self):
        """Test CaptionRequest validation for missing image_data"""
        with pytest.raises(ValidationError):
            CaptionRequest()


class TestCaptionResponse:
    """Test cases for CaptionResponse model"""
    
    def test_caption_response_creation(self):
        """Test creating a valid CaptionResponse"""
        response = CaptionResponse(
            caption="A dog sitting in a park",
            confidence=0.85,
            processing_time=2.34,
            image_id="img_123456789"
        )
        
        assert response.caption == "A dog sitting in a park"
        assert response.confidence == 0.85
        assert response.processing_time == 2.34
        assert response.image_id == "img_123456789"
        assert isinstance(response.timestamp, datetime)
    
    def test_caption_response_validation_confidence(self):
        """Test CaptionResponse validation for confidence"""
        # Test minimum boundary
        with pytest.raises(ValidationError):
            CaptionResponse(
                caption="Test caption",
                confidence=-0.1,  # below minimum of 0.0
                processing_time=1.0,
                image_id="test"
            )
        
        # Test maximum boundary
        with pytest.raises(ValidationError):
            CaptionResponse(
                caption="Test caption",
                confidence=1.1,  # above maximum of 1.0
                processing_time=1.0,
                image_id="test"
            )
    
    def test_caption_response_validation_processing_time(self):
        """Test CaptionResponse validation for processing_time"""
        with pytest.raises(ValidationError):
            CaptionResponse(
                caption="Test caption",
                confidence=0.8,
                processing_time=-1.0,  # negative processing time
                image_id="test"
            )
    
    def test_caption_response_json_serialization(self):
        """Test CaptionResponse JSON serialization"""
        response = CaptionResponse(
            caption="A beautiful sunset",
            confidence=0.92,
            processing_time=1.56,
            image_id="img_987654321"
        )
        
        json_data = response.model_dump()
        
        assert json_data["caption"] == "A beautiful sunset"
        assert json_data["confidence"] == 0.92
        assert json_data["processing_time"] == 1.56
        assert json_data["image_id"] == "img_987654321"
        assert "timestamp" in json_data


class TestImageMetadata:
    """Test cases for ImageMetadata model"""
    
    def test_image_metadata_creation(self):
        """Test creating a valid ImageMetadata"""
        metadata = ImageMetadata(
            width=1920,
            height=1080,
            format="JPEG",
            size_bytes=2048576
        )
        
        assert metadata.width == 1920
        assert metadata.height == 1080
        assert metadata.format == "JPEG"
        assert metadata.size_bytes == 2048576
        assert isinstance(metadata.upload_timestamp, datetime)
    
    def test_image_metadata_validation(self):
        """Test ImageMetadata validation"""
        # Test invalid width
        with pytest.raises(ValidationError):
            ImageMetadata(
                width=0,  # must be greater than 0
                height=1080,
                format="JPEG",
                size_bytes=1024
            )
        
        # Test invalid height
        with pytest.raises(ValidationError):
            ImageMetadata(
                width=1920,
                height=-100,  # must be greater than 0
                format="JPEG",
                size_bytes=1024
            )
        
        # Test invalid size_bytes
        with pytest.raises(ValidationError):
            ImageMetadata(
                width=1920,
                height=1080,
                format="JPEG",
                size_bytes=0  # must be greater than 0
            )


class TestModelPrediction:
    """Test cases for ModelPrediction model"""
    
    def test_model_prediction_creation(self):
        """Test creating a valid ModelPrediction"""
        prediction = ModelPrediction(
            raw_output=[0.1, 0.8, 0.05, 0.05],
            decoded_caption="A cat on a table",
            confidence_scores=[0.9, 0.8, 0.7, 0.6],
            feature_vector=[0.1, 0.2, 0.3, 0.4, 0.5]
        )
        
        assert len(prediction.raw_output) == 4
        assert prediction.decoded_caption == "A cat on a table"
        assert len(prediction.confidence_scores) == 4
        assert len(prediction.feature_vector) == 5
    
    def test_model_prediction_empty_lists(self):
        """Test ModelPrediction with empty lists"""
        prediction = ModelPrediction(
            raw_output=[],
            decoded_caption="Empty prediction",
            confidence_scores=[],
            feature_vector=[]
        )
        
        assert prediction.raw_output == []
        assert prediction.confidence_scores == []
        assert prediction.feature_vector == []


class TestAPIResponse:
    """Test cases for APIResponse model"""
    
    def test_api_response_success(self):
        """Test creating a successful APIResponse"""
        response = APIResponse(
            success=True,
            data={"result": "success", "value": 42}
        )
        
        assert response.success is True
        assert response.data["result"] == "success"
        assert response.data["value"] == 42
        assert response.error is None
        assert isinstance(response.timestamp, datetime)
    
    def test_api_response_error(self):
        """Test creating an error APIResponse"""
        error = ErrorResponse(
            error=ErrorCode.VALIDATION_ERROR,
            message="Validation failed"
        )
        
        response = APIResponse(
            success=False,
            error=error
        )
        
        assert response.success is False
        assert response.data is None
        assert response.error.error == ErrorCode.VALIDATION_ERROR
        assert response.error.message == "Validation failed"
    
    def test_api_response_json_serialization(self):
        """Test APIResponse JSON serialization"""
        response = APIResponse(
            success=True,
            data={"test": "value"}
        )
        
        json_data = response.model_dump()
        
        assert json_data["success"] is True
        assert json_data["data"]["test"] == "value"
        assert json_data["error"] is None
        assert "timestamp" in json_data


class TestErrorCode:
    """Test cases for ErrorCode enum"""
    
    def test_error_code_values(self):
        """Test that all error codes have expected values"""
        assert ErrorCode.INVALID_IMAGE_FORMAT == "INVALID_IMAGE_FORMAT"
        assert ErrorCode.IMAGE_TOO_LARGE == "IMAGE_TOO_LARGE"
        assert ErrorCode.MODEL_ERROR == "MODEL_ERROR"
        assert ErrorCode.PROCESSING_ERROR == "PROCESSING_ERROR"
        assert ErrorCode.VALIDATION_ERROR == "VALIDATION_ERROR"
        assert ErrorCode.INTERNAL_ERROR == "INTERNAL_ERROR"
    
    def test_error_code_enum_membership(self):
        """Test ErrorCode enum membership"""
        all_codes = list(ErrorCode)
        
        assert len(all_codes) == 6
        assert ErrorCode.INVALID_IMAGE_FORMAT in all_codes
        assert ErrorCode.INTERNAL_ERROR in all_codes