"""
Integration tests for CaptionService class.

Tests the complete ML pipeline integration including CNN feature extraction,
LSTM caption generation, error handling, and performance tracking.
"""

import pytest
import numpy as np
import io
import time
from unittest.mock import Mock, patch, MagicMock
from PIL import Image

from app.services.caption_service import CaptionService, CaptionServiceError, CaptionResult
from app.services.model_manager import ModelManagerError
from app.services.caption_generator import CaptionGeneratorError
from app.services.image_processor import ImageProcessingError


# Test fixtures
@pytest.fixture
def caption_service():
    """Create CaptionService instance for testing."""
    return CaptionService(
        target_image_size=(224, 224),
        max_file_size=5 * 1024 * 1024,
        default_temperature=1.0,
        use_beam_search=False,
        beam_width=3
    )


@pytest.fixture
def sample_image_data():
    """Create sample image data for testing."""
    img = Image.new('RGB', (300, 200), color='red')
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG')
    return buffer.getvalue()


@pytest.fixture
def mock_components():
    """Create mocked components for testing."""
    with patch('app.services.caption_service.ModelManager') as mock_mm, \
         patch('app.services.caption_service.LSTMCaptionGenerator') as mock_cg, \
         patch('app.services.caption_service.ImageProcessor') as mock_ip:
        
        # Mock ModelManager
        mock_model_manager = Mock()
        mock_model_manager.load_model.return_value = Mock()
        mock_model_manager.extract_features.return_value = np.random.rand(1, 2048)
        mock_model_manager.get_model_info.return_value = {
            'loaded': True, 'model_type': 'ResNet50'
        }
        mock_mm.get_instance.return_value = mock_model_manager
        
        # Mock CaptionGenerator
        mock_caption_gen = Mock()
        mock_caption_gen.initialize_for_inference.return_value = None
        mock_caption_gen.generate_caption_greedy.return_value = ("a red car", 0.85)
        mock_caption_gen.generate_caption_beam_search.return_value = ("a red car on the road", 0.90)
        mock_caption_gen.get_model_info.return_value = {
            'loaded': True, 'vocabulary_loaded': True
        }
        mock_cg.get_instance.return_value = mock_caption_gen
        
        # Mock ImageProcessor
        mock_image_proc = Mock()
        mock_image_proc.preprocess_image.return_value = np.random.rand(1, 224, 224, 3)
        mock_image_proc.get_image_info.return_value = {
            'format': 'JPEG', 'width': 300, 'height': 200, 'file_size_bytes': 1024
        }
        mock_image_proc.validate_image_data.return_value = True
        mock_ip.return_value = mock_image_proc
        
        yield {
            'model_manager': mock_model_manager,
            'caption_generator': mock_caption_gen,
            'image_processor': mock_image_proc
        }


class TestCaptionServiceInitialization:
    """Test CaptionService initialization and configuration."""
    
    def test_initialization_default_params(self):
        """Test initialization with default parameters."""
        service = CaptionService()
        
        assert service.target_image_size == (224, 224)
        assert service.max_file_size == 10 * 1024 * 1024
        assert service.default_temperature == 1.0
        assert service.use_beam_search is False
        assert service.beam_width == 3
        assert not service._initialized
    
    def test_initialization_custom_params(self):
        """Test initialization with custom parameters."""
        service = CaptionService(
            target_image_size=(256, 256),
            max_file_size=5 * 1024 * 1024,
            default_temperature=0.8,
            use_beam_search=True,
            beam_width=5
        )
        
        assert service.target_image_size == (256, 256)
        assert service.max_file_size == 5 * 1024 * 1024
        assert service.default_temperature == 0.8
        assert service.use_beam_search is True
        assert service.beam_width == 5
    
    def test_component_initialization(self, mock_components):
        """Test that components are properly initialized."""
        service = CaptionService()
        
        # Components should be created
        assert service.image_processor is not None
        assert service.model_manager is not None
        assert service.caption_generator is not None


class TestCaptionServiceInitialize:
    """Test CaptionService initialize method."""
    
    def test_initialize_success(self, mock_components):
        """Test successful initialization."""
        service = CaptionService()
        
        result = service.initialize()
        
        assert result is True
        assert service._initialized is True
        assert service._initialization_error is None
        
        # Verify components were initialized
        mock_components['model_manager'].load_model.assert_called_once()
        mock_components['caption_generator'].initialize_for_inference.assert_called_once()
    
    def test_initialize_already_initialized(self, mock_components):
        """Test initialization when already initialized."""
        service = CaptionService()
        service._initialized = True
        
        result = service.initialize()
        
        assert result is True
        # Components should not be initialized again
        mock_components['model_manager'].load_model.assert_not_called()
    
    def test_initialize_model_manager_failure(self, mock_components):
        """Test initialization failure in model manager."""
        service = CaptionService()
        mock_components['model_manager'].load_model.side_effect = ModelManagerError("Model load failed")
        
        result = service.initialize()
        
        assert result is False
        assert service._initialized is False
        assert "Model load failed" in service._initialization_error
    
    def test_initialize_caption_generator_failure(self, mock_components):
        """Test initialization failure in caption generator."""
        service = CaptionService()
        mock_components['caption_generator'].initialize_for_inference.side_effect = CaptionGeneratorError("Init failed")
        
        result = service.initialize()
        
        assert result is False
        assert service._initialized is False
        assert "Init failed" in service._initialization_error


class TestCaptionGeneration:
    """Test caption generation functionality."""
    
    def test_generate_caption_success_greedy(self, mock_components, sample_image_data):
        """Test successful caption generation with greedy decoding."""
        service = CaptionService()
        service._initialized = True
        
        result = service.generate_caption(sample_image_data)
        
        assert isinstance(result, CaptionResult)
        assert result.success is True
        assert result.caption == "a red car"
        assert result.confidence == 0.85
        assert result.processing_time > 0
        assert result.error_message is None
        
        # Verify pipeline was called correctly
        mock_components['image_processor'].preprocess_image.assert_called_once()
        mock_components['model_manager'].extract_features.assert_called_once()
        mock_components['caption_generator'].generate_caption_greedy.assert_called_once()
    
    def test_generate_caption_success_beam_search(self, mock_components, sample_image_data):
        """Test successful caption generation with beam search."""
        service = CaptionService(use_beam_search=True)
        service._initialized = True
        
        result = service.generate_caption(sample_image_data)
        
        assert result.success is True
        assert result.caption == "a red car on the road"
        assert result.confidence == 0.90
        
        # Verify beam search was used
        mock_components['caption_generator'].generate_caption_beam_search.assert_called_once()
        mock_components['caption_generator'].generate_caption_greedy.assert_not_called()
    
    def test_generate_caption_custom_parameters(self, mock_components, sample_image_data):
        """Test caption generation with custom parameters."""
        service = CaptionService()
        service._initialized = True
        
        result = service.generate_caption(
            sample_image_data,
            temperature=0.8,
            use_beam_search=True,
            beam_width=5
        )
        
        assert result.success is True
        
        # Verify custom parameters were used
        mock_components['caption_generator'].generate_caption_beam_search.assert_called_with(
            mock_components['model_manager'].extract_features.return_value,
            beam_width=5,
            temperature=0.8
        )
    
    def test_generate_caption_auto_initialize(self, mock_components, sample_image_data):
        """Test that caption generation auto-initializes if needed."""
        service = CaptionService()
        # Service not initialized
        
        result = service.generate_caption(sample_image_data)
        
        assert result.success is True
        # Verify initialization was called
        mock_components['model_manager'].load_model.assert_called_once()
        mock_components['caption_generator'].initialize_for_inference.assert_called_once()
    
    def test_generate_caption_initialization_failure(self, mock_components, sample_image_data):
        """Test caption generation when initialization fails."""
        service = CaptionService()
        mock_components['model_manager'].load_model.side_effect = Exception("Init failed")
        
        result = service.generate_caption(sample_image_data)
        
        assert result.success is False
        assert "Init failed" in result.error_message
        assert result.caption == ""
        assert result.confidence == 0.0


class TestErrorHandling:
    """Test error handling in caption generation."""
    
    def test_image_processing_error(self, mock_components, sample_image_data):
        """Test handling of image processing errors."""
        service = CaptionService()
        service._initialized = True
        
        mock_components['image_processor'].preprocess_image.side_effect = ImageProcessingError("Invalid image")
        
        result = service.generate_caption(sample_image_data)
        
        assert result.success is False
        assert "Image processing failed" in result.error_message
        assert "Invalid image" in result.error_message
    
    def test_feature_extraction_error(self, mock_components, sample_image_data):
        """Test handling of CNN feature extraction errors."""
        service = CaptionService()
        service._initialized = True
        
        mock_components['model_manager'].extract_features.side_effect = ModelManagerError("Feature extraction failed")
        
        result = service.generate_caption(sample_image_data)
        
        assert result.success is False
        assert "CNN feature extraction failed" in result.error_message
        assert "Feature extraction failed" in result.error_message
    
    def test_caption_generation_error(self, mock_components, sample_image_data):
        """Test handling of caption generation errors."""
        service = CaptionService()
        service._initialized = True
        
        mock_components['caption_generator'].generate_caption_greedy.side_effect = CaptionGeneratorError("Generation failed")
        
        result = service.generate_caption(sample_image_data)
        
        assert result.success is False
        assert "Caption generation failed" in result.error_message
        assert "Generation failed" in result.error_message
    
    def test_unexpected_error(self, mock_components, sample_image_data):
        """Test handling of unexpected errors."""
        service = CaptionService()
        service._initialized = True
        
        mock_components['model_manager'].extract_features.side_effect = RuntimeError("Unexpected error")
        
        result = service.generate_caption(sample_image_data)
        
        assert result.success is False
        assert "Unexpected error" in result.error_message


class TestCaptionPostProcessing:
    """Test caption post-processing functionality."""
    
    def test_post_process_empty_caption(self, caption_service):
        """Test post-processing of empty caption."""
        result = caption_service._post_process_caption("")
        assert result == "a photo"
        
        result = caption_service._post_process_caption("   ")
        assert result == "a photo"
    
    def test_post_process_add_article(self, caption_service):
        """Test adding articles to captions."""
        result = caption_service._post_process_caption("red car")
        assert result == "a red car"
        
        result = caption_service._post_process_caption("elephant walking")
        assert result == "an elephant walking"
    
    def test_post_process_remove_duplicates(self, caption_service):
        """Test removing duplicate words."""
        result = caption_service._post_process_caption("a a red red car")
        assert result == "a red car"
    
    def test_post_process_length_limit(self, caption_service):
        """Test caption length limiting."""
        long_caption = "a person standing next to a large building with many windows and doors and architectural features and decorative elements and beautiful landscaping and surrounding trees and plants and flowers and pathways and walkways and outdoor furniture and lighting fixtures"
        result = caption_service._post_process_caption(long_caption)
        assert len(result) <= 100
        assert result.endswith("...")
    
    def test_post_process_preserve_existing_article(self, caption_service):
        """Test that existing articles are preserved."""
        result = caption_service._post_process_caption("the red car")
        assert result == "the red car"
        
        result = caption_service._post_process_caption("an old house")
        assert result == "an old house"


class TestServiceStatus:
    """Test service status and health monitoring."""
    
    def test_get_service_status_initialized(self, mock_components):
        """Test service status when initialized."""
        service = CaptionService()
        service._initialized = True
        
        status = service.get_service_status()
        
        assert status['initialized'] is True
        assert status['initialization_error'] is None
        assert 'components' in status
        assert 'image_processor' in status['components']
        assert 'cnn_model' in status['components']
        assert 'lstm_model' in status['components']
    
    def test_get_service_status_not_initialized(self, mock_components):
        """Test service status when not initialized."""
        service = CaptionService()
        service._initialization_error = "Init failed"
        
        status = service.get_service_status()
        
        assert status['initialized'] is False
        assert status['initialization_error'] == "Init failed"
    
    def test_validate_image_quick_success(self, mock_components, sample_image_data):
        """Test quick image validation success."""
        service = CaptionService()
        
        is_valid, error = service.validate_image_quick(sample_image_data)
        
        assert is_valid is True
        assert error is None
        mock_components['image_processor'].validate_image_data.assert_called_once()
    
    def test_validate_image_quick_failure(self, mock_components, sample_image_data):
        """Test quick image validation failure."""
        service = CaptionService()
        mock_components['image_processor'].validate_image_data.side_effect = ImageProcessingError("Invalid format")
        
        is_valid, error = service.validate_image_quick(sample_image_data)
        
        assert is_valid is False
        assert "Invalid format" in error
    
    def test_clear_cache(self, mock_components):
        """Test clearing model caches."""
        service = CaptionService()
        
        service.clear_cache()
        
        mock_components['model_manager'].clear_cache.assert_called_once()


class TestIntegrationScenarios:
    """Test complete integration scenarios."""
    
    def test_end_to_end_pipeline_greedy(self, mock_components, sample_image_data):
        """Test complete end-to-end pipeline with greedy decoding."""
        service = CaptionService()
        
        result = service.generate_caption(sample_image_data)
        
        # Verify complete pipeline execution
        assert result.success is True
        assert isinstance(result.caption, str)
        assert isinstance(result.confidence, float)
        assert isinstance(result.processing_time, float)
        assert isinstance(result.image_metadata, dict)
        assert isinstance(result.model_info, dict)
        
        # Verify all components were called in order
        mock_components['image_processor'].get_image_info.assert_called_once()
        mock_components['image_processor'].preprocess_image.assert_called_once()
        mock_components['model_manager'].extract_features.assert_called_once()
        mock_components['caption_generator'].generate_caption_greedy.assert_called_once()
    
    def test_end_to_end_pipeline_beam_search(self, mock_components, sample_image_data):
        """Test complete end-to-end pipeline with beam search."""
        service = CaptionService(use_beam_search=True, beam_width=5)
        
        result = service.generate_caption(sample_image_data)
        
        assert result.success is True
        mock_components['caption_generator'].generate_caption_beam_search.assert_called_once()
    
    def test_multiple_requests_same_service(self, mock_components, sample_image_data):
        """Test multiple caption requests using the same service instance."""
        service = CaptionService()
        
        # First request
        result1 = service.generate_caption(sample_image_data)
        assert result1.success is True
        
        # Second request (should reuse initialized components)
        result2 = service.generate_caption(sample_image_data)
        assert result2.success is True
        
        # Initialization should only happen once
        assert mock_components['model_manager'].load_model.call_count == 1
        assert mock_components['caption_generator'].initialize_for_inference.call_count == 1
    
    def test_performance_tracking(self, mock_components, sample_image_data):
        """Test that performance is properly tracked."""
        service = CaptionService()
        
        # Add small delay to mock processing
        def slow_extract_features(*args, **kwargs):
            time.sleep(0.01)  # 10ms delay
            return np.random.rand(1, 2048)
        
        mock_components['model_manager'].extract_features.side_effect = slow_extract_features
        
        result = service.generate_caption(sample_image_data)
        
        assert result.success is True
        assert result.processing_time >= 0.01  # Should include the delay
        assert result.processing_time < 1.0    # Should be reasonable
    
    def test_metadata_collection(self, mock_components, sample_image_data):
        """Test that metadata is properly collected."""
        service = CaptionService()
        
        result = service.generate_caption(sample_image_data)
        
        assert result.success is True
        
        # Check image metadata
        assert 'format' in result.image_metadata
        assert 'width' in result.image_metadata
        assert 'height' in result.image_metadata
        
        # Check model info
        assert 'cnn_model' in result.model_info
        assert 'lstm_model' in result.model_info
        assert 'image_processor' in result.model_info


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_very_small_image(self, mock_components):
        """Test with very small image."""
        # Create tiny image
        img = Image.new('RGB', (50, 50), color='blue')
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG')
        tiny_image_data = buffer.getvalue()
        
        service = CaptionService()
        result = service.generate_caption(tiny_image_data)
        
        # Should still work (preprocessing handles resizing)
        assert result.success is True
    
    def test_metadata_extraction_failure(self, mock_components, sample_image_data):
        """Test when metadata extraction fails."""
        service = CaptionService()
        service._initialized = True
        
        # Mock metadata extraction to fail
        mock_components['image_processor'].get_image_info.side_effect = Exception("Metadata failed")
        
        result = service.generate_caption(sample_image_data)
        
        # Should still succeed with limited metadata
        assert result.success is True
        assert 'error' in result.image_metadata or 'file_size_bytes' in result.image_metadata
    
    def test_model_info_extraction_failure(self, mock_components, sample_image_data):
        """Test when model info extraction fails."""
        service = CaptionService()
        service._initialized = True
        
        # Mock model info extraction to fail
        mock_components['model_manager'].get_model_info.side_effect = Exception("Model info failed")
        
        result = service.generate_caption(sample_image_data)
        
        # Should still succeed with limited model info
        assert result.success is True


if __name__ == "__main__":
    pytest.main([__file__])