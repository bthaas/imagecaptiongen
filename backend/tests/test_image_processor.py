"""
Unit tests for ImageProcessor class.

Tests image validation, preprocessing, and error handling functionality.
"""

import io
import pytest
import numpy as np
from PIL import Image
from unittest.mock import patch, MagicMock

from app.services.image_processor import ImageProcessor, ImageProcessingError


# Global fixtures available to all test classes
@pytest.fixture
def processor():
    """Create ImageProcessor instance for testing."""
    return ImageProcessor()

@pytest.fixture
def sample_jpeg_data():
    """Create sample JPEG image data for testing."""
    img = Image.new('RGB', (100, 100), color='red')
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG')
    return buffer.getvalue()

@pytest.fixture
def sample_png_data():
    """Create sample PNG image data for testing."""
    img = Image.new('RGB', (150, 150), color='blue')
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    return buffer.getvalue()

@pytest.fixture
def sample_webp_data():
    """Create sample WebP image data for testing."""
    img = Image.new('RGB', (200, 200), color='green')
    buffer = io.BytesIO()
    img.save(buffer, format='WEBP')
    return buffer.getvalue()

@pytest.fixture
def large_image_data():
    """Create large image data for size testing."""
    img = Image.new('RGB', (2000, 2000), color='white')
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG', quality=95)
    return buffer.getvalue()

@pytest.fixture
def small_image_data():
    """Create very small image data for size testing."""
    img = Image.new('RGB', (20, 20), color='black')
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG')
    return buffer.getvalue()


class TestImageValidation:
    """Test image validation functionality."""
    
    def test_validate_valid_jpeg(self, processor, sample_jpeg_data):
        """Test validation of valid JPEG image."""
        assert processor.validate_image_data(sample_jpeg_data) is True
    
    def test_validate_valid_png(self, processor, sample_png_data):
        """Test validation of valid PNG image."""
        assert processor.validate_image_data(sample_png_data) is True
    
    def test_validate_valid_webp(self, processor, sample_webp_data):
        """Test validation of valid WebP image."""
        assert processor.validate_image_data(sample_webp_data) is True
    
    def test_validate_empty_data(self, processor):
        """Test validation fails with empty data."""
        with pytest.raises(ImageProcessingError, match="Empty image data"):
            processor.validate_image_data(b'')
    
    def test_validate_invalid_data(self, processor):
        """Test validation fails with invalid image data."""
        with pytest.raises(ImageProcessingError, match="Invalid image data"):
            processor.validate_image_data(b'not an image')
    
    def test_validate_oversized_image(self, processor):
        """Test validation fails with oversized image."""
        # Create processor with small max size
        small_processor = ImageProcessor(max_file_size=1000)
        
        # Create image larger than limit
        img = Image.new('RGB', (500, 500), color='red')
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=95)
        large_data = buffer.getvalue()
        
        with pytest.raises(ImageProcessingError, match="exceeds maximum"):
            small_processor.validate_image_data(large_data)
    
    def test_validate_too_small_dimensions(self, processor, small_image_data):
        """Test validation fails with too small dimensions."""
        with pytest.raises(ImageProcessingError, match="too small"):
            processor.validate_image_data(small_image_data)
    
    def test_validate_too_large_dimensions(self, processor):
        """Test validation fails with too large dimensions."""
        # Mock image with huge dimensions
        with patch('PIL.Image.open') as mock_open:
            mock_img = MagicMock()
            mock_img.format = 'JPEG'
            mock_img.size = (5000, 5000)  # Larger than MAX_DIMENSION
            mock_open.return_value.__enter__.return_value = mock_img
            
            with pytest.raises(ImageProcessingError, match="too large"):
                processor.validate_image_data(b'fake_data')


class TestFormatValidation:
    """Test format validation functionality."""
    
    def test_validate_jpeg_format(self, processor, sample_jpeg_data):
        """Test JPEG format validation."""
        assert processor.validate_format(sample_jpeg_data) == 'JPEG'
    
    def test_validate_png_format(self, processor, sample_png_data):
        """Test PNG format validation."""
        assert processor.validate_format(sample_png_data) == 'PNG'
    
    def test_validate_webp_format(self, processor, sample_webp_data):
        """Test WebP format validation."""
        assert processor.validate_format(sample_webp_data) == 'WEBP'
    
    def test_validate_unsupported_format(self, processor):
        """Test validation fails with unsupported format."""
        # Mock unsupported format
        with patch('PIL.Image.open') as mock_open:
            mock_img = MagicMock()
            mock_img.format = 'BMP'  # Unsupported format
            mock_open.return_value.__enter__.return_value = mock_img
            
            with pytest.raises(ImageProcessingError, match="Unsupported format"):
                processor.validate_format(b'fake_data')


class TestImageResizing:
    """Test image resizing functionality."""
    
    def test_resize_to_default_size(self, processor):
        """Test resizing to default target size."""
        img = Image.new('RGB', (300, 400), color='red')
        resized = processor.resize_image(img)
        
        assert resized.size == processor.target_size
        assert resized.mode == 'RGB'
    
    def test_resize_to_custom_size(self, processor):
        """Test resizing to custom target size."""
        img = Image.new('RGB', (100, 200), color='blue')
        target_size = (128, 128)
        resized = processor.resize_image(img, target_size)
        
        assert resized.size == target_size
        assert resized.mode == 'RGB'
    
    def test_resize_maintains_aspect_ratio(self, processor):
        """Test that resizing maintains aspect ratio through padding."""
        # Create rectangular image
        img = Image.new('RGB', (400, 200), color='green')
        resized = processor.resize_image(img, (224, 224))
        
        # Should be padded to square
        assert resized.size == (224, 224)
    
    def test_resize_error_handling(self, processor):
        """Test resize error handling."""
        with patch.object(Image.Image, 'copy', side_effect=Exception("Resize error")):
            img = Image.new('RGB', (100, 100), color='red')
            with pytest.raises(ImageProcessingError, match="Failed to resize"):
                processor.resize_image(img)


class TestImageNormalization:
    """Test image normalization functionality."""
    
    def test_normalize_uint8_array(self, processor):
        """Test normalization of uint8 array."""
        # Create array with values 0-255
        array = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        normalized = processor.normalize_image(array)
        
        assert normalized.dtype == np.float32
        assert normalized.min() >= 0.0
        assert normalized.max() <= 1.0
    
    def test_normalize_float_array(self, processor):
        """Test normalization of float array."""
        # Create array already in [0, 1] range
        array = np.random.random((224, 224, 3)).astype(np.float32)
        normalized = processor.normalize_image(array)
        
        assert normalized.dtype == np.float32
        assert normalized.min() >= 0.0
        assert normalized.max() <= 1.0
    
    def test_normalize_clips_values(self, processor):
        """Test that normalization clips out-of-range values."""
        # Create array with values outside [0, 1]
        array = np.array([-0.5, 0.5, 1.5], dtype=np.float32)
        normalized = processor.normalize_image(array)
        
        assert normalized.min() >= 0.0
        assert normalized.max() <= 1.0
    
    def test_normalize_error_handling(self, processor):
        """Test normalization error handling."""
        with patch('numpy.clip', side_effect=Exception("Normalize error")):
            array = np.random.random((10, 10, 3))
            with pytest.raises(ImageProcessingError, match="Failed to normalize"):
                processor.normalize_image(array)


class TestImagePreprocessing:
    """Test complete image preprocessing pipeline."""
    
    def test_preprocess_valid_image(self, processor, sample_jpeg_data):
        """Test preprocessing of valid image."""
        result = processor.preprocess_image(sample_jpeg_data)
        
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert result.shape == (1, 224, 224, 3)  # Batch dimension added
        assert result.min() >= 0.0
        assert result.max() <= 1.0
    
    def test_preprocess_rgba_image(self, processor):
        """Test preprocessing of RGBA image (should convert to RGB)."""
        img = Image.new('RGBA', (100, 100), color=(255, 0, 0, 128))
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        rgba_data = buffer.getvalue()
        
        result = processor.preprocess_image(rgba_data)
        
        assert result.shape == (1, 224, 224, 3)  # Should be RGB
    
    def test_preprocess_grayscale_image(self, processor):
        """Test preprocessing of grayscale image (should convert to RGB)."""
        img = Image.new('L', (100, 100), color=128)
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG')
        gray_data = buffer.getvalue()
        
        result = processor.preprocess_image(gray_data)
        
        assert result.shape == (1, 224, 224, 3)  # Should be RGB
    
    def test_preprocess_invalid_image(self, processor):
        """Test preprocessing fails with invalid image."""
        with pytest.raises(ImageProcessingError):
            processor.preprocess_image(b'invalid data')
    
    def test_preprocess_custom_target_size(self):
        """Test preprocessing with custom target size."""
        processor = ImageProcessor(target_size=(128, 128))
        img = Image.new('RGB', (200, 200), color='blue')
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG')
        image_data = buffer.getvalue()
        
        result = processor.preprocess_image(image_data)
        
        assert result.shape == (1, 128, 128, 3)


class TestImageInfo:
    """Test image metadata extraction."""
    
    def test_get_image_info_jpeg(self, processor, sample_jpeg_data):
        """Test extracting info from JPEG image."""
        info = processor.get_image_info(sample_jpeg_data)
        
        assert info['format'] == 'JPEG'
        assert info['mode'] == 'RGB'
        assert info['width'] == 100
        assert info['height'] == 100
        assert info['size'] == (100, 100)
        assert info['file_size_bytes'] == len(sample_jpeg_data)
        assert 'has_transparency' in info
    
    def test_get_image_info_png(self, processor, sample_png_data):
        """Test extracting info from PNG image."""
        info = processor.get_image_info(sample_png_data)
        
        assert info['format'] == 'PNG'
        assert info['width'] == 150
        assert info['height'] == 150
    
    def test_get_image_info_with_transparency(self, processor):
        """Test extracting info from image with transparency."""
        img = Image.new('RGBA', (100, 100), color=(255, 0, 0, 128))
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        rgba_data = buffer.getvalue()
        
        info = processor.get_image_info(rgba_data)
        
        assert info['mode'] == 'RGBA'
        assert info['has_transparency'] is True
    
    def test_get_image_info_invalid_data(self, processor):
        """Test info extraction fails with invalid data."""
        with pytest.raises(ImageProcessingError, match="Could not extract image info"):
            processor.get_image_info(b'invalid data')


class TestEdgeCases:
    """Test edge cases and error scenarios."""
    
    def test_corrupted_image_data(self, processor):
        """Test handling of corrupted image data."""
        # Create definitely corrupted data by mixing random bytes with JPEG header
        corrupted_data = b'\xff\xd8\xff\xe0' + b'random_corrupted_data' * 100
        
        with pytest.raises(ImageProcessingError):
            processor.validate_image_data(corrupted_data)
    
    def test_very_small_valid_image(self, processor):
        """Test handling of very small but valid image."""
        # Create image just above minimum size
        img = Image.new('RGB', (33, 33), color='red')
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG')
        small_data = buffer.getvalue()
        
        # Should be valid
        assert processor.validate_image_data(small_data) is True
    
    def test_square_vs_rectangular_images(self, processor):
        """Test preprocessing of square vs rectangular images."""
        # Square image
        square_img = Image.new('RGB', (200, 200), color='red')
        square_buffer = io.BytesIO()
        square_img.save(square_buffer, format='JPEG')
        
        # Rectangular image
        rect_img = Image.new('RGB', (400, 200), color='blue')
        rect_buffer = io.BytesIO()
        rect_img.save(rect_buffer, format='JPEG')
        
        square_result = processor.preprocess_image(square_buffer.getvalue())
        rect_result = processor.preprocess_image(rect_buffer.getvalue())
        
        # Both should result in same output shape
        assert square_result.shape == rect_result.shape == (1, 224, 224, 3)
    
    def test_processor_configuration(self):
        """Test ImageProcessor with custom configuration."""
        custom_processor = ImageProcessor(
            target_size=(256, 256),
            max_file_size=5 * 1024 * 1024  # 5MB
        )
        
        assert custom_processor.target_size == (256, 256)
        assert custom_processor.max_file_size == 5 * 1024 * 1024
    
    def test_batch_dimension_handling(self, processor, sample_jpeg_data):
        """Test that batch dimension is properly added."""
        result = processor.preprocess_image(sample_jpeg_data)
        
        # Should have batch dimension
        assert len(result.shape) == 4
        assert result.shape[0] == 1  # Batch size of 1