"""
Image processing service for the AI Image Caption Generator.

This module provides image validation, preprocessing, and normalization
functionality for the caption generation pipeline.
"""

import io
import logging
from typing import Tuple, Optional, Union
from PIL import Image, ImageOps
import numpy as np


logger = logging.getLogger(__name__)


class ImageProcessingError(Exception):
    """Custom exception for image processing errors."""
    pass


class ImageProcessor:
    """
    Handles image validation, preprocessing, and normalization for ML model input.
    
    Supports JPEG, PNG, and WebP formats with configurable size limits and
    preprocessing parameters for optimal model performance.
    """
    
    # Supported image formats
    SUPPORTED_FORMATS = {'JPEG', 'PNG', 'WEBP'}
    
    # Default configuration
    DEFAULT_TARGET_SIZE = (224, 224)  # Standard input size for most CNN models
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB max file size
    MIN_DIMENSION = 32  # Minimum width/height in pixels
    MAX_DIMENSION = 4096  # Maximum width/height in pixels
    
    def __init__(self, 
                 target_size: Tuple[int, int] = DEFAULT_TARGET_SIZE,
                 max_file_size: int = MAX_FILE_SIZE):
        """
        Initialize ImageProcessor with configuration parameters.
        
        Args:
            target_size: Target dimensions (width, height) for model input
            max_file_size: Maximum allowed file size in bytes
        """
        self.target_size = target_size
        self.max_file_size = max_file_size
        
    def validate_image_data(self, image_data: bytes) -> bool:
        """
        Validate image data for format, size, and basic integrity.
        
        Args:
            image_data: Raw image bytes
            
        Returns:
            bool: True if image is valid
            
        Raises:
            ImageProcessingError: If image validation fails
        """
        if not image_data:
            raise ImageProcessingError("Empty image data provided")
            
        if len(image_data) > self.max_file_size:
            raise ImageProcessingError(
                f"Image size {len(image_data)} bytes exceeds maximum {self.max_file_size} bytes"
            )
            
        try:
            # Try to open and verify the image
            with Image.open(io.BytesIO(image_data)) as img:
                # Verify format is supported
                if img.format not in self.SUPPORTED_FORMATS:
                    raise ImageProcessingError(
                        f"Unsupported image format: {img.format}. "
                        f"Supported formats: {', '.join(self.SUPPORTED_FORMATS)}"
                    )
                
                # Check image dimensions
                width, height = img.size
                if width < self.MIN_DIMENSION or height < self.MIN_DIMENSION:
                    raise ImageProcessingError(
                        f"Image dimensions {width}x{height} too small. "
                        f"Minimum dimension: {self.MIN_DIMENSION}px"
                    )
                    
                if width > self.MAX_DIMENSION or height > self.MAX_DIMENSION:
                    raise ImageProcessingError(
                        f"Image dimensions {width}x{height} too large. "
                        f"Maximum dimension: {self.MAX_DIMENSION}px"
                    )
                
                # Verify image can be loaded (not corrupted)
                img.verify()
                
        except Exception as e:
            if isinstance(e, ImageProcessingError):
                raise
            raise ImageProcessingError(f"Invalid image data: {str(e)}")
            
        return True
    
    def validate_format(self, image_data: bytes) -> str:
        """
        Validate and return the image format.
        
        Args:
            image_data: Raw image bytes
            
        Returns:
            str: Image format (JPEG, PNG, WEBP)
            
        Raises:
            ImageProcessingError: If format is unsupported
        """
        try:
            with Image.open(io.BytesIO(image_data)) as img:
                format_name = img.format
                if format_name not in self.SUPPORTED_FORMATS:
                    raise ImageProcessingError(
                        f"Unsupported format: {format_name}. "
                        f"Supported: {', '.join(self.SUPPORTED_FORMATS)}"
                    )
                return format_name
        except Exception as e:
            if isinstance(e, ImageProcessingError):
                raise
            raise ImageProcessingError(f"Could not determine image format: {str(e)}")
    
    def resize_image(self, image: Image.Image, target_size: Optional[Tuple[int, int]] = None) -> Image.Image:
        """
        Resize image to target dimensions while maintaining aspect ratio.
        
        Args:
            image: PIL Image object
            target_size: Target dimensions (width, height). Uses instance default if None.
            
        Returns:
            Image.Image: Resized image
        """
        if target_size is None:
            target_size = self.target_size
            
        try:
            # Use thumbnail to maintain aspect ratio, then pad if needed
            image_copy = image.copy()
            image_copy.thumbnail(target_size, Image.Resampling.LANCZOS)
            
            # Create new image with target size and paste resized image centered
            new_image = Image.new('RGB', target_size, (0, 0, 0))
            
            # Calculate position to center the image
            x = (target_size[0] - image_copy.width) // 2
            y = (target_size[1] - image_copy.height) // 2
            
            new_image.paste(image_copy, (x, y))
            
            return new_image
            
        except Exception as e:
            raise ImageProcessingError(f"Failed to resize image: {str(e)}")
    
    def normalize_image(self, image_array: np.ndarray) -> np.ndarray:
        """
        Normalize image array for model input.
        
        Converts pixel values to [0, 1] range and ensures proper data type.
        
        Args:
            image_array: Image array with pixel values
            
        Returns:
            np.ndarray: Normalized image array
        """
        try:
            # Ensure array is float32
            normalized = image_array.astype(np.float32)
            
            # Normalize to [0, 1] range
            if normalized.max() > 1.0:
                normalized = normalized / 255.0
                
            # Ensure values are in valid range
            normalized = np.clip(normalized, 0.0, 1.0)
            
            return normalized
            
        except Exception as e:
            raise ImageProcessingError(f"Failed to normalize image: {str(e)}")
    
    def preprocess_image(self, image_data: bytes) -> np.ndarray:
        """
        Complete preprocessing pipeline: validate, resize, and normalize image.
        
        Args:
            image_data: Raw image bytes
            
        Returns:
            np.ndarray: Preprocessed image array ready for model input
            
        Raises:
            ImageProcessingError: If any preprocessing step fails
        """
        try:
            # Validate image data
            self.validate_image_data(image_data)
            
            # Load image
            with Image.open(io.BytesIO(image_data)) as img:
                # Convert to RGB if needed (handles RGBA, grayscale, etc.)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Resize image
                resized_img = self.resize_image(img)
                
                # Convert to numpy array
                image_array = np.array(resized_img)
                
                # Normalize for model input
                normalized_array = self.normalize_image(image_array)
                
                # Add batch dimension if needed
                if len(normalized_array.shape) == 3:
                    normalized_array = np.expand_dims(normalized_array, axis=0)
                
                logger.info(f"Successfully preprocessed image to shape: {normalized_array.shape}")
                
                return normalized_array
                
        except Exception as e:
            if isinstance(e, ImageProcessingError):
                raise
            raise ImageProcessingError(f"Preprocessing failed: {str(e)}")
    
    def get_image_info(self, image_data: bytes) -> dict:
        """
        Extract metadata information from image.
        
        Args:
            image_data: Raw image bytes
            
        Returns:
            dict: Image metadata including format, size, dimensions
        """
        try:
            with Image.open(io.BytesIO(image_data)) as img:
                return {
                    'format': img.format,
                    'mode': img.mode,
                    'size': img.size,
                    'width': img.width,
                    'height': img.height,
                    'file_size_bytes': len(image_data),
                    'has_transparency': img.mode in ('RGBA', 'LA') or 'transparency' in img.info
                }
        except Exception as e:
            raise ImageProcessingError(f"Could not extract image info: {str(e)}")