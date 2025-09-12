"""
Caption Service for orchestrating the complete ML pipeline.

This service integrates the CNN feature extraction and LSTM caption generation
components to provide end-to-end image captioning functionality with proper
error handling, confidence scoring, and performance tracking.
"""

import time
import logging
from typing import Tuple, Dict, Optional, Union
import numpy as np
from dataclasses import dataclass

from .model_manager import ModelManager, ModelManagerError
from .caption_generator import LSTMCaptionGenerator, CaptionGeneratorError
from .image_processor import ImageProcessor, ImageProcessingError

logger = logging.getLogger(__name__)


class CaptionServiceError(Exception):
    """Custom exception for caption service errors."""
    pass


@dataclass
class CaptionResult:
    """
    Result object for caption generation containing all relevant information.
    """
    caption: str
    confidence: float
    processing_time: float
    image_metadata: Dict
    model_info: Dict
    success: bool = True
    error_message: Optional[str] = None


class CaptionService:
    """
    Main service class that orchestrates the complete ML pipeline for image captioning.
    
    Integrates CNN feature extraction, LSTM caption generation, and image processing
    with comprehensive error handling and performance monitoring.
    """
    
    def __init__(self, 
                 target_image_size: Tuple[int, int] = (224, 224),
                 max_file_size: int = 10 * 1024 * 1024,
                 default_temperature: float = 1.0,
                 use_beam_search: bool = False,
                 beam_width: int = 3):
        """
        Initialize CaptionService with configuration parameters.
        
        Args:
            target_image_size: Target size for image preprocessing
            max_file_size: Maximum allowed image file size in bytes
            default_temperature: Default temperature for caption generation
            use_beam_search: Whether to use beam search by default
            beam_width: Beam width for beam search decoding
        """
        self.target_image_size = target_image_size
        self.max_file_size = max_file_size
        self.default_temperature = default_temperature
        self.use_beam_search = use_beam_search
        self.beam_width = beam_width
        
        # Initialize components
        self.image_processor = ImageProcessor(
            target_size=target_image_size,
            max_file_size=max_file_size
        )
        self.model_manager = ModelManager.get_instance()
        self.caption_generator = LSTMCaptionGenerator.get_instance()
        
        # Track initialization status
        self._initialized = False
        self._initialization_error = None
    
    def initialize(self) -> bool:
        """
        Initialize the ML pipeline components.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        if self._initialized:
            logger.info("CaptionService already initialized")
            return True
        
        try:
            logger.info("Initializing CaptionService components...")
            
            # Initialize CNN model for feature extraction
            logger.info("Loading CNN model for feature extraction...")
            self.model_manager.load_model()
            
            # Initialize LSTM caption generator
            logger.info("Initializing LSTM caption generator...")
            self.caption_generator.initialize_for_inference()
            
            self._initialized = True
            self._initialization_error = None
            
            logger.info("CaptionService initialization completed successfully")
            return True
            
        except Exception as e:
            error_msg = f"Failed to initialize CaptionService: {str(e)}"
            logger.error(error_msg)
            self._initialization_error = error_msg
            self._initialized = False
            return False
    
    def generate_caption(self, 
                        image_data: bytes,
                        temperature: Optional[float] = None,
                        use_beam_search: Optional[bool] = None,
                        beam_width: Optional[int] = None) -> CaptionResult:
        """
        Generate caption for an image using the complete ML pipeline.
        
        Args:
            image_data: Raw image bytes
            temperature: Sampling temperature (uses default if None)
            use_beam_search: Whether to use beam search (uses default if None)
            beam_width: Beam width for beam search (uses default if None)
            
        Returns:
            CaptionResult: Complete result with caption, confidence, and metadata
        """
        start_time = time.time()
        
        # Check initialization
        if not self._initialized:
            if not self.initialize():
                return CaptionResult(
                    caption="",
                    confidence=0.0,
                    processing_time=time.time() - start_time,
                    image_metadata={},
                    model_info={},
                    success=False,
                    error_message=self._initialization_error or "Service not initialized"
                )
        
        # Use provided parameters or defaults
        temp = temperature if temperature is not None else self.default_temperature
        use_beam = use_beam_search if use_beam_search is not None else self.use_beam_search
        beam_w = beam_width if beam_width is not None else self.beam_width
        
        try:
            # Step 1: Process and validate image
            logger.debug("Processing image...")
            image_metadata = self._process_image_metadata(image_data)
            preprocessed_image = self.image_processor.preprocess_image(image_data)
            
            # Step 2: Extract CNN features
            logger.debug("Extracting CNN features...")
            cnn_features = self.model_manager.extract_features(preprocessed_image)
            
            # Step 3: Generate caption using LSTM
            logger.debug("Generating caption...")
            if use_beam:
                caption, confidence = self.caption_generator.generate_caption_beam_search(
                    cnn_features, beam_width=beam_w, temperature=temp
                )
            else:
                caption, confidence = self.caption_generator.generate_caption_greedy(
                    cnn_features, temperature=temp
                )
            
            # Step 4: Post-process caption
            caption = self._post_process_caption(caption)
            
            processing_time = time.time() - start_time
            
            # Get model information
            model_info = self._get_model_info()
            
            logger.info(f"Caption generated successfully in {processing_time:.3f}s: '{caption}'")
            
            return CaptionResult(
                caption=caption,
                confidence=confidence,
                processing_time=processing_time,
                image_metadata=image_metadata,
                model_info=model_info,
                success=True
            )
            
        except ImageProcessingError as e:
            error_msg = f"Image processing failed: {str(e)}"
            logger.error(error_msg)
            return self._create_error_result(start_time, error_msg, image_data)
            
        except ModelManagerError as e:
            error_msg = f"CNN feature extraction failed: {str(e)}"
            logger.error(error_msg)
            return self._create_error_result(start_time, error_msg, image_data)
            
        except CaptionGeneratorError as e:
            error_msg = f"Caption generation failed: {str(e)}"
            logger.error(error_msg)
            return self._create_error_result(start_time, error_msg, image_data)
            
        except Exception as e:
            error_msg = f"Unexpected error in caption generation: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return self._create_error_result(start_time, error_msg, image_data)
    
    def _process_image_metadata(self, image_data: bytes) -> Dict:
        """
        Extract and process image metadata.
        
        Args:
            image_data: Raw image bytes
            
        Returns:
            Dict: Image metadata
        """
        try:
            return self.image_processor.get_image_info(image_data)
        except Exception as e:
            logger.warning(f"Could not extract image metadata: {str(e)}")
            return {
                'file_size_bytes': len(image_data),
                'format': 'unknown',
                'error': str(e)
            }
    
    def _post_process_caption(self, caption: str) -> str:
        """
        Post-process generated caption for better quality.
        
        Args:
            caption: Raw generated caption
            
        Returns:
            str: Processed caption
        """
        if not caption or not caption.strip():
            return "a photo"
        
        # Clean up caption
        caption = caption.strip()
        
        # Ensure it starts with lowercase (unless it's a proper noun)
        if caption and not caption[0].isupper():
            caption = caption[0].lower() + caption[1:] if len(caption) > 1 else caption.lower()
        
        # Remove duplicate words
        words = caption.split()
        cleaned_words = []
        for word in words:
            if not cleaned_words or word != cleaned_words[-1]:
                cleaned_words.append(word)
        
        caption = ' '.join(cleaned_words)
        
        # Ensure reasonable length
        if len(caption) > 100:
            caption = caption[:97] + "..."
        
        # Add article if missing and appropriate
        if caption and not caption.startswith(('a ', 'an ', 'the ')):
            if caption[0].lower() in 'aeiou':
                caption = f"an {caption}"
            else:
                caption = f"a {caption}"
        
        return caption
    
    def _get_model_info(self) -> Dict:
        """
        Get information about loaded models.
        
        Returns:
            Dict: Model information
        """
        try:
            return {
                'cnn_model': self.model_manager.get_model_info(),
                'lstm_model': self.caption_generator.get_model_info(),
                'image_processor': {
                    'target_size': self.target_image_size,
                    'max_file_size': self.max_file_size
                }
            }
        except Exception as e:
            logger.warning(f"Could not get model info: {str(e)}")
            return {'error': str(e)}
    
    def _create_error_result(self, start_time: float, error_message: str, 
                           image_data: Optional[bytes] = None) -> CaptionResult:
        """
        Create error result with available information.
        
        Args:
            start_time: Processing start time
            error_message: Error message
            image_data: Original image data (optional)
            
        Returns:
            CaptionResult: Error result
        """
        processing_time = time.time() - start_time
        
        # Try to get image metadata if possible
        image_metadata = {}
        if image_data:
            try:
                image_metadata = self._process_image_metadata(image_data)
            except:
                image_metadata = {'file_size_bytes': len(image_data)}
        
        # Try to get model info if possible
        model_info = {}
        try:
            model_info = self._get_model_info()
        except:
            pass
        
        return CaptionResult(
            caption="",
            confidence=0.0,
            processing_time=processing_time,
            image_metadata=image_metadata,
            model_info=model_info,
            success=False,
            error_message=error_message
        )
    
    def get_service_status(self) -> Dict:
        """
        Get current service status and health information.
        
        Returns:
            Dict: Service status information
        """
        status = {
            'initialized': self._initialized,
            'initialization_error': self._initialization_error,
            'components': {}
        }
        
        try:
            # Check image processor
            status['components']['image_processor'] = {
                'available': True,
                'target_size': self.target_image_size,
                'max_file_size': self.max_file_size
            }
            
            # Check CNN model
            cnn_info = self.model_manager.get_model_info()
            status['components']['cnn_model'] = {
                'loaded': cnn_info.get('loaded', False),
                'model_type': cnn_info.get('model_type', 'unknown')
            }
            
            # Check LSTM model
            lstm_info = self.caption_generator.get_model_info()
            status['components']['lstm_model'] = {
                'loaded': lstm_info.get('loaded', False),
                'vocabulary_loaded': lstm_info.get('vocabulary_loaded', False)
            }
            
        except Exception as e:
            status['components']['error'] = str(e)
        
        return status
    
    def clear_cache(self) -> None:
        """
        Clear model caches to free memory.
        """
        try:
            self.model_manager.clear_cache()
            logger.info("Model caches cleared")
        except Exception as e:
            logger.warning(f"Error clearing caches: {str(e)}")
    
    def validate_image_quick(self, image_data: bytes) -> Tuple[bool, Optional[str]]:
        """
        Quick validation of image data without full preprocessing.
        
        Args:
            image_data: Raw image bytes
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            self.image_processor.validate_image_data(image_data)
            return True, None
        except ImageProcessingError as e:
            return False, str(e)
        except Exception as e:
            return False, f"Validation error: {str(e)}"