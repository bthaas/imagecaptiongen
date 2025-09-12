"""
Model Manager for CNN feature extraction using pre-trained models.
Handles loading, caching, and inference of TensorFlow models.
"""

import os
import logging
import numpy as np
from typing import Optional, Tuple
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D
import threading

logger = logging.getLogger(__name__)


class ModelManagerError(Exception):
    """Custom exception for model manager errors."""
    pass


class ModelManager:
    """
    Manages CNN model loading and feature extraction.
    Uses ResNet50 as the base model with classification layers removed.
    """
    
    _instance: Optional['ModelManager'] = None
    _lock = threading.Lock()
    
    def __init__(self):
        """Initialize ModelManager with default configuration."""
        self.model: Optional[Model] = None
        self.input_shape = (224, 224, 3)
        self.feature_dim = 2048
        self._model_loaded = False
        
    @classmethod
    def get_instance(cls) -> 'ModelManager':
        """Get singleton instance of ModelManager."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    def load_model(self) -> Model:
        """
        Load and configure the pre-trained CNN model for feature extraction.
        
        Returns:
            tf.keras.Model: Configured model for feature extraction
            
        Raises:
            ModelManagerError: If model loading fails
        """
        if self._model_loaded and self.model is not None:
            logger.info("Model already loaded, returning cached instance")
            return self.model
            
        try:
            logger.info("Loading ResNet50 model...")
            
            # Load pre-trained ResNet50 without top classification layer
            base_model = ResNet50(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
            
            # Add global average pooling to get fixed-size feature vectors
            x = base_model.output
            x = GlobalAveragePooling2D()(x)
            
            # Create the feature extraction model
            self.model = Model(inputs=base_model.input, outputs=x)
            
            # Freeze all layers since we're using pre-trained features
            for layer in self.model.layers:
                layer.trainable = False
                
            self._model_loaded = True
            logger.info(f"Model loaded successfully. Feature dimension: {self.feature_dim}")
            
            return self.model
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise ModelManagerError(f"Model loading failed: {str(e)}")
    
    def extract_features(self, image_array: np.ndarray) -> np.ndarray:
        """
        Extract features from preprocessed image array.
        
        Args:
            image_array: Preprocessed image array of shape (1, 224, 224, 3)
            
        Returns:
            np.ndarray: Feature vector of shape (1, 2048)
            
        Raises:
            ModelManagerError: If feature extraction fails
        """
        if not self._model_loaded or self.model is None:
            self.load_model()
            
        try:
            # Validate input shape
            if image_array.shape != (1, 224, 224, 3):
                raise ModelManagerError(
                    f"Invalid input shape: {image_array.shape}. Expected: (1, 224, 224, 3)"
                )
            
            # Preprocess for ResNet50
            preprocessed = preprocess_input(image_array.copy())
            
            # Extract features
            features = self.model.predict(preprocessed, verbose=0)
            
            logger.debug(f"Extracted features shape: {features.shape}")
            return features
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {str(e)}")
            raise ModelManagerError(f"Feature extraction failed: {str(e)}")
    
    def preprocess_image_for_model(self, image_array: np.ndarray) -> np.ndarray:
        """
        Preprocess image array for model input.
        
        Args:
            image_array: Image array of shape (height, width, 3)
            
        Returns:
            np.ndarray: Preprocessed array of shape (1, 224, 224, 3)
            
        Raises:
            ModelManagerError: If preprocessing fails
        """
        try:
            # Ensure image is in correct format
            if len(image_array.shape) != 3 or image_array.shape[2] != 3:
                raise ModelManagerError(
                    f"Invalid image shape: {image_array.shape}. Expected: (height, width, 3)"
                )
            
            # Resize to model input size
            image_tensor = tf.convert_to_tensor(image_array, dtype=tf.float32)
            resized = tf.image.resize(image_tensor, [224, 224])
            
            # Add batch dimension
            batched = tf.expand_dims(resized, 0)
            
            return batched.numpy()
            
        except Exception as e:
            logger.error(f"Image preprocessing failed: {str(e)}")
            raise ModelManagerError(f"Image preprocessing failed: {str(e)}")
    
    def get_model_info(self) -> dict:
        """
        Get information about the loaded model.
        
        Returns:
            dict: Model information including architecture and parameters
        """
        if not self._model_loaded or self.model is None:
            return {
                "loaded": False,
                "model_type": "ResNet50",
                "input_shape": self.input_shape,
                "feature_dim": self.feature_dim
            }
        
        return {
            "loaded": True,
            "model_type": "ResNet50",
            "input_shape": self.input_shape,
            "feature_dim": self.feature_dim,
            "total_params": self.model.count_params(),
            "layers": len(self.model.layers)
        }
    
    def clear_cache(self):
        """Clear the cached model to free memory."""
        if self.model is not None:
            del self.model
            self.model = None
            self._model_loaded = False
            logger.info("Model cache cleared")