"""
Unit tests for ModelManager class.
Tests CNN model loading, feature extraction, and error handling.
"""

import pytest
import numpy as np
import tensorflow as tf
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os

from app.services.model_manager import ModelManager, ModelManagerError


class TestModelManager:
    """Test cases for ModelManager class."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Clear any existing singleton instance
        ModelManager._instance = None
        self.model_manager = ModelManager()
    
    def teardown_method(self):
        """Clean up after each test method."""
        # Clear singleton instance
        ModelManager._instance = None
        if hasattr(self.model_manager, 'model') and self.model_manager.model:
            self.model_manager.clear_cache()
    
    def test_singleton_pattern(self):
        """Test that ModelManager follows singleton pattern."""
        instance1 = ModelManager.get_instance()
        instance2 = ModelManager.get_instance()
        
        assert instance1 is instance2
        assert isinstance(instance1, ModelManager)
    
    def test_initialization(self):
        """Test ModelManager initialization with default values."""
        assert self.model_manager.model is None
        assert self.model_manager.input_shape == (224, 224, 3)
        assert self.model_manager.feature_dim == 2048
        assert self.model_manager._model_loaded is False
    
    @patch('app.services.model_manager.GlobalAveragePooling2D')
    @patch('app.services.model_manager.ResNet50')
    @patch('app.services.model_manager.Model')
    def test_load_model_success(self, mock_model_class, mock_resnet50, mock_pooling):
        """Test successful model loading."""
        # Mock ResNet50 base model
        mock_base_model = Mock()
        mock_base_model.output = Mock()
        mock_base_model.input = Mock()
        mock_base_model.layers = [Mock(), Mock()]
        mock_resnet50.return_value = mock_base_model
        
        # Mock GlobalAveragePooling2D
        mock_pooling_layer = Mock()
        mock_pooling.return_value = mock_pooling_layer
        mock_pooled_output = Mock()
        mock_pooling_layer.return_value = mock_pooled_output
        
        # Mock the final model with layers attribute
        mock_final_model = Mock()
        mock_final_model.layers = [Mock(), Mock()]  # Mock layers for freezing
        mock_model_class.return_value = mock_final_model
        
        # Load model
        result = self.model_manager.load_model()
        
        # Verify ResNet50 was called with correct parameters
        mock_resnet50.assert_called_once_with(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3)
        )
        
        # Verify GlobalAveragePooling2D was applied
        mock_pooling.assert_called_once()
        mock_pooling_layer.assert_called_once_with(mock_base_model.output)
        
        # Verify Model was created with correct inputs/outputs
        mock_model_class.assert_called_once_with(
            inputs=mock_base_model.input, 
            outputs=mock_pooled_output
        )
        
        # Verify model was created and configured
        assert result == mock_final_model
        assert self.model_manager.model == mock_final_model
        assert self.model_manager._model_loaded is True
        
        # Verify layers were frozen (the code freezes layers on the final model)
        for layer in mock_final_model.layers:
            assert layer.trainable is False
    
    @patch('app.services.model_manager.ResNet50')
    def test_load_model_already_loaded(self, mock_resnet50):
        """Test that model loading returns cached instance when already loaded."""
        # Set up already loaded model
        mock_model = Mock()
        self.model_manager.model = mock_model
        self.model_manager._model_loaded = True
        
        # Load model again
        result = self.model_manager.load_model()
        
        # Verify cached model is returned and ResNet50 is not called again
        assert result == mock_model
        mock_resnet50.assert_not_called()
    
    @patch('app.services.model_manager.ResNet50')
    def test_load_model_failure(self, mock_resnet50):
        """Test model loading failure handling."""
        # Mock ResNet50 to raise exception
        mock_resnet50.side_effect = Exception("Model loading failed")
        
        # Verify exception is raised
        with pytest.raises(ModelManagerError, match="Model loading failed"):
            self.model_manager.load_model()
        
        # Verify model state remains unloaded
        assert self.model_manager.model is None
        assert self.model_manager._model_loaded is False
    
    def test_extract_features_model_not_loaded(self):
        """Test feature extraction when model is not loaded."""
        # Create valid input
        image_array = np.random.rand(1, 224, 224, 3).astype(np.float32)
        
        # Mock load_model to set up model
        with patch.object(self.model_manager, 'load_model') as mock_load:
            mock_model = Mock()
            expected_features = np.random.rand(1, 2048)
            mock_model.predict.return_value = expected_features
            mock_load.return_value = mock_model
            
            # Set up the model manager to use the mocked model after load_model is called
            def side_effect():
                self.model_manager.model = mock_model
                self.model_manager._model_loaded = True
                return mock_model
            mock_load.side_effect = side_effect
            
            # Extract features (this should trigger load_model)
            with patch('app.services.model_manager.preprocess_input') as mock_preprocess:
                mock_preprocess.return_value = image_array
                result = self.model_manager.extract_features(image_array)
                
                # Verify load_model was called
                mock_load.assert_called_once()
                assert result.shape == (1, 2048)
                np.testing.assert_array_equal(result, expected_features)
    
    def test_extract_features_success(self):
        """Test successful feature extraction."""
        # Set up loaded model
        mock_model = Mock()
        expected_features = np.random.rand(1, 2048)
        mock_model.predict.return_value = expected_features
        
        self.model_manager.model = mock_model
        self.model_manager._model_loaded = True
        
        # Create valid input
        image_array = np.random.rand(1, 224, 224, 3).astype(np.float32)
        
        # Mock preprocess_input
        with patch('app.services.model_manager.preprocess_input') as mock_preprocess:
            mock_preprocess.return_value = image_array
            
            # Extract features
            result = self.model_manager.extract_features(image_array)
            
            # Verify preprocessing and prediction
            mock_preprocess.assert_called_once()
            mock_model.predict.assert_called_once_with(image_array, verbose=0)
            np.testing.assert_array_equal(result, expected_features)
    
    def test_extract_features_invalid_shape(self):
        """Test feature extraction with invalid input shape."""
        # Set up loaded model
        self.model_manager.model = Mock()
        self.model_manager._model_loaded = True
        
        # Create invalid input shape
        invalid_array = np.random.rand(224, 224, 3)  # Missing batch dimension
        
        # Verify exception is raised
        with pytest.raises(ModelManagerError, match="Invalid input shape"):
            self.model_manager.extract_features(invalid_array)
    
    def test_extract_features_prediction_failure(self):
        """Test feature extraction when model prediction fails."""
        # Set up model that raises exception
        mock_model = Mock()
        mock_model.predict.side_effect = Exception("Prediction failed")
        
        self.model_manager.model = mock_model
        self.model_manager._model_loaded = True
        
        # Create valid input
        image_array = np.random.rand(1, 224, 224, 3).astype(np.float32)
        
        # Mock preprocess_input
        with patch('app.services.model_manager.preprocess_input'):
            # Verify exception is raised
            with pytest.raises(ModelManagerError, match="Feature extraction failed"):
                self.model_manager.extract_features(image_array)
    
    @patch('tensorflow.image.resize')
    @patch('tensorflow.expand_dims')
    @patch('tensorflow.convert_to_tensor')
    def test_preprocess_image_for_model_success(self, mock_convert, mock_expand, mock_resize):
        """Test successful image preprocessing for model input."""
        # Set up input
        input_array = np.random.rand(300, 400, 3).astype(np.float32)
        
        # Mock TensorFlow operations
        mock_tensor = Mock()
        mock_convert.return_value = mock_tensor
        
        mock_resized = Mock()
        mock_resize.return_value = mock_resized
        
        mock_batched = Mock()
        mock_batched.numpy.return_value = np.random.rand(1, 224, 224, 3)
        mock_expand.return_value = mock_batched
        
        # Preprocess image
        result = self.model_manager.preprocess_image_for_model(input_array)
        
        # Verify TensorFlow operations were called correctly
        mock_convert.assert_called_once_with(input_array, dtype=tf.float32)
        mock_resize.assert_called_once_with(mock_tensor, [224, 224])
        mock_expand.assert_called_once_with(mock_resized, 0)
        
        # Verify output shape
        assert result.shape == (1, 224, 224, 3)
    
    def test_preprocess_image_invalid_shape(self):
        """Test image preprocessing with invalid input shape."""
        # Create invalid input (grayscale image)
        invalid_array = np.random.rand(224, 224)
        
        # Verify exception is raised
        with pytest.raises(ModelManagerError, match="Invalid image shape"):
            self.model_manager.preprocess_image_for_model(invalid_array)
    
    @patch('tensorflow.convert_to_tensor')
    def test_preprocess_image_tensorflow_error(self, mock_convert):
        """Test image preprocessing when TensorFlow operations fail."""
        # Mock TensorFlow to raise exception
        mock_convert.side_effect = Exception("TensorFlow error")
        
        # Create valid input
        input_array = np.random.rand(224, 224, 3)
        
        # Verify exception is raised
        with pytest.raises(ModelManagerError, match="Image preprocessing failed"):
            self.model_manager.preprocess_image_for_model(input_array)
    
    def test_get_model_info_not_loaded(self):
        """Test getting model info when model is not loaded."""
        info = self.model_manager.get_model_info()
        
        expected = {
            "loaded": False,
            "model_type": "ResNet50",
            "input_shape": (224, 224, 3),
            "feature_dim": 2048
        }
        
        assert info == expected
    
    def test_get_model_info_loaded(self):
        """Test getting model info when model is loaded."""
        # Set up loaded model
        mock_model = Mock()
        mock_model.count_params.return_value = 25636712
        mock_model.layers = [Mock() for _ in range(175)]
        
        self.model_manager.model = mock_model
        self.model_manager._model_loaded = True
        
        info = self.model_manager.get_model_info()
        
        expected = {
            "loaded": True,
            "model_type": "ResNet50",
            "input_shape": (224, 224, 3),
            "feature_dim": 2048,
            "total_params": 25636712,
            "layers": 175
        }
        
        assert info == expected
    
    def test_clear_cache(self):
        """Test clearing model cache."""
        # Set up loaded model
        mock_model = Mock()
        self.model_manager.model = mock_model
        self.model_manager._model_loaded = True
        
        # Clear cache
        self.model_manager.clear_cache()
        
        # Verify model is cleared
        assert self.model_manager.model is None
        assert self.model_manager._model_loaded is False


class TestModelManagerIntegration:
    """Integration tests for ModelManager with real TensorFlow operations."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        ModelManager._instance = None
        self.model_manager = ModelManager()
    
    def teardown_method(self):
        """Clean up after each test method."""
        ModelManager._instance = None
        if hasattr(self.model_manager, 'model') and self.model_manager.model:
            self.model_manager.clear_cache()
    
    @pytest.mark.slow
    def test_real_model_loading(self):
        """Test loading real ResNet50 model (slow test)."""
        # This test actually downloads and loads the model
        model = self.model_manager.load_model()
        
        # Verify model properties
        assert model is not None
        assert self.model_manager._model_loaded is True
        assert model.input_shape == (None, 224, 224, 3)
        assert model.output_shape == (None, 2048)
    
    @pytest.mark.slow
    def test_real_feature_extraction(self):
        """Test real feature extraction with sample image (slow test)."""
        # Load model
        self.model_manager.load_model()
        
        # Create sample image
        sample_image = np.random.rand(224, 224, 3).astype(np.float32) * 255
        
        # Preprocess image
        preprocessed = self.model_manager.preprocess_image_for_model(sample_image)
        
        # Extract features
        features = self.model_manager.extract_features(preprocessed)
        
        # Verify feature properties
        assert features.shape == (1, 2048)
        assert features.dtype == np.float32
        assert not np.isnan(features).any()
        assert not np.isinf(features).any()