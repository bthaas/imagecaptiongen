"""
Unit tests for LSTM caption generation components.
Tests vocabulary management, model architecture, and caption generation.
"""

import pytest
import numpy as np
import tempfile
import os
import json
from unittest.mock import Mock, patch, MagicMock

from app.services.caption_generator import (
    VocabularyManager, LSTMCaptionGenerator, CaptionGeneratorError
)


class TestVocabularyManager:
    """Test cases for VocabularyManager class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.vocab_manager = VocabularyManager(vocab_size=100, max_length=10)
        self.sample_captions = [
            "a man sitting on a bench",
            "a woman walking in the park",
            "a dog running on the beach",
            "a cat sleeping on the couch"
        ]
    
    def test_initialization(self):
        """Test VocabularyManager initialization."""
        assert self.vocab_manager.vocab_size == 100
        assert self.vocab_manager.max_length == 10
        assert self.vocab_manager.start_token == "<start>"
        assert self.vocab_manager.end_token == "<end>"
        assert self.vocab_manager.unk_token == "<unk>"
        assert self.vocab_manager.pad_token == "<pad>"
        assert self.vocab_manager.tokenizer is None
    
    def test_build_vocabulary(self):
        """Test vocabulary building from captions."""
        self.vocab_manager.build_vocabulary(self.sample_captions)
        
        # Check that tokenizer is created
        assert self.vocab_manager.tokenizer is not None
        
        # Check that special tokens are in vocabulary
        assert self.vocab_manager.start_token in self.vocab_manager.word_to_idx
        assert self.vocab_manager.end_token in self.vocab_manager.word_to_idx
        assert self.vocab_manager.unk_token in self.vocab_manager.word_to_idx
        
        # Check that common words are in vocabulary
        assert "a" in self.vocab_manager.word_to_idx
        assert "man" in self.vocab_manager.word_to_idx
        assert "sitting" in self.vocab_manager.word_to_idx
        
        # Check bidirectional mapping
        for word, idx in self.vocab_manager.word_to_idx.items():
            assert self.vocab_manager.idx_to_word[idx] == word
    
    def test_load_default_vocabulary(self):
        """Test loading default vocabulary."""
        self.vocab_manager.load_default_vocabulary()
        
        # Check that vocabulary is loaded
        assert len(self.vocab_manager.word_to_idx) > 0
        assert self.vocab_manager.tokenizer is not None
        
        # Check that special tokens are present
        assert self.vocab_manager.start_token in self.vocab_manager.word_to_idx
        assert self.vocab_manager.end_token in self.vocab_manager.word_to_idx
        
        # Check that common words are present
        assert "a" in self.vocab_manager.word_to_idx
        assert "person" in self.vocab_manager.word_to_idx
        assert "dog" in self.vocab_manager.word_to_idx
    
    def test_text_to_sequences(self):
        """Test text to sequence conversion."""
        self.vocab_manager.build_vocabulary(self.sample_captions)
        
        texts = ["a man sitting", "a dog running"]
        sequences = self.vocab_manager.text_to_sequences(texts)
        
        assert len(sequences) == 2
        assert all(isinstance(seq, list) for seq in sequences)
        assert all(isinstance(token, int) for seq in sequences for token in seq)
    
    def test_sequences_to_text(self):
        """Test sequence to text conversion."""
        self.vocab_manager.build_vocabulary(self.sample_captions)
        
        # Create sample sequences
        sequences = [[1, 2, 3], [1, 4, 5]]
        texts = self.vocab_manager.sequences_to_text(sequences)
        
        assert len(texts) == 2
        assert all(isinstance(text, str) for text in texts)
    
    def test_save_and_load_vocabulary(self):
        """Test vocabulary saving and loading."""
        self.vocab_manager.build_vocabulary(self.sample_captions)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            vocab_path = f.name
        
        try:
            # Save vocabulary
            self.vocab_manager.save_vocabulary(vocab_path)
            assert os.path.exists(vocab_path)
            
            # Create new manager and load vocabulary
            new_manager = VocabularyManager()
            new_manager.load_vocabulary(vocab_path)
            
            # Check that vocabularies match
            assert new_manager.word_to_idx == self.vocab_manager.word_to_idx
            assert new_manager.idx_to_word == self.vocab_manager.idx_to_word
            assert new_manager.vocab_size == self.vocab_manager.vocab_size
            assert new_manager.max_length == self.vocab_manager.max_length
            
        finally:
            if os.path.exists(vocab_path):
                os.unlink(vocab_path)
    
    def test_build_vocabulary_error_handling(self):
        """Test error handling in vocabulary building."""
        with patch.object(self.vocab_manager, 'tokenizer', None):
            with pytest.raises(CaptionGeneratorError):
                # This should fail because tokenizer creation is mocked to fail
                with patch('app.services.caption_generator.Tokenizer', side_effect=Exception("Mock error")):
                    self.vocab_manager.build_vocabulary(self.sample_captions)


class TestLSTMCaptionGenerator:
    """Test cases for LSTMCaptionGenerator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.generator = LSTMCaptionGenerator(
            vocab_size=100, max_length=10, embedding_dim=64, lstm_units=128
        )
        self.sample_features = np.random.randn(1, 2048)
    
    def test_initialization(self):
        """Test LSTMCaptionGenerator initialization."""
        assert self.generator.vocab_size == 100
        assert self.generator.max_length == 10
        assert self.generator.embedding_dim == 64
        assert self.generator.lstm_units == 128
        assert self.generator.feature_dim == 2048
        assert isinstance(self.generator.vocabulary, VocabularyManager)
        assert self.generator.model is None
        assert not self.generator._model_loaded
    
    def test_singleton_pattern(self):
        """Test singleton pattern implementation."""
        instance1 = LSTMCaptionGenerator.get_instance()
        instance2 = LSTMCaptionGenerator.get_instance()
        assert instance1 is instance2
    
    @patch('app.services.caption_generator.tf.keras.Model')
    def test_build_model(self, mock_model_class):
        """Test model building."""
        # Mock the model
        mock_model = Mock()
        mock_model_class.return_value = mock_model
        
        # Build model
        result = self.generator.build_model()
        
        # Check that model was created and compiled
        assert result == mock_model
        assert self.generator.model == mock_model
        assert self.generator._model_loaded
        mock_model.compile.assert_called_once()
    
    def test_initialize_for_inference(self):
        """Test initialization for inference."""
        with patch.object(self.generator.vocabulary, 'load_default_vocabulary') as mock_load_vocab:
            with patch.object(self.generator, 'build_model') as mock_build_model:
                self.generator.initialize_for_inference()
                
                mock_load_vocab.assert_called_once()
                mock_build_model.assert_called_once()
    
    def test_generate_caption_greedy_not_loaded(self):
        """Test greedy caption generation when model not loaded."""
        with pytest.raises(CaptionGeneratorError, match="Model not loaded"):
            self.generator.generate_caption_greedy(self.sample_features)
    
    @patch('app.services.caption_generator.pad_sequences')
    def test_generate_caption_greedy(self, mock_pad_sequences):
        """Test greedy caption generation."""
        # Setup mocks
        self.generator._model_loaded = True
        self.generator.model = Mock()
        self.generator.vocabulary.load_default_vocabulary()
        
        # Mock model prediction
        mock_predictions = np.array([[[0.1, 0.8, 0.1] + [0.0] * 97]])  # 100 vocab size
        self.generator.model.predict.return_value = mock_predictions
        
        # Mock pad_sequences
        mock_pad_sequences.return_value = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        
        # Generate caption
        caption, confidence = self.generator.generate_caption_greedy(self.sample_features)
        
        # Check results
        assert isinstance(caption, str)
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0
        self.generator.model.predict.assert_called()
    
    def test_generate_caption_beam_search_not_loaded(self):
        """Test beam search caption generation when model not loaded."""
        with pytest.raises(CaptionGeneratorError, match="Model not loaded"):
            self.generator.generate_caption_beam_search(self.sample_features)
    
    @patch('app.services.caption_generator.pad_sequences')
    def test_generate_caption_beam_search(self, mock_pad_sequences):
        """Test beam search caption generation."""
        # Setup mocks
        self.generator._model_loaded = True
        self.generator.model = Mock()
        self.generator.vocabulary.load_default_vocabulary()
        
        # Mock model prediction
        mock_predictions = np.array([[[0.1, 0.8, 0.1] + [0.0] * 97]])
        self.generator.model.predict.return_value = mock_predictions
        
        # Mock pad_sequences
        mock_pad_sequences.return_value = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        
        # Generate caption
        caption, confidence = self.generator.generate_caption_beam_search(
            self.sample_features, beam_width=2
        )
        
        # Check results
        assert isinstance(caption, str)
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0
        self.generator.model.predict.assert_called()
    
    def test_save_model_no_model(self):
        """Test saving model when no model exists."""
        with pytest.raises(CaptionGeneratorError, match="No model to save"):
            self.generator.save_model("model.h5", "vocab.json")
    
    @patch('app.services.caption_generator.tf.keras.models.load_model')
    def test_load_model(self, mock_load_model):
        """Test model loading."""
        # Setup mocks
        mock_model = Mock()
        mock_load_model.return_value = mock_model
        
        with patch.object(self.generator.vocabulary, 'load_vocabulary') as mock_load_vocab:
            # Load model
            self.generator.load_model("model.h5", "vocab.json")
            
            # Check that model and vocabulary were loaded
            mock_load_vocab.assert_called_once_with("vocab.json")
            mock_load_model.assert_called_once_with("model.h5")
            assert self.generator.model == mock_model
            assert self.generator._model_loaded
    
    def test_get_model_info_not_loaded(self):
        """Test getting model info when model not loaded."""
        info = self.generator.get_model_info()
        
        assert info["loaded"] is False
        assert info["vocab_size"] == 100
        assert info["max_length"] == 10
        assert info["embedding_dim"] == 64
        assert info["lstm_units"] == 128
        assert info["feature_dim"] == 2048
        assert info["vocabulary_loaded"] is False
    
    def test_get_model_info_loaded(self):
        """Test getting model info when model is loaded."""
        self.generator._model_loaded = True
        self.generator.model = Mock()
        self.generator.vocabulary.tokenizer = Mock()
        
        info = self.generator.get_model_info()
        
        assert info["loaded"] is True
        assert info["vocabulary_loaded"] is True
    
    def test_error_handling_in_generation(self):
        """Test error handling during caption generation."""
        self.generator._model_loaded = True
        self.generator.model = Mock()
        self.generator.vocabulary.load_default_vocabulary()
        
        # Mock model to raise exception
        self.generator.model.predict.side_effect = Exception("Mock prediction error")
        
        with pytest.raises(CaptionGeneratorError, match="Caption generation failed"):
            self.generator.generate_caption_greedy(self.sample_features)


class TestIntegration:
    """Integration tests for caption generation components."""
    
    def test_end_to_end_caption_generation(self):
        """Test end-to-end caption generation process."""
        # Create generator
        generator = LSTMCaptionGenerator(vocab_size=50, max_length=5)
        
        # Initialize for inference
        generator.initialize_for_inference()
        
        # Generate sample features
        features = np.random.randn(1, 2048)
        
        # Generate caption (this will use random weights, so caption quality will be poor)
        caption, confidence = generator.generate_caption_greedy(features, temperature=1.0)
        
        # Check that we get valid outputs
        assert isinstance(caption, str)
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0
    
    def test_vocabulary_integration(self):
        """Test vocabulary integration with caption generator."""
        generator = LSTMCaptionGenerator(vocab_size=100, max_length=10)
        
        # Build vocabulary
        sample_captions = [
            "a person walking",
            "a dog running",
            "a car driving"
        ]
        generator.vocabulary.build_vocabulary(sample_captions)
        
        # Check that vocabulary is properly integrated
        assert generator.vocabulary.tokenizer is not None
        assert len(generator.vocabulary.word_to_idx) > 0
        
        # Test text processing
        sequences = generator.vocabulary.text_to_sequences(["a person"])
        texts = generator.vocabulary.sequences_to_text(sequences)
        
        assert len(texts) == 1
        assert isinstance(texts[0], str)


if __name__ == "__main__":
    pytest.main([__file__])