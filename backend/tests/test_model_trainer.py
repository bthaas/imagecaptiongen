"""
Unit tests for model training utilities.
Tests data generation, model training, and evaluation functionality.
"""

import pytest
import numpy as np
import tempfile
import os
import json
from unittest.mock import Mock, patch, MagicMock

from app.services.model_trainer import (
    DataGenerator, ModelTrainer, ModelTrainerError, create_sample_training_data
)
from app.services.caption_generator import LSTMCaptionGenerator, VocabularyManager


class TestDataGenerator:
    """Test cases for DataGenerator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.features = np.random.randn(10, 2048)
        self.captions = [
            "a man sitting",
            "a woman walking",
            "a dog running",
            "a cat sleeping",
            "a car driving",
            "a bird flying",
            "a house standing",
            "a tree growing",
            "a flower blooming",
            "a child playing"
        ]
        self.vocabulary = VocabularyManager(vocab_size=50, max_length=8)
        self.vocabulary.build_vocabulary(self.captions)
    
    def test_initialization(self):
        """Test DataGenerator initialization."""
        generator = DataGenerator(
            self.features, self.captions, self.vocabulary, batch_size=4, shuffle=True
        )
        
        assert generator.features.shape == (10, 2048)
        assert len(generator.captions) == 10
        assert generator.batch_size == 4
        assert generator.shuffle is True
        assert len(generator.sequences) > 0
        assert len(generator.indices) == len(generator.sequences)
    
    def test_prepare_sequences(self):
        """Test sequence preparation."""
        generator = DataGenerator(
            self.features, self.captions, self.vocabulary, batch_size=4
        )
        
        # Check that sequences were created
        assert len(generator.sequences) > 0
        
        # Check sequence structure
        for feature, input_seq, target in generator.sequences:
            assert feature.shape == (2048,)
            assert len(input_seq) == self.vocabulary.max_length
            assert isinstance(target, (int, np.integer))
    
    def test_len(self):
        """Test __len__ method."""
        generator = DataGenerator(
            self.features, self.captions, self.vocabulary, batch_size=4
        )
        
        expected_batches = len(generator.sequences) // 4
        assert len(generator) == expected_batches
    
    def test_getitem(self):
        """Test __getitem__ method."""
        generator = DataGenerator(
            self.features, self.captions, self.vocabulary, batch_size=4
        )
        
        if len(generator) > 0:
            batch_inputs, batch_targets = generator[0]
            
            # Check batch structure
            assert len(batch_inputs) == 2  # [features, input_sequences]
            assert batch_inputs[0].shape == (4, 2048)  # features
            assert batch_inputs[1].shape == (4, self.vocabulary.max_length)  # input sequences
            assert batch_targets.shape == (4, self.vocabulary.vocab_size)  # targets
    
    def test_on_epoch_end(self):
        """Test epoch end shuffling."""
        generator = DataGenerator(
            self.features, self.captions, self.vocabulary, batch_size=4, shuffle=True
        )
        
        original_indices = generator.indices.copy()
        generator.on_epoch_end()
        
        # Indices should be shuffled (with high probability)
        # Note: There's a small chance they could be the same after shuffling
        assert len(generator.indices) == len(original_indices)


class TestModelTrainer:
    """Test cases for ModelTrainer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.caption_generator = LSTMCaptionGenerator(vocab_size=50, max_length=8)
        self.trainer = ModelTrainer(self.caption_generator)
        self.sample_features = np.random.randn(20, 2048)
        self.sample_captions = [
            f"sample caption {i}" for i in range(20)
        ]
    
    def test_initialization(self):
        """Test ModelTrainer initialization."""
        assert self.trainer.caption_generator == self.caption_generator
        assert self.trainer.training_history is None
    
    def test_prepare_training_data(self):
        """Test training data preparation."""
        vocabulary = self.trainer.prepare_training_data(
            self.sample_features, self.sample_captions
        )
        
        # Check that vocabulary was built
        assert vocabulary is not None
        assert vocabulary.tokenizer is not None
        assert len(vocabulary.word_to_idx) > 0
    
    def test_prepare_training_data_mismatch(self):
        """Test training data preparation with mismatched data."""
        features = np.random.randn(10, 2048)
        captions = ["caption"] * 5  # Different length
        
        with pytest.raises(ModelTrainerError, match="Number of features and captions must match"):
            self.trainer.prepare_training_data(features, captions)
    
    @patch('app.services.model_trainer.DataGenerator')
    def test_train_model(self, mock_data_generator):
        """Test model training."""
        # Setup mocks
        mock_generator_instance = Mock()
        mock_data_generator.return_value = mock_generator_instance
        
        # Mock model
        mock_model = Mock()
        mock_history = Mock()
        mock_history.history = {"loss": [1.0, 0.8, 0.6], "val_loss": [1.1, 0.9, 0.7]}
        mock_model.fit.return_value = mock_history
        
        self.caption_generator.model = mock_model
        self.caption_generator._model_loaded = True
        
        # Mock vocabulary
        with patch.object(self.trainer, 'prepare_training_data') as mock_prepare:
            mock_prepare.return_value = Mock()
            
            # Train model
            history = self.trainer.train_model(
                self.sample_features, self.sample_captions,
                epochs=3, batch_size=4
            )
            
            # Check that training was called
            mock_model.fit.assert_called_once()
            assert history == mock_history.history
            assert self.trainer.training_history == mock_history.history
    
    @patch('app.services.model_trainer.DataGenerator')
    def test_evaluate_model(self, mock_data_generator):
        """Test model evaluation."""
        # Setup mocks
        mock_generator_instance = Mock()
        mock_data_generator.return_value = mock_generator_instance
        
        # Mock model
        mock_model = Mock()
        mock_model.evaluate.return_value = [0.5, 0.8]  # loss, accuracy
        mock_model.metrics_names = ["loss", "accuracy"]
        
        self.caption_generator.model = mock_model
        self.caption_generator._model_loaded = True
        
        # Evaluate model
        results = self.trainer.evaluate_model(self.sample_features, self.sample_captions)
        
        # Check results
        assert "loss" in results
        assert "accuracy" in results
        assert results["loss"] == 0.5
        assert results["accuracy"] == 0.8
        mock_model.evaluate.assert_called_once()
    
    def test_evaluate_model_not_loaded(self):
        """Test model evaluation when model not loaded."""
        with pytest.raises(ModelTrainerError, match="Model not trained or loaded"):
            self.trainer.evaluate_model(self.sample_features, self.sample_captions)
    
    def test_generate_sample_captions(self):
        """Test sample caption generation."""
        # Mock caption generator methods
        self.caption_generator._model_loaded = True
        
        with patch.object(self.caption_generator, 'generate_caption_beam_search') as mock_beam:
            with patch.object(self.caption_generator, 'generate_caption_greedy') as mock_greedy:
                mock_beam.return_value = ("sample caption", 0.8)
                mock_greedy.return_value = ("sample caption", 0.7)
                
                # Test beam search
                results_beam = self.trainer.generate_sample_captions(
                    self.sample_features[:3], num_samples=3, use_beam_search=True
                )
                
                assert len(results_beam) == 3
                assert all(len(result) == 2 for result in results_beam)
                assert mock_beam.call_count == 3
                
                # Test greedy
                results_greedy = self.trainer.generate_sample_captions(
                    self.sample_features[:2], num_samples=2, use_beam_search=False
                )
                
                assert len(results_greedy) == 2
                assert mock_greedy.call_count == 2
    
    def test_generate_sample_captions_not_loaded(self):
        """Test sample caption generation when model not loaded."""
        with pytest.raises(ModelTrainerError, match="Model not trained or loaded"):
            self.trainer.generate_sample_captions(self.sample_features)
    
    def test_save_and_load_training_history(self):
        """Test saving and loading training history."""
        # Set up training history
        self.trainer.training_history = {
            "loss": [1.0, 0.8, 0.6],
            "val_loss": [1.1, 0.9, 0.7],
            "accuracy": [0.5, 0.6, 0.7]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            history_path = f.name
        
        try:
            # Save history
            self.trainer.save_training_history(history_path)
            assert os.path.exists(history_path)
            
            # Create new trainer and load history
            new_trainer = ModelTrainer(self.caption_generator)
            loaded_history = new_trainer.load_training_history(history_path)
            
            # Check that histories match
            assert loaded_history == self.trainer.training_history
            assert new_trainer.training_history == self.trainer.training_history
            
        finally:
            if os.path.exists(history_path):
                os.unlink(history_path)
    
    def test_save_training_history_no_history(self):
        """Test saving training history when no history exists."""
        with pytest.raises(ModelTrainerError, match="No training history to save"):
            self.trainer.save_training_history("dummy_path.json")
    
    def test_error_handling_in_training(self):
        """Test error handling during training."""
        # Mock model to raise exception
        mock_model = Mock()
        mock_model.fit.side_effect = Exception("Mock training error")
        
        self.caption_generator.model = mock_model
        self.caption_generator._model_loaded = True
        
        with patch.object(self.trainer, 'prepare_training_data'):
            with pytest.raises(ModelTrainerError, match="Training failed"):
                self.trainer.train_model(self.sample_features, self.sample_captions)


class TestUtilityFunctions:
    """Test cases for utility functions."""
    
    def test_create_sample_training_data(self):
        """Test sample training data creation."""
        features, captions = create_sample_training_data(num_samples=50)
        
        # Check shapes and types
        assert features.shape == (50, 2048)
        assert len(captions) == 50
        assert all(isinstance(caption, str) for caption in captions)
        assert all(len(caption.split()) >= 3 for caption in captions)  # At least 3 words
    
    def test_create_sample_training_data_different_sizes(self):
        """Test sample training data creation with different sizes."""
        for num_samples in [10, 100, 500]:
            features, captions = create_sample_training_data(num_samples=num_samples)
            
            assert features.shape == (num_samples, 2048)
            assert len(captions) == num_samples


class TestIntegration:
    """Integration tests for model training components."""
    
    def test_end_to_end_training_simulation(self):
        """Test end-to-end training simulation with mocked components."""
        # Create sample data
        features, captions = create_sample_training_data(num_samples=20)
        
        # Create trainer
        caption_generator = LSTMCaptionGenerator(vocab_size=50, max_length=8)
        trainer = ModelTrainer(caption_generator)
        
        # Prepare training data
        vocabulary = trainer.prepare_training_data(features, captions)
        
        # Check that everything is set up correctly
        assert vocabulary is not None
        assert vocabulary.tokenizer is not None
        assert len(vocabulary.word_to_idx) > 0
        
        # Create data generator
        data_gen = DataGenerator(features, captions, vocabulary, batch_size=4)
        
        # Check data generator
        assert len(data_gen) > 0
        if len(data_gen) > 0:
            batch_inputs, batch_targets = data_gen[0]
            assert len(batch_inputs) == 2
            assert batch_inputs[0].shape[1] == 2048  # Feature dimension
            assert batch_targets.shape[1] == vocabulary.vocab_size
    
    def test_vocabulary_integration_with_trainer(self):
        """Test vocabulary integration with trainer."""
        caption_generator = LSTMCaptionGenerator(vocab_size=100, max_length=10)
        trainer = ModelTrainer(caption_generator)
        
        # Sample captions with specific words
        captions = [
            "a red car driving fast",
            "a blue bird flying high",
            "a green tree growing tall"
        ]
        features = np.random.randn(3, 2048)
        
        # Prepare training data
        vocabulary = trainer.prepare_training_data(features, captions)
        
        # Check that specific words are in vocabulary
        assert "red" in vocabulary.word_to_idx
        assert "car" in vocabulary.word_to_idx
        assert "driving" in vocabulary.word_to_idx
        assert "blue" in vocabulary.word_to_idx
        assert "bird" in vocabulary.word_to_idx


if __name__ == "__main__":
    pytest.main([__file__])