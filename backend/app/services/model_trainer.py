"""
Model training utilities for LSTM caption generation.
Provides functionality for training the caption generation model on custom datasets.
"""

import os
import logging
import numpy as np
from typing import List, Tuple, Dict, Optional, Generator
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import json
from .caption_generator import LSTMCaptionGenerator, VocabularyManager, CaptionGeneratorError

logger = logging.getLogger(__name__)


class ModelTrainerError(Exception):
    """Custom exception for model trainer errors."""
    pass


class DataGenerator(tf.keras.utils.Sequence):
    """
    Data generator for training the caption generation model.
    Generates batches of (CNN features, text sequences) -> target sequences.
    """
    
    def __init__(self, features: np.ndarray, captions: List[str], 
                 vocabulary: VocabularyManager, batch_size: int = 32, shuffle: bool = True):
        """
        Initialize data generator.
        
        Args:
            features: CNN features array of shape (n_samples, feature_dim)
            captions: List of caption strings
            vocabulary: Vocabulary manager instance
            batch_size: Batch size for training
            shuffle: Whether to shuffle data between epochs
        """
        super().__init__()
        self.features = features
        self.captions = captions
        self.vocabulary = vocabulary
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Prepare sequences
        self.sequences = self._prepare_sequences()
        self.indices = np.arange(len(self.sequences))
        
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def _prepare_sequences(self) -> List[Tuple[np.ndarray, List[int], List[int]]]:
        """
        Prepare training sequences from captions.
        
        Returns:
            List of (feature, input_seq, target_seq) tuples
        """
        sequences = []
        
        for i, caption in enumerate(self.captions):
            # Add start and end tokens
            processed_caption = f"{self.vocabulary.start_token} {caption.lower().strip()} {self.vocabulary.end_token}"
            
            # Convert to sequence
            seq = self.vocabulary.text_to_sequences([processed_caption])[0]
            
            # Pad the full sequence
            padded_seq = pad_sequences([seq], maxlen=self.vocabulary.max_length, padding='post')[0]
            
            # Create input (all but last) and target (all but first) sequences
            input_seq = padded_seq[:-1]  # Remove last token
            target_seq = padded_seq[1:]  # Remove first token
            
            # Pad to ensure consistent length
            if len(input_seq) < self.vocabulary.max_length - 1:
                input_seq = np.pad(input_seq, (0, self.vocabulary.max_length - 1 - len(input_seq)))
            if len(target_seq) < self.vocabulary.max_length - 1:
                target_seq = np.pad(target_seq, (0, self.vocabulary.max_length - 1 - len(target_seq)))
                
            sequences.append((self.features[i], input_seq[:self.vocabulary.max_length-1], target_seq[:self.vocabulary.max_length-1]))
        
        return sequences
    
    def __len__(self) -> int:
        """Return number of batches per epoch."""
        return len(self.sequences) // self.batch_size
    
    def __getitem__(self, index: int) -> Tuple[Tuple[tf.Tensor, tf.Tensor], tf.Tensor]:
        """
        Generate one batch of data.
        
        Args:
            index: Batch index
            
        Returns:
            Tuple of ((features, input_sequences), target_sequences)
        """
        # Get batch indices
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        
        # Initialize batch arrays
        batch_features = np.zeros((self.batch_size, self.features.shape[1]), dtype=np.float32)
        batch_input_seqs = np.zeros((self.batch_size, self.vocabulary.max_length - 1), dtype=np.int32)
        batch_targets = np.zeros((self.batch_size, self.vocabulary.max_length - 1, self.vocabulary.vocab_size), dtype=np.float32)
        
        # Fill batch
        for i, seq_idx in enumerate(batch_indices):
            feature, input_seq, target_seq = self.sequences[seq_idx]
            
            batch_features[i] = feature
            batch_input_seqs[i] = input_seq
            
            # One-hot encode target sequence
            for j, target_word in enumerate(target_seq):
                if target_word < self.vocabulary.vocab_size:
                    batch_targets[i, j, target_word] = 1
        
        # Convert to tensors
        return (tf.convert_to_tensor(batch_features), tf.convert_to_tensor(batch_input_seqs)), tf.convert_to_tensor(batch_targets)
    
    def on_epoch_end(self):
        """Shuffle indices after each epoch."""
        if self.shuffle:
            np.random.shuffle(self.indices)


class ModelTrainer:
    """
    Trainer for LSTM caption generation model.
    """
    
    def __init__(self, caption_generator: LSTMCaptionGenerator):
        """
        Initialize model trainer.
        
        Args:
            caption_generator: Caption generator instance to train
        """
        self.caption_generator = caption_generator
        self.training_history: Optional[Dict] = None
    
    def prepare_training_data(self, image_features: np.ndarray, captions: List[str]) -> VocabularyManager:
        """
        Prepare training data and build vocabulary.
        
        Args:
            image_features: CNN features array of shape (n_samples, feature_dim)
            captions: List of caption strings
            
        Returns:
            VocabularyManager: Built vocabulary manager
        """
        try:
            if len(image_features) != len(captions):
                raise ModelTrainerError("Number of features and captions must match")
            
            # Build vocabulary from captions
            self.caption_generator.vocabulary.build_vocabulary(captions)
            
            logger.info(f"Training data prepared: {len(captions)} samples")
            return self.caption_generator.vocabulary
            
        except Exception as e:
            logger.error(f"Failed to prepare training data: {str(e)}")
            raise ModelTrainerError(f"Training data preparation failed: {str(e)}")
    
    def train_model(self, 
                   image_features: np.ndarray, 
                   captions: List[str],
                   validation_split: float = 0.2,
                   epochs: int = 50,
                   batch_size: int = 32,
                   save_path: Optional[str] = None) -> Dict:
        """
        Train the caption generation model.
        
        Args:
            image_features: CNN features array of shape (n_samples, feature_dim)
            captions: List of caption strings
            validation_split: Fraction of data to use for validation
            epochs: Number of training epochs
            batch_size: Training batch size
            save_path: Path to save the best model
            
        Returns:
            Dict: Training history
        """
        try:
            # Prepare training data
            self.prepare_training_data(image_features, captions)
            
            # Build model if not already built
            if not self.caption_generator._model_loaded:
                self.caption_generator.build_model()
            
            # Split data
            n_samples = len(image_features)
            n_val = int(n_samples * validation_split)
            n_train = n_samples - n_val
            
            # Shuffle data
            indices = np.random.permutation(n_samples)
            train_indices = indices[:n_train]
            val_indices = indices[n_train:]
            
            train_features = image_features[train_indices]
            train_captions = [captions[i] for i in train_indices]
            val_features = image_features[val_indices]
            val_captions = [captions[i] for i in val_indices]
            
            # Create data generators
            train_generator = DataGenerator(
                train_features, train_captions, 
                self.caption_generator.vocabulary, batch_size, shuffle=True
            )
            
            val_generator = DataGenerator(
                val_features, val_captions,
                self.caption_generator.vocabulary, batch_size, shuffle=False
            )
            
            # Setup callbacks
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True,
                    verbose=1
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-7,
                    verbose=1
                )
            ]
            
            if save_path:
                callbacks.append(
                    ModelCheckpoint(
                        filepath=save_path,
                        monitor='val_loss',
                        save_best_only=True,
                        verbose=1
                    )
                )
            
            # Train model
            logger.info(f"Starting training with {n_train} training samples, {n_val} validation samples")
            
            history = self.caption_generator.model.fit(
                train_generator,
                validation_data=val_generator,
                epochs=epochs,
                callbacks=callbacks,
                verbose=1
            )
            
            self.training_history = history.history
            
            logger.info("Training completed successfully")
            return self.training_history
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise ModelTrainerError(f"Training failed: {str(e)}")
    
    def evaluate_model(self, image_features: np.ndarray, captions: List[str]) -> Dict:
        """
        Evaluate the trained model on test data.
        
        Args:
            image_features: Test CNN features
            captions: Test captions
            
        Returns:
            Dict: Evaluation metrics
        """
        try:
            if not self.caption_generator._model_loaded:
                raise ModelTrainerError("Model not trained or loaded")
            
            # Create test generator
            test_generator = DataGenerator(
                image_features, captions,
                self.caption_generator.vocabulary, batch_size=32, shuffle=False
            )
            
            # Evaluate
            results = self.caption_generator.model.evaluate(test_generator, verbose=1)
            
            # Create results dictionary
            metrics = {}
            for i, metric_name in enumerate(self.caption_generator.model.metrics_names):
                metrics[metric_name] = results[i]
            
            logger.info(f"Evaluation completed: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}")
            raise ModelTrainerError(f"Evaluation failed: {str(e)}")
    
    def generate_sample_captions(self, image_features: np.ndarray, 
                               num_samples: int = 5, use_beam_search: bool = True) -> List[Tuple[str, float]]:
        """
        Generate sample captions for evaluation.
        
        Args:
            image_features: CNN features for sample images
            num_samples: Number of samples to generate
            use_beam_search: Whether to use beam search or greedy decoding
            
        Returns:
            List of (caption, confidence) tuples
        """
        try:
            if not self.caption_generator._model_loaded:
                raise ModelTrainerError("Model not trained or loaded")
            
            results = []
            num_samples = min(num_samples, len(image_features))
            
            for i in range(num_samples):
                features = image_features[i:i+1]  # Keep batch dimension
                
                if use_beam_search:
                    caption, confidence = self.caption_generator.generate_caption_beam_search(features)
                else:
                    caption, confidence = self.caption_generator.generate_caption_greedy(features)
                
                results.append((caption, confidence))
                logger.info(f"Sample {i+1}: '{caption}' (confidence: {confidence:.3f})")
            
            return results
            
        except Exception as e:
            logger.error(f"Sample generation failed: {str(e)}")
            raise ModelTrainerError(f"Sample generation failed: {str(e)}")
    
    def save_training_history(self, filepath: str) -> None:
        """
        Save training history to file.
        
        Args:
            filepath: Path to save the history
        """
        if self.training_history is None:
            raise ModelTrainerError("No training history to save")
        
        try:
            with open(filepath, 'w') as f:
                json.dump(self.training_history, f, indent=2)
            
            logger.info(f"Training history saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save training history: {str(e)}")
            raise ModelTrainerError(f"Training history saving failed: {str(e)}")
    
    def load_training_history(self, filepath: str) -> Dict:
        """
        Load training history from file.
        
        Args:
            filepath: Path to load the history from
            
        Returns:
            Dict: Training history
        """
        try:
            with open(filepath, 'r') as f:
                self.training_history = json.load(f)
            
            logger.info(f"Training history loaded from {filepath}")
            return self.training_history
            
        except Exception as e:
            logger.error(f"Failed to load training history: {str(e)}")
            raise ModelTrainerError(f"Training history loading failed: {str(e)}")


def create_sample_training_data(num_samples: int = 100) -> Tuple[np.ndarray, List[str]]:
    """
    Create sample training data for testing purposes.
    
    Args:
        num_samples: Number of samples to create
        
    Returns:
        Tuple of (features, captions)
    """
    # Generate random CNN features
    features = np.random.randn(num_samples, 2048)
    
    # Generate sample captions
    sample_words = [
        "a", "the", "person", "man", "woman", "dog", "cat", "car", "house", "tree",
        "sitting", "standing", "walking", "red", "blue", "green", "big", "small",
        "on", "in", "with", "near", "next to", "street", "park", "beach"
    ]
    
    captions = []
    for _ in range(num_samples):
        # Generate random caption
        length = np.random.randint(3, 8)
        words = np.random.choice(sample_words, size=length, replace=True)
        caption = " ".join(words)
        captions.append(caption)
    
    return features, captions