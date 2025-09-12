"""
LSTM Caption Generator for generating natural language captions from CNN features.
Implements sequence generation with vocabulary management and decoding strategies.
"""

import os
import json
import logging
import numpy as np
from typing import List, Dict, Optional, Tuple, Union
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    LSTM, Dense, Embedding, RepeatVector, TimeDistributed,
    Input, Concatenate, Attention, Dropout
)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import pickle
import threading

logger = logging.getLogger(__name__)


class CaptionGeneratorError(Exception):
    """Custom exception for caption generator errors."""
    pass


class VocabularyManager:
    """
    Manages vocabulary and tokenization for caption generation.
    """
    
    def __init__(self, vocab_size: int = 10000, max_length: int = 20):
        """
        Initialize vocabulary manager.
        
        Args:
            vocab_size: Maximum vocabulary size
            max_length: Maximum caption length
        """
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.tokenizer: Optional[Tokenizer] = None
        self.word_to_idx: Dict[str, int] = {}
        self.idx_to_word: Dict[int, str] = {}
        self.start_token = "<start>"
        self.end_token = "<end>"
        self.unk_token = "<unk>"
        self.pad_token = "<pad>"
        
    def build_vocabulary(self, captions: List[str]) -> None:
        """
        Build vocabulary from training captions.
        
        Args:
            captions: List of training captions
        """
        try:
            # Add special tokens to captions
            processed_captions = []
            for caption in captions:
                processed_caption = f"{self.start_token} {caption.lower().strip()} {self.end_token}"
                processed_captions.append(processed_caption)
            
            # Create tokenizer
            self.tokenizer = Tokenizer(
                num_words=self.vocab_size,
                oov_token=self.unk_token,
                filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
            )
            
            # Fit on texts
            self.tokenizer.fit_on_texts(processed_captions)
            
            # Build word mappings
            self.word_to_idx = self.tokenizer.word_index.copy()
            self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
            
            # Ensure pad token is at index 0
            if self.pad_token not in self.word_to_idx:
                self.word_to_idx[self.pad_token] = 0
                self.idx_to_word[0] = self.pad_token
            
            # Ensure special tokens are in vocabulary
            for token in [self.start_token, self.end_token]:
                if token not in self.word_to_idx:
                    # Find next available index
                    max_idx = max(self.word_to_idx.values()) if self.word_to_idx else 0
                    new_idx = max_idx + 1
                    self.word_to_idx[token] = new_idx
                    self.idx_to_word[new_idx] = token
            
            logger.info(f"Vocabulary built with {len(self.word_to_idx)} words")
            
        except Exception as e:
            logger.error(f"Failed to build vocabulary: {str(e)}")
            raise CaptionGeneratorError(f"Vocabulary building failed: {str(e)}")
    
    def load_default_vocabulary(self) -> None:
        """
        Load a default vocabulary for inference when no training data is available.
        This creates a basic vocabulary with common words for image captioning.
        """
        try:
            # Default vocabulary with common image captioning words
            default_words = [
                self.pad_token, self.start_token, self.end_token, self.unk_token,
                "a", "an", "the", "is", "are", "and", "of", "in", "on", "at", "with",
                "man", "woman", "person", "people", "child", "boy", "girl",
                "dog", "cat", "bird", "horse", "car", "truck", "bike", "motorcycle",
                "house", "building", "tree", "flower", "grass", "water", "sky", "cloud",
                "red", "blue", "green", "yellow", "black", "white", "brown", "orange",
                "big", "small", "tall", "short", "old", "young", "new", "beautiful",
                "sitting", "standing", "walking", "running", "playing", "eating",
                "street", "road", "park", "beach", "mountain", "city", "field",
                "wearing", "holding", "looking", "smiling", "happy", "sunny", "cloudy"
            ]
            
            # Create word mappings
            self.word_to_idx = {word: idx for idx, word in enumerate(default_words)}
            self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
            
            # Create a simple tokenizer
            self.tokenizer = Tokenizer(
                num_words=len(default_words),
                oov_token=self.unk_token
            )
            self.tokenizer.word_index = self.word_to_idx.copy()
            
            logger.info(f"Default vocabulary loaded with {len(self.word_to_idx)} words")
            
        except Exception as e:
            logger.error(f"Failed to load default vocabulary: {str(e)}")
            raise CaptionGeneratorError(f"Default vocabulary loading failed: {str(e)}")
    
    def text_to_sequences(self, texts: List[str]) -> List[List[int]]:
        """
        Convert texts to sequences of token indices.
        
        Args:
            texts: List of text strings
            
        Returns:
            List of token sequences
        """
        if self.tokenizer is None:
            raise CaptionGeneratorError("Tokenizer not initialized")
        
        return self.tokenizer.texts_to_sequences(texts)
    
    def sequences_to_text(self, sequences: List[List[int]]) -> List[str]:
        """
        Convert sequences of token indices back to text.
        
        Args:
            sequences: List of token sequences
            
        Returns:
            List of text strings
        """
        texts = []
        for sequence in sequences:
            words = []
            for idx in sequence:
                if idx in self.idx_to_word and idx != 0:  # Skip padding
                    word = self.idx_to_word[idx]
                    if word not in [self.start_token, self.end_token, self.pad_token]:
                        words.append(word)
                    elif word == self.end_token:
                        break
            texts.append(" ".join(words))
        return texts
    
    def save_vocabulary(self, filepath: str) -> None:
        """Save vocabulary to file."""
        vocab_data = {
            'word_to_idx': self.word_to_idx,
            'idx_to_word': self.idx_to_word,
            'vocab_size': self.vocab_size,
            'max_length': self.max_length
        }
        
        with open(filepath, 'w') as f:
            json.dump(vocab_data, f)
        
        logger.info(f"Vocabulary saved to {filepath}")
    
    def load_vocabulary(self, filepath: str) -> None:
        """Load vocabulary from file."""
        with open(filepath, 'r') as f:
            vocab_data = json.load(f)
        
        self.word_to_idx = vocab_data['word_to_idx']
        self.idx_to_word = {int(k): v for k, v in vocab_data['idx_to_word'].items()}
        self.vocab_size = vocab_data['vocab_size']
        self.max_length = vocab_data['max_length']
        
        # Recreate tokenizer
        self.tokenizer = Tokenizer(num_words=self.vocab_size, oov_token=self.unk_token)
        self.tokenizer.word_index = self.word_to_idx.copy()
        
        logger.info(f"Vocabulary loaded from {filepath}")


class LSTMCaptionGenerator:
    """
    LSTM-based caption generator that combines CNN features with sequence generation.
    """
    
    _instance: Optional['LSTMCaptionGenerator'] = None
    _lock = threading.Lock()
    
    def __init__(self, vocab_size: int = 10000, max_length: int = 20, 
                 embedding_dim: int = 256, lstm_units: int = 512):
        """
        Initialize LSTM caption generator.
        
        Args:
            vocab_size: Vocabulary size
            max_length: Maximum caption length
            embedding_dim: Word embedding dimension
            lstm_units: LSTM hidden units
        """
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.feature_dim = 2048  # CNN feature dimension from ResNet50
        
        self.vocabulary = VocabularyManager(vocab_size, max_length)
        self.model: Optional[Model] = None
        self._model_loaded = False
        
    @classmethod
    def get_instance(cls) -> 'LSTMCaptionGenerator':
        """Get singleton instance of LSTMCaptionGenerator."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    def build_model(self) -> Model:
        """
        Build the LSTM caption generation model.
        
        Returns:
            tf.keras.Model: Caption generation model
        """
        try:
            # CNN feature input
            feature_input = Input(shape=(self.feature_dim,), name='feature_input')
            
            # Text sequence input (max_length - 1 for input sequences)
            text_input = Input(shape=(self.max_length - 1,), name='text_input')
            
            # Feature processing - project to embedding dimension
            feature_dense = Dense(self.embedding_dim, activation='relu')(feature_input)
            feature_repeat = RepeatVector(self.max_length - 1)(feature_dense)
            
            # Text embedding
            text_embedding = Embedding(
                input_dim=self.vocab_size,
                output_dim=self.embedding_dim
            )(text_input)
            
            # Add features to each time step of text embedding
            combined = tf.keras.layers.Add()([feature_repeat, text_embedding])
            
            # LSTM layers
            lstm1 = LSTM(self.lstm_units, return_sequences=True, dropout=0.2)(combined)
            lstm2 = LSTM(self.lstm_units, return_sequences=True, dropout=0.2)(lstm1)
            
            # Output layer
            output = TimeDistributed(
                Dense(self.vocab_size, activation='softmax')
            )(lstm2)
            
            # Create model
            self.model = Model(inputs=[feature_input, text_input], outputs=output)
            
            # Compile model
            self.model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            self._model_loaded = True
            logger.info("LSTM caption generation model built successfully")
            
            return self.model
            
        except Exception as e:
            logger.error(f"Failed to build model: {str(e)}")
            raise CaptionGeneratorError(f"Model building failed: {str(e)}")
    
    def generate_caption_greedy(self, cnn_features: np.ndarray, 
                               temperature: float = 1.0) -> Tuple[str, float]:
        """
        Generate caption using greedy decoding.
        
        Args:
            cnn_features: CNN feature vector of shape (1, 2048)
            temperature: Sampling temperature for diversity
            
        Returns:
            Tuple of (caption, confidence_score)
        """
        if not self._model_loaded or self.model is None:
            raise CaptionGeneratorError("Model not loaded")
        
        try:
            # Initialize with start token
            start_idx = self.vocabulary.word_to_idx.get(self.vocabulary.start_token, 1)
            sequence = [start_idx]
            confidence_scores = []
            
            for _ in range(self.max_length - 1):
                # Prepare input (pad to max_length - 1 to match training)
                padded_sequence = pad_sequences([sequence], maxlen=self.max_length - 1, padding='post')
                
                # Predict next word
                predictions = self.model.predict([cnn_features, padded_sequence], verbose=0)
                
                # Get prediction for current position
                next_word_probs = predictions[0, len(sequence) - 1, :]
                
                # Apply temperature
                if temperature != 1.0:
                    next_word_probs = np.log(next_word_probs + 1e-8) / temperature
                    next_word_probs = np.exp(next_word_probs)
                    next_word_probs = next_word_probs / np.sum(next_word_probs)
                
                # Get next word (greedy)
                next_word_idx = np.argmax(next_word_probs)
                confidence_scores.append(float(next_word_probs[next_word_idx]))
                
                # Check for end token
                if next_word_idx == self.vocabulary.word_to_idx.get(self.vocabulary.end_token, 2):
                    break
                
                sequence.append(next_word_idx)
            
            # Convert to text
            caption = self.vocabulary.sequences_to_text([sequence])[0]
            
            # Calculate average confidence
            avg_confidence = np.mean(confidence_scores) if confidence_scores else 0.0
            
            return caption, avg_confidence
            
        except Exception as e:
            logger.error(f"Caption generation failed: {str(e)}")
            raise CaptionGeneratorError(f"Caption generation failed: {str(e)}")
    
    def generate_caption_beam_search(self, cnn_features: np.ndarray, 
                                   beam_width: int = 3, temperature: float = 1.0) -> Tuple[str, float]:
        """
        Generate caption using beam search decoding.
        
        Args:
            cnn_features: CNN feature vector of shape (1, 2048)
            beam_width: Number of beams to keep
            temperature: Sampling temperature for diversity
            
        Returns:
            Tuple of (caption, confidence_score)
        """
        if not self._model_loaded or self.model is None:
            raise CaptionGeneratorError("Model not loaded")
        
        try:
            start_idx = self.vocabulary.word_to_idx.get(self.vocabulary.start_token, 1)
            end_idx = self.vocabulary.word_to_idx.get(self.vocabulary.end_token, 2)
            
            # Initialize beams: (sequence, score)
            beams = [([start_idx], 0.0)]
            completed_beams = []
            
            for step in range(self.max_length - 1):
                candidates = []
                
                for sequence, score in beams:
                    if len(sequence) > 0 and sequence[-1] == end_idx:
                        completed_beams.append((sequence, score))
                        continue
                    
                    # Prepare input (pad to max_length - 1 to match training)
                    padded_sequence = pad_sequences([sequence], maxlen=self.max_length - 1, padding='post')
                    
                    # Predict next word
                    predictions = self.model.predict([cnn_features, padded_sequence], verbose=0)
                    next_word_probs = predictions[0, len(sequence) - 1, :]
                    
                    # Apply temperature
                    if temperature != 1.0:
                        next_word_probs = np.log(next_word_probs + 1e-8) / temperature
                        next_word_probs = np.exp(next_word_probs)
                        next_word_probs = next_word_probs / np.sum(next_word_probs)
                    
                    # Get top candidates
                    top_indices = np.argsort(next_word_probs)[-beam_width:]
                    
                    for idx in top_indices:
                        new_sequence = sequence + [idx]
                        new_score = score + np.log(next_word_probs[idx] + 1e-8)
                        candidates.append((new_sequence, new_score))
                
                # Keep top beams
                if candidates:
                    beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_width]
                
                # Stop if all beams are completed
                if not beams:
                    break
            
            # Add remaining beams to completed
            completed_beams.extend(beams)
            
            if not completed_beams:
                return "a photo", 0.1
            
            # Get best beam
            best_sequence, best_score = max(completed_beams, key=lambda x: x[1])
            
            # Convert to text
            caption = self.vocabulary.sequences_to_text([best_sequence])[0]
            
            # Convert log score to confidence
            confidence = np.exp(best_score / len(best_sequence)) if len(best_sequence) > 0 else 0.0
            
            return caption, float(confidence)
            
        except Exception as e:
            logger.error(f"Beam search caption generation failed: {str(e)}")
            raise CaptionGeneratorError(f"Beam search caption generation failed: {str(e)}")
    
    def initialize_for_inference(self) -> None:
        """
        Initialize the model for inference with default vocabulary.
        This is used when no trained model is available.
        """
        try:
            # Load default vocabulary
            self.vocabulary.load_default_vocabulary()
            
            # Build model architecture
            self.build_model()
            
            # Initialize with random weights (for demo purposes)
            # In production, this would load pre-trained weights
            logger.warning("Using randomly initialized model weights - captions will be poor quality")
            
        except Exception as e:
            logger.error(f"Failed to initialize for inference: {str(e)}")
            raise CaptionGeneratorError(f"Inference initialization failed: {str(e)}")
    
    def save_model(self, model_path: str, vocab_path: str) -> None:
        """
        Save the trained model and vocabulary.
        
        Args:
            model_path: Path to save the model
            vocab_path: Path to save the vocabulary
        """
        if self.model is None:
            raise CaptionGeneratorError("No model to save")
        
        try:
            self.model.save(model_path)
            self.vocabulary.save_vocabulary(vocab_path)
            logger.info(f"Model saved to {model_path}, vocabulary saved to {vocab_path}")
            
        except Exception as e:
            logger.error(f"Failed to save model: {str(e)}")
            raise CaptionGeneratorError(f"Model saving failed: {str(e)}")
    
    def load_model(self, model_path: str, vocab_path: str) -> None:
        """
        Load a trained model and vocabulary.
        
        Args:
            model_path: Path to the saved model
            vocab_path: Path to the saved vocabulary
        """
        try:
            # Load vocabulary first
            self.vocabulary.load_vocabulary(vocab_path)
            
            # Load model
            self.model = tf.keras.models.load_model(model_path)
            self._model_loaded = True
            
            logger.info(f"Model loaded from {model_path}, vocabulary loaded from {vocab_path}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise CaptionGeneratorError(f"Model loading failed: {str(e)}")
    
    def get_model_info(self) -> Dict:
        """Get information about the caption generation model."""
        return {
            "loaded": self._model_loaded,
            "vocab_size": self.vocab_size,
            "max_length": self.max_length,
            "embedding_dim": self.embedding_dim,
            "lstm_units": self.lstm_units,
            "feature_dim": self.feature_dim,
            "vocabulary_loaded": self.vocabulary.tokenizer is not None
        }