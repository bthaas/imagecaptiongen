#!/usr/bin/env python3
"""
Fixed full dataset training script with correct vocabulary size.
Uses cached features for faster training.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import logging
from PIL import Image
import json
import pickle
from tqdm import tqdm

# Add parent directory to path to import our modules
sys.path.append(str(Path(__file__).parent.parent))

from app.services.caption_generator import LSTMCaptionGenerator
from app.services.model_trainer import ModelTrainer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_features_cache(cache_path):
    """Load features from cache file."""
    with open(cache_path, 'rb') as f:
        cache_data = pickle.load(f)
    
    logger.info(f"Features loaded from cache {cache_path}")
    return cache_data['features'], cache_data['captions']


def get_vocabulary_size(captions):
    """Pre-calculate vocabulary size from captions."""
    from app.services.caption_generator import VocabularyManager
    
    # Create temporary vocabulary manager to calculate size
    temp_vocab = VocabularyManager(vocab_size=50000, max_length=20)  # Large enough
    temp_vocab.build_vocabulary(captions)
    
    actual_vocab_size = len(temp_vocab.word_to_idx)
    logger.info(f"Calculated vocabulary size: {actual_vocab_size}")
    
    return actual_vocab_size


def main():
    """Main training function with correct vocabulary size."""
    
    print("🚀 AI Image Caption Generator - Fixed Full Dataset Training")
    print("=" * 65)
    print("Training with correct vocabulary size using cached features")
    print("=" * 65)
    
    try:
        # Load cached features (much faster!)
        cache_path = Path("data/features_cache.pkl")
        
        if not cache_path.exists():
            logger.error("Features cache not found. Please run the full dataset training first.")
            return 1
        
        logger.info("Loading cached features...")
        features, captions = load_features_cache(cache_path)
        
        logger.info(f"Dataset loaded: {len(features)} samples")
        logger.info(f"Feature shape: {features.shape}")
        
        # Pre-calculate correct vocabulary size
        logger.info("Calculating vocabulary size...")
        vocab_size = get_vocabulary_size(captions)
        
        # Add some buffer for safety
        vocab_size = min(vocab_size + 100, 10000)  # Cap at 10k for memory efficiency
        
        logger.info("Sample captions:")
        for i, caption in enumerate(captions[:5]):
            logger.info(f"  {i+1}. {caption}")
        
        # Initialize components with CORRECT vocabulary size
        caption_generator = LSTMCaptionGenerator(
            vocab_size=vocab_size,  # Use calculated vocabulary size
            max_length=20,          # Longer sequences
            embedding_dim=300,      # Rich embeddings
            lstm_units=512          # More LSTM units
        )
        trainer = ModelTrainer(caption_generator)
        
        # Create models directory
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        # Setup model checkpointing
        checkpoint_path = models_dir / "full_dataset_fixed_best_model.keras"
        
        # Train model with correct vocabulary
        logger.info("Starting fixed full dataset training...")
        logger.info("Training configuration:")
        logger.info(f"  - Samples: {len(features)}")
        logger.info(f"  - Vocabulary size: {vocab_size}")
        logger.info(f"  - Max sequence length: 20")
        logger.info(f"  - Epochs: 25")
        logger.info(f"  - Batch size: 32")
        logger.info(f"  - Validation split: 0.15")
        
        history = trainer.train_model(
            image_features=features,
            captions=captions,
            validation_split=0.15,  # 15% for validation
            epochs=25,              # More epochs for full dataset
            batch_size=32,          # Reasonable batch size
            save_path=str(checkpoint_path)
        )
        
        # Save final model
        final_model_path = models_dir / "full_dataset_fixed_caption_model.keras"
        caption_generator.model.save(str(final_model_path))
        
        # Save vocabulary
        vocab_path = models_dir / "full_dataset_fixed_vocabulary.json"
        caption_generator.vocabulary.save_vocabulary(str(vocab_path))
        
        # Save training history
        history_path = models_dir / "full_dataset_fixed_training_history.json"
        trainer.save_training_history(str(history_path))
        
        print(f"\n🎉 Fixed full dataset training completed successfully!")
        print(f"📁 Best model saved to: {checkpoint_path}")
        print(f"📁 Final model saved to: {final_model_path}")
        print(f"📁 Vocabulary saved to: {vocab_path}")
        print(f"📁 Training history saved to: {history_path}")
        
        # Test generation with diverse samples
        logger.info("Testing caption generation on diverse samples...")
        
        # Select diverse test samples (every 1000th sample)
        test_indices = list(range(0, len(features), len(features) // 10))[:10]
        test_features = features[test_indices]
        test_captions_original = [captions[i] for i in test_indices]
        
        results = trainer.generate_sample_captions(
            test_features, 
            num_samples=len(test_features), 
            use_beam_search=False  # Use greedy for stability
        )
        
        print(f"\n📝 Generated Captions vs Original:")
        print("=" * 80)
        for i, ((generated, confidence), original) in enumerate(zip(results, test_captions_original)):
            print(f"\n{i+1}. Generated: '{generated}' (confidence: {confidence:.3f})")
            print(f"   Original:  '{original}'")
        
        # Print training summary
        if history:
            final_loss = history['loss'][-1]
            final_val_loss = history['val_loss'][-1]
            final_acc = history['accuracy'][-1]
            final_val_acc = history['val_accuracy'][-1]
            
            print(f"\n📊 Training Summary:")
            print(f"   Final Training Loss: {final_loss:.4f}")
            print(f"   Final Validation Loss: {final_val_loss:.4f}")
            print(f"   Final Training Accuracy: {final_acc:.4f}")
            print(f"   Final Validation Accuracy: {final_val_acc:.4f}")
        
        # Verify vocabulary size matches
        actual_vocab_size = len(caption_generator.vocabulary.word_to_idx)
        model_vocab_size = caption_generator.vocab_size
        
        print(f"\n✅ Vocabulary Verification:")
        print(f"   Model vocabulary size: {model_vocab_size}")
        print(f"   Actual vocabulary size: {actual_vocab_size}")
        print(f"   Match: {'✅ YES' if actual_vocab_size <= model_vocab_size else '❌ NO'}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Fixed full dataset training failed: {e}")
        print(f"\n❌ Training failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())