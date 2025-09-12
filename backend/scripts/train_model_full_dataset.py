#!/usr/bin/env python3
"""
Full dataset training script for the AI image caption generator.
Uses all 8000+ images with proper batching and memory management.
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


def load_cnn_model():
    """Load pre-trained CNN model for feature extraction."""
    model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    return model


def extract_features_batch(image_paths, cnn_model, batch_size=32):
    """Extract CNN features from a batch of images."""
    features = []
    
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Extracting features"):
        batch_paths = image_paths[i:i+batch_size]
        batch_images = []
        valid_indices = []
        
        # Load and preprocess batch
        for j, image_path in enumerate(batch_paths):
            try:
                img = image.load_img(image_path, target_size=(224, 224))
                img_array = image.img_to_array(img)
                img_array = preprocess_input(img_array)
                batch_images.append(img_array)
                valid_indices.append(i + j)
            except Exception as e:
                logger.warning(f"Failed to load {image_path}: {e}")
                continue
        
        if batch_images:
            # Extract features for batch
            batch_array = np.array(batch_images)
            batch_features = cnn_model.predict(batch_array, verbose=0)
            
            # Add to features list
            for k, feature in enumerate(batch_features):
                features.append((valid_indices[k], feature.flatten()))
    
    return features


def save_features_cache(features, captions, cache_path):
    """Save extracted features to cache file."""
    cache_data = {
        'features': features,
        'captions': captions
    }
    
    with open(cache_path, 'wb') as f:
        pickle.dump(cache_data, f)
    
    logger.info(f"Features cached to {cache_path}")


def load_features_cache(cache_path):
    """Load features from cache file."""
    with open(cache_path, 'rb') as f:
        cache_data = pickle.load(f)
    
    logger.info(f"Features loaded from cache {cache_path}")
    return cache_data['features'], cache_data['captions']


def load_full_dataset(data_dir: str = "data", use_cache: bool = True):
    """Load the complete dataset with CNN feature extraction."""
    
    cache_path = Path(data_dir) / "features_cache.pkl"
    
    # Try to load from cache first
    if use_cache and cache_path.exists():
        logger.info("Loading features from cache...")
        try:
            return load_features_cache(cache_path)
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}. Extracting features...")
    
    # Load captions
    captions_file = Path(data_dir) / "captions.txt"
    images_dir = Path(data_dir) / "Images"
    
    logger.info("Loading captions from dataset...")
    df = pd.read_csv(captions_file)
    
    # Group captions by image - use all captions, not just first one
    captions_dict = {}
    for _, row in df.iterrows():
        image_name = row['image']
        caption = row['caption']
        if image_name not in captions_dict:
            captions_dict[image_name] = []
        captions_dict[image_name].append(caption)
    
    logger.info(f"Loaded captions for {len(captions_dict)} unique images")
    
    # Prepare image paths and corresponding captions
    image_paths = []
    all_captions = []
    
    for image_name, image_captions in captions_dict.items():
        image_path = images_dir / image_name
        if image_path.exists():
            # Use all captions for each image (data augmentation)
            for caption in image_captions:
                image_paths.append(str(image_path))
                all_captions.append(caption)
    
    logger.info(f"Total training samples: {len(image_paths)} (with caption augmentation)")
    
    # Load CNN model for feature extraction
    logger.info("Loading ResNet50 for feature extraction...")
    cnn_model = load_cnn_model()
    
    # Extract features in batches
    logger.info("Extracting CNN features from all images...")
    feature_results = extract_features_batch(image_paths, cnn_model, batch_size=32)
    
    # Organize features and captions
    features = []
    final_captions = []
    
    for idx, feature_vector in feature_results:
        features.append(feature_vector)
        final_captions.append(all_captions[idx])
    
    features = np.array(features)
    
    logger.info(f"Successfully extracted features for {len(features)} samples")
    
    # Cache the results
    if use_cache:
        save_features_cache(features, final_captions, cache_path)
    
    return features, final_captions


def main():
    """Main training function."""
    
    print("🚀 AI Image Caption Generator - Full Dataset Training")
    print("=" * 60)
    print("Training on complete dataset with ~8000+ images")
    print("=" * 60)
    
    try:
        # Load full dataset
        logger.info("Loading complete dataset...")
        features, captions = load_full_dataset()
        
        logger.info(f"Dataset loaded: {len(features)} samples")
        logger.info(f"Feature shape: {features.shape}")
        logger.info("Sample captions:")
        for i, caption in enumerate(captions[:5]):
            logger.info(f"  {i+1}. {caption}")
        
        # Initialize components with larger vocabulary for full dataset
        caption_generator = LSTMCaptionGenerator(
            vocab_size=5000,  # Larger vocabulary for full dataset
            max_length=20,    # Longer sequences
            embedding_dim=300,  # Richer embeddings
            lstm_units=512    # More LSTM units
        )
        trainer = ModelTrainer(caption_generator)
        
        # Create models directory
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        # Setup model checkpointing
        checkpoint_path = models_dir / "full_dataset_best_model.keras"
        
        # Train model with full dataset
        logger.info("Starting full dataset training...")
        logger.info("Training configuration:")
        logger.info(f"  - Samples: {len(features)}")
        logger.info(f"  - Vocabulary size: 5000")
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
        final_model_path = models_dir / "full_dataset_caption_model.keras"
        caption_generator.model.save(str(final_model_path))
        
        # Save vocabulary
        vocab_path = models_dir / "full_dataset_vocabulary.json"
        caption_generator.vocabulary.save_vocabulary(str(vocab_path))
        
        # Save training history
        history_path = models_dir / "full_dataset_training_history.json"
        trainer.save_training_history(str(history_path))
        
        print(f"\n🎉 Full dataset training completed successfully!")
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
            use_beam_search=True
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
        
        return 0
        
    except Exception as e:
        logger.error(f"Full dataset training failed: {e}")
        print(f"\n❌ Training failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())