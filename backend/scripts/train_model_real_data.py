#!/usr/bin/env python3
"""
Training script using real image captions with extracted CNN features.
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
import logging
from PIL import Image

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


def extract_features(image_path, cnn_model):
    """Extract CNN features from an image."""
    try:
        # Load and preprocess image
        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        
        # Extract features
        features = cnn_model.predict(img_array, verbose=0)
        return features.flatten()
    except Exception as e:
        logger.warning(f"Failed to extract features from {image_path}: {e}")
        return None


def load_real_dataset(data_dir: str = "data", max_samples: int = 500):
    """Load real dataset with CNN feature extraction."""
    
    # Load captions
    captions_file = Path(data_dir) / "captions.txt"
    images_dir = Path(data_dir) / "Images"
    
    logger.info("Loading captions...")
    df = pd.read_csv(captions_file)
    
    # Group captions by image and take first caption for each
    captions_dict = {}
    for _, row in df.iterrows():
        image_name = row['image']
        caption = row['caption']
        if image_name not in captions_dict:
            captions_dict[image_name] = caption
    
    # Load CNN model for feature extraction
    logger.info("Loading CNN model for feature extraction...")
    cnn_model = load_cnn_model()
    
    # Process images and extract features
    features = []
    captions = []
    processed_count = 0
    
    logger.info(f"Processing up to {max_samples} images...")
    
    for image_name, caption in list(captions_dict.items())[:max_samples]:
        image_path = images_dir / image_name
        
        if image_path.exists():
            # Extract CNN features
            feature_vector = extract_features(image_path, cnn_model)
            
            if feature_vector is not None:
                features.append(feature_vector)
                captions.append(caption)
                processed_count += 1
                
                if processed_count % 50 == 0:
                    logger.info(f"Processed {processed_count}/{max_samples} images")
        
        if processed_count >= max_samples:
            break
    
    logger.info(f"Successfully processed {len(features)} images")
    return np.array(features), captions


def main():
    """Main training function."""
    
    print("🚀 AI Image Caption Generator - Real Data Training")
    print("=" * 50)
    
    try:
        # Load real dataset with CNN features
        logger.info("Loading real dataset...")
        features, captions = load_real_dataset(max_samples=300)  # Start with 300 samples
        
        logger.info(f"Loaded {len(features)} samples")
        logger.info("Sample captions:")
        for i, caption in enumerate(captions[:5]):
            logger.info(f"  {i+1}. {caption}")
        
        # Initialize components
        caption_generator = LSTMCaptionGenerator(vocab_size=2000, max_length=16)
        trainer = ModelTrainer(caption_generator)
        
        # Train model
        logger.info("Starting model training...")
        history = trainer.train_model(
            image_features=features,
            captions=captions,
            validation_split=0.2,
            epochs=15,  # More epochs for real data
            batch_size=16
        )
        
        # Create models directory
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        # Save model
        model_path = models_dir / "real_data_caption_model.keras"
        caption_generator.model.save(str(model_path))
        
        # Save vocabulary
        vocab_path = models_dir / "real_data_vocabulary.json"
        caption_generator.vocabulary.save_vocabulary(str(vocab_path))
        
        print(f"\n🎉 Training completed successfully!")
        print(f"📁 Model saved to: {model_path}")
        print(f"📁 Vocabulary saved to: {vocab_path}")
        
        # Test generation with some sample features
        logger.info("Testing caption generation...")
        sample_features = features[:5]
        results = trainer.generate_sample_captions(sample_features, num_samples=5, use_beam_search=True)
        
        print("\n📝 Sample Generated Captions:")
        print("-" * 50)
        for i, (caption, confidence) in enumerate(results):
            print(f"{i+1}. '{caption}' (confidence: {confidence:.3f})")
        
        # Show corresponding original captions for comparison
        print("\n📝 Original Captions for Comparison:")
        print("-" * 50)
        for i, caption in enumerate(captions[:5]):
            print(f"{i+1}. '{caption}'")
        
        return 0
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        print(f"\n❌ Training failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())