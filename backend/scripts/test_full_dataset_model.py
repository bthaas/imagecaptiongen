#!/usr/bin/env python3
"""
Test script for the full dataset trained model.
"""

import os
import sys
import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import logging
import json

# Add parent directory to path to import our modules
sys.path.append(str(Path(__file__).parent.parent))

from app.services.caption_generator import LSTMCaptionGenerator

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_trained_model():
    """Load the trained model with correct vocabulary size."""
    
    # Load vocabulary to get the correct size
    vocab_path = Path("models/full_dataset_vocabulary.json")
    
    if not vocab_path.exists():
        raise FileNotFoundError("Vocabulary file not found. Please run training first.")
    
    with open(vocab_path, 'r') as f:
        vocab_data = json.load(f)
    
    vocab_size = len(vocab_data['word_to_idx'])
    logger.info(f"Loading model with vocabulary size: {vocab_size}")
    
    # Initialize caption generator with correct vocab size
    caption_generator = LSTMCaptionGenerator(
        vocab_size=vocab_size,  # Use actual vocabulary size
        max_length=20,
        embedding_dim=300,
        lstm_units=512
    )
    
    # Load vocabulary
    caption_generator.vocabulary.load_vocabulary(str(vocab_path))
    
    # Load trained model
    model_path = Path("models/full_dataset_best_model.keras")
    if not model_path.exists():
        raise FileNotFoundError("Trained model not found. Please run training first.")
    
    caption_generator.model = tf.keras.models.load_model(str(model_path))
    caption_generator._model_loaded = True
    
    logger.info("Model loaded successfully!")
    return caption_generator


def extract_features_from_image(image_path):
    """Extract CNN features from a single image."""
    # Load CNN model
    cnn_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    
    # Load and preprocess image
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    # Extract features
    features = cnn_model.predict(img_array, verbose=0)
    return features


def test_model_with_sample_images():
    """Test the trained model with sample images."""
    
    try:
        # Load trained model
        logger.info("Loading trained model...")
        caption_generator = load_trained_model()
        
        # Test with some sample images from the dataset
        images_dir = Path("data/Images")
        sample_images = list(images_dir.glob("*.jpg"))[:5]  # Test with first 5 images
        
        if not sample_images:
            logger.warning("No sample images found in data/Images/")
            return
        
        logger.info(f"Testing with {len(sample_images)} sample images...")
        
        for i, image_path in enumerate(sample_images):
            logger.info(f"\nTesting image {i+1}: {image_path.name}")
            
            try:
                # Extract features
                features = extract_features_from_image(image_path)
                
                # Generate caption using greedy decoding (more stable)
                caption, confidence = caption_generator.generate_caption_greedy(features)
                
                print(f"Image: {image_path.name}")
                print(f"Generated Caption: '{caption}'")
                print(f"Confidence: {confidence:.3f}")
                print("-" * 50)
                
            except Exception as e:
                logger.error(f"Failed to process {image_path.name}: {e}")
                continue
        
        logger.info("Model testing completed!")
        
    except Exception as e:
        logger.error(f"Model testing failed: {e}")
        return


def main():
    """Main function."""
    
    print("🚀 Testing Full Dataset Trained Model")
    print("=" * 50)
    
    test_model_with_sample_images()


if __name__ == "__main__":
    main()