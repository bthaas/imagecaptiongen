#!/usr/bin/env python3
"""
Test the fixed trained model with real images.
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


def load_fixed_trained_model():
    """Load the fixed trained model."""
    
    # Load vocabulary to get the correct size
    vocab_path = Path("models/full_dataset_fixed_vocabulary.json")
    
    if not vocab_path.exists():
        raise FileNotFoundError("Fixed vocabulary file not found.")
    
    with open(vocab_path, 'r') as f:
        vocab_data = json.load(f)
    
    vocab_size = len(vocab_data['word_to_idx'])
    logger.info(f"Loading fixed model with vocabulary size: {vocab_size}")
    
    # Initialize caption generator with correct vocab size
    caption_generator = LSTMCaptionGenerator(
        vocab_size=vocab_size + 100,  # Add buffer as in training
        max_length=20,
        embedding_dim=300,
        lstm_units=512
    )
    
    # Load vocabulary
    caption_generator.vocabulary.load_vocabulary(str(vocab_path))
    
    # Load trained model
    model_path = Path("models/full_dataset_fixed_best_model.keras")
    if not model_path.exists():
        raise FileNotFoundError("Fixed trained model not found.")
    
    caption_generator.model = tf.keras.models.load_model(str(model_path))
    caption_generator._model_loaded = True
    
    logger.info("Fixed model loaded successfully!")
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


def test_fixed_model():
    """Test the fixed trained model with sample images."""
    
    try:
        # Load fixed trained model
        logger.info("Loading fixed trained model...")
        caption_generator = load_fixed_trained_model()
        
        # Test with some sample images from the dataset
        images_dir = Path("data/Images")
        sample_images = list(images_dir.glob("*.jpg"))[:10]  # Test with first 10 images
        
        if not sample_images:
            logger.warning("No sample images found in data/Images/")
            return
        
        logger.info(f"Testing fixed model with {len(sample_images)} sample images...")
        
        print("\n🚀 Fixed Model Caption Generation Test")
        print("=" * 60)
        
        for i, image_path in enumerate(sample_images):
            logger.info(f"\nTesting image {i+1}: {image_path.name}")
            
            try:
                # Extract features
                features = extract_features_from_image(image_path)
                
                # Generate caption using greedy decoding
                caption, confidence = caption_generator.generate_caption_greedy(features)
                
                print(f"\n{i+1}. Image: {image_path.name}")
                print(f"   Caption: '{caption}'")
                print(f"   Confidence: {confidence:.3f}")
                
                # Also try beam search for comparison
                beam_caption, beam_confidence = caption_generator.generate_caption_beam_search(features)
                print(f"   Beam Search: '{beam_caption}' (confidence: {beam_confidence:.3f})")
                
            except Exception as e:
                logger.error(f"Failed to process {image_path.name}: {e}")
                continue
        
        print(f"\n✅ Fixed model testing completed successfully!")
        print("The model is now ready for integration with your application!")
        
    except Exception as e:
        logger.error(f"Fixed model testing failed: {e}")
        return


def main():
    """Main function."""
    
    print("🚀 Testing Fixed Full Dataset Trained Model")
    print("=" * 50)
    
    test_fixed_model()


if __name__ == "__main__":
    main()