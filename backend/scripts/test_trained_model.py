#!/usr/bin/env python3
"""
Script to test the trained caption generation model.
"""

import os
import sys
import numpy as np
from pathlib import Path
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input

# Add parent directory to path to import our modules
sys.path.append(str(Path(__file__).parent.parent))

from app.services.caption_generator import LSTMCaptionGenerator
from app.services.image_processor import ImageProcessor

def load_trained_model(model_path: str, vocab_path: str) -> LSTMCaptionGenerator:
    """
    Load the trained model and vocabulary.
    
    Args:
        model_path: Path to the trained model
        vocab_path: Path to the vocabulary file
        
    Returns:
        Loaded caption generator
    """
    caption_generator = LSTMCaptionGenerator()
    
    # Load vocabulary
    caption_generator.vocabulary.load_vocabulary(vocab_path)
    
    # Load model
    caption_generator.model = tf.keras.models.load_model(model_path)
    caption_generator._model_loaded = True
    
    print(f"✅ Model loaded from {model_path}")
    print(f"✅ Vocabulary loaded from {vocab_path}")
    print(f"📊 Vocabulary size: {caption_generator.vocabulary.vocab_size}")
    
    return caption_generator

def extract_features_from_image(image_path: str) -> np.ndarray:
    """
    Extract CNN features from a single image.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        CNN features array
    """
    # Load CNN model
    cnn_model = InceptionV3(weights='imagenet', include_top=False, pooling='avg')
    
    # Load and preprocess image
    img = image.load_img(image_path, target_size=(299, 299))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    # Extract features
    features = cnn_model.predict(img_array, verbose=0)
    
    return features

def test_model_on_images(caption_generator: LSTMCaptionGenerator, test_images: List[str]):
    """
    Test the model on a list of images.
    
    Args:
        caption_generator: Trained caption generator
        test_images: List of image file paths
    """
    print(f"\n🧪 Testing model on {len(test_images)} images...")
    print("=" * 50)
    
    for i, img_path in enumerate(test_images):
        try:
            print(f"\n📸 Image {i+1}: {Path(img_path).name}")
            
            # Extract features
            features = extract_features_from_image(img_path)
            
            # Generate caption using beam search
            caption, confidence = caption_generator.generate_caption_beam_search(features)
            
            print(f"🤖 Generated: '{caption}'")
            print(f"📊 Confidence: {confidence:.3f}")
            
        except Exception as e:
            print(f"❌ Failed to process {img_path}: {e}")

def main():
    """Main testing function."""
    
    print("🧪 AI Image Caption Generator - Model Testing")
    print("=" * 50)
    
    # Paths
    models_dir = Path("../models")
    model_path = models_dir / "best_caption_model.h5"
    vocab_path = models_dir / "vocabulary.json"
    data_dir = Path("../data")
    images_dir = data_dir / "Images"
    
    # Check if trained model exists
    if not model_path.exists():
        print(f"❌ Trained model not found at {model_path}")
        print("   Please run train_model.py first")
        return 1
    
    if not vocab_path.exists():
        print(f"❌ Vocabulary file not found at {vocab_path}")
        print("   Please run train_model.py first")
        return 1
    
    try:
        # Load trained model
        caption_generator = load_trained_model(str(model_path), str(vocab_path))
        
        # Get some test images
        image_files = list(images_dir.glob("*.jpg"))[:10]  # Test on first 10 images
        test_images = [str(img) for img in image_files]
        
        # Test the model
        test_model_on_images(caption_generator, test_images)
        
        print(f"\n🎉 Testing completed!")
        
        return 0
        
    except Exception as e:
        print(f"❌ Testing failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())