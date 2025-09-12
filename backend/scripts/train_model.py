#!/usr/bin/env python3
"""
Script to train the AI image caption generator model on Flickr8k dataset.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input
import logging
from typing import Dict, List, Optional, Tuple
import ssl
import urllib.request

# Fix SSL certificate issues
ssl._create_default_https_context = ssl._create_unverified_context

# Add parent directory to path to import our modules
sys.path.append(str(Path(__file__).parent.parent))

from app.services.caption_generator import LSTMCaptionGenerator
from app.services.model_trainer import ModelTrainer
from app.services.image_processor import ImageProcessor

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DatasetLoader:
    """Load and preprocess Flickr8k dataset for training."""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.images_dir = self.data_dir / "Images"
        self.captions_file = self.data_dir / "captions.txt"
        
        # Initialize CNN model for feature extraction with SSL fix
        try:
            self.cnn_model = InceptionV3(weights='imagenet', include_top=False, pooling='avg')
            logger.info("CNN feature extractor loaded")
        except Exception as e:
            logger.warning(f"Failed to load InceptionV3 with imagenet weights: {e}")
            logger.info("Trying to load InceptionV3 without pre-trained weights...")
            self.cnn_model = InceptionV3(weights=None, include_top=False, pooling='avg')
            logger.info("CNN feature extractor loaded without pre-trained weights")
    
    def load_captions(self) -> Dict[str, List[str]]:
        """
        Load captions from the dataset file.
        
        Returns:
            Dict mapping image filenames to lists of captions
        """
        try:
            df = pd.read_csv(self.captions_file)
            
            # Group captions by image
            captions_dict = {}
            for _, row in df.iterrows():
                image_name = row['image']
                caption = row['caption']
                
                if image_name not in captions_dict:
                    captions_dict[image_name] = []
                captions_dict[image_name].append(caption)
            
            logger.info(f"Loaded captions for {len(captions_dict)} images")
            return captions_dict
            
        except Exception as e:
            logger.error(f"Failed to load captions: {e}")
            raise
    
    def extract_image_features(self, image_files: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Extract CNN features from images.
        
        Args:
            image_files: List of image filenames
            batch_size: Batch size for feature extraction
            
        Returns:
            Array of CNN features
        """
        features = []
        
        for i in range(0, len(image_files), batch_size):
            batch_files = image_files[i:i + batch_size]
            batch_images = []
            
            for img_file in batch_files:
                try:
                    img_path = self.images_dir / img_file
                    
                    # Load and preprocess image
                    img = image.load_img(img_path, target_size=(299, 299))
                    img_array = image.img_to_array(img)
                    img_array = np.expand_dims(img_array, axis=0)
                    img_array = preprocess_input(img_array)
                    
                    batch_images.append(img_array[0])
                    
                except Exception as e:
                    logger.warning(f"Failed to process {img_file}: {e}")
                    # Add zero features for failed images
                    batch_images.append(np.zeros((299, 299, 3)))
            
            if batch_images:
                batch_array = np.array(batch_images)
                batch_features = self.cnn_model.predict(batch_array, verbose=0)
                features.extend(batch_features)
            
            if (i // batch_size + 1) % 10 == 0:
                logger.info(f"Processed {i + len(batch_files)}/{len(image_files)} images")
        
        return np.array(features)
    
    def prepare_training_data(self, max_samples: Optional[int] = None) -> Tuple[np.ndarray, List[str], List[str]]:
        """
        Prepare training data by loading captions and extracting features.
        
        Args:
            max_samples: Maximum number of samples to use (for testing)
            
        Returns:
            Tuple of (features, captions, image_files)
        """
        logger.info("Loading dataset...")
        
        # Load captions
        captions_dict = self.load_captions()
        
        # Get list of available images
        available_images = [f for f in os.listdir(self.images_dir) if f.endswith('.jpg')]
        
        # Filter to images that have captions
        valid_images = [img for img in available_images if img in captions_dict]
        
        if max_samples:
            valid_images = valid_images[:max_samples]
        
        logger.info(f"Processing {len(valid_images)} images...")
        
        # Extract features
        features = self.extract_image_features(valid_images)
        
        # Prepare caption list (use first caption for each image)
        captions = [captions_dict[img][0] for img in valid_images]
        
        logger.info(f"Dataset prepared: {len(features)} samples")
        return features, captions, valid_images


def main():
    """Main training function."""
    
    print("🚀 AI Image Caption Generator - Model Training")
    print("=" * 50)
    
    # Configuration
    MAX_SAMPLES = 1000  # Limit for initial training (set to None for full dataset)
    EPOCHS = 20
    BATCH_SIZE = 32
    VALIDATION_SPLIT = 0.2
    
    try:
        # Initialize components
        logger.info("Initializing training components...")
        
        dataset_loader = DatasetLoader()
        caption_generator = LSTMCaptionGenerator()
        trainer = ModelTrainer(caption_generator)
        
        # Load and prepare data
        logger.info("Loading and preparing dataset...")
        features, captions, image_files = dataset_loader.prepare_training_data(max_samples=MAX_SAMPLES)
        
        logger.info(f"Training configuration:")
        logger.info(f"  - Samples: {len(features)}")
        logger.info(f"  - Epochs: {EPOCHS}")
        logger.info(f"  - Batch size: {BATCH_SIZE}")
        logger.info(f"  - Validation split: {VALIDATION_SPLIT}")
        
        # Create models directory
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        model_save_path = models_dir / "best_caption_model.h5"
        
        # Train model
        logger.info("Starting model training...")
        history = trainer.train_model(
            image_features=features,
            captions=captions,
            validation_split=VALIDATION_SPLIT,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            save_path=str(model_save_path)
        )
        
        # Save training history
        history_path = models_dir / "training_history.json"
        trainer.save_training_history(str(history_path))
        
        # Generate sample captions for evaluation
        logger.info("Generating sample captions...")
        sample_features = features[:5]  # Use first 5 images
        sample_results = trainer.generate_sample_captions(sample_features, num_samples=5)
        
        print("\n📝 Sample Generated Captions:")
        print("-" * 40)
        for i, (caption, confidence) in enumerate(sample_results):
            print(f"{i+1}. '{caption}' (confidence: {confidence:.3f})")
        
        # Save vocabulary
        vocab_path = models_dir / "vocabulary.json"
        caption_generator.vocabulary.save_vocabulary(str(vocab_path))
        
        print(f"\n🎉 Training completed successfully!")
        print(f"📁 Model saved to: {model_save_path}")
        print(f"📁 Vocabulary saved to: {vocab_path}")
        print(f"📁 Training history saved to: {history_path}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        print(f"\n❌ Training failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())