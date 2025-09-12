#!/usr/bin/env python3
"""
Better training script with meaningful synthetic data for the AI image caption generator.
"""

import os
import sys
import numpy as np
from pathlib import Path
import tensorflow as tf
import logging

# Add parent directory to path to import our modules
sys.path.append(str(Path(__file__).parent.parent))

from app.services.caption_generator import LSTMCaptionGenerator
from app.services.model_trainer import ModelTrainer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_meaningful_training_data(num_samples: int = 200):
    """
    Create meaningful synthetic training data with consistent feature-caption pairs.
    """
    
    # Define categories with associated features and captions
    categories = {
        'person': {
            'captions': [
                'a person standing in the street',
                'a man walking on the sidewalk',
                'a woman sitting on a bench',
                'a person wearing a red shirt',
                'a young man smiling at the camera',
                'a woman holding a bag',
                'a person riding a bicycle',
                'a man in a blue jacket',
                'a woman with long hair',
                'a person looking at the camera'
            ],
            'feature_pattern': np.array([1.0, 0.8, 0.2, 0.1, 0.9, 0.7])  # Person-like features
        },
        'animal': {
            'captions': [
                'a dog sitting in the grass',
                'a cat lying on the ground',
                'a bird flying in the sky',
                'a brown dog running in the park',
                'a small cat near a tree',
                'a white dog playing outside',
                'a black cat on the street',
                'a dog with a collar',
                'a cat looking at something',
                'a bird perched on a branch'
            ],
            'feature_pattern': np.array([0.2, 1.0, 0.8, 0.9, 0.3, 0.6])  # Animal-like features
        },
        'vehicle': {
            'captions': [
                'a car parked on the street',
                'a red car driving down the road',
                'a truck on the highway',
                'a blue car in the parking lot',
                'a motorcycle on the street',
                'a white car near a building',
                'a bus driving through the city',
                'a car with open doors',
                'a vehicle on the road',
                'a car in front of a house'
            ],
            'feature_pattern': np.array([0.1, 0.2, 1.0, 0.8, 0.4, 0.5])  # Vehicle-like features
        },
        'building': {
            'captions': [
                'a house with a red roof',
                'a tall building in the city',
                'a white house with windows',
                'a building with many floors',
                'a small house near trees',
                'a modern building downtown',
                'a house with a garden',
                'a building with glass windows',
                'a brick house on the street',
                'a large building with columns'
            ],
            'feature_pattern': np.array([0.3, 0.1, 0.4, 1.0, 0.8, 0.9])  # Building-like features
        },
        'nature': {
            'captions': [
                'a tree in the park',
                'green grass in the field',
                'flowers blooming in spring',
                'a large tree with leaves',
                'water flowing in the river',
                'mountains in the distance',
                'a beautiful sunset sky',
                'clouds in the blue sky',
                'a forest with tall trees',
                'a beach with sand and water'
            ],
            'feature_pattern': np.array([0.4, 0.3, 0.2, 0.6, 1.0, 0.8])  # Nature-like features
        }
    }
    
    features = []
    captions = []
    
    samples_per_category = num_samples // len(categories)
    
    for category, data in categories.items():
        for i in range(samples_per_category):
            # Create feature vector based on category pattern + some noise
            base_pattern = data['feature_pattern']
            # Repeat pattern to fill 2048 dimensions
            full_pattern = np.tile(base_pattern, 2048 // len(base_pattern) + 1)[:2048]
            # Add some random noise
            noise = np.random.normal(0, 0.1, 2048)
            feature_vector = full_pattern + noise
            
            # Select random caption from category
            caption = np.random.choice(data['captions'])
            
            features.append(feature_vector)
            captions.append(caption)
    
    # Add some random samples to fill remaining slots
    remaining = num_samples - len(features)
    for _ in range(remaining):
        # Random category
        category = np.random.choice(list(categories.keys()))
        data = categories[category]
        
        base_pattern = data['feature_pattern']
        full_pattern = np.tile(base_pattern, 2048 // len(base_pattern) + 1)[:2048]
        noise = np.random.normal(0, 0.1, 2048)
        feature_vector = full_pattern + noise
        
        caption = np.random.choice(data['captions'])
        
        features.append(feature_vector)
        captions.append(caption)
    
    return np.array(features), captions


def main():
    """Main training function."""
    
    print("🚀 AI Image Caption Generator - Better Training")
    print("=" * 50)
    
    try:
        # Create meaningful training data
        logger.info("Creating meaningful synthetic training data...")
        features, captions = create_meaningful_training_data(num_samples=200)
        
        logger.info(f"Created {len(features)} training samples")
        logger.info("Sample captions:")
        for i, caption in enumerate(captions[:5]):
            logger.info(f"  {i+1}. {caption}")
        
        # Initialize components
        caption_generator = LSTMCaptionGenerator(vocab_size=1000, max_length=12)
        trainer = ModelTrainer(caption_generator)
        
        # Train model
        logger.info("Starting model training...")
        history = trainer.train_model(
            image_features=features,
            captions=captions,
            validation_split=0.2,
            epochs=10,  # More epochs for better learning
            batch_size=16   # Reasonable batch size
        )
        
        # Create models directory
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        # Save model
        model_path = models_dir / "better_caption_model.keras"
        caption_generator.model.save(str(model_path))
        
        # Save vocabulary
        vocab_path = models_dir / "better_vocabulary.json"
        caption_generator.vocabulary.save_vocabulary(str(vocab_path))
        
        print(f"\n🎉 Training completed successfully!")
        print(f"📁 Model saved to: {model_path}")
        print(f"📁 Vocabulary saved to: {vocab_path}")
        
        # Test generation with different feature patterns
        logger.info("Testing caption generation with different feature types...")
        
        # Test with person-like features
        person_features = np.tile(np.array([1.0, 0.8, 0.2, 0.1, 0.9, 0.7]), 2048 // 6 + 1)[:2048]
        person_features = person_features.reshape(1, -1) + np.random.normal(0, 0.05, (1, 2048))
        
        # Test with animal-like features  
        animal_features = np.tile(np.array([0.2, 1.0, 0.8, 0.9, 0.3, 0.6]), 2048 // 6 + 1)[:2048]
        animal_features = animal_features.reshape(1, -1) + np.random.normal(0, 0.05, (1, 2048))
        
        # Test with vehicle-like features
        vehicle_features = np.tile(np.array([0.1, 0.2, 1.0, 0.8, 0.4, 0.5]), 2048 // 6 + 1)[:2048]
        vehicle_features = vehicle_features.reshape(1, -1) + np.random.normal(0, 0.05, (1, 2048))
        
        test_features = np.vstack([person_features, animal_features, vehicle_features])
        test_labels = ["Person-like features", "Animal-like features", "Vehicle-like features"]
        
        results = trainer.generate_sample_captions(test_features, num_samples=3, use_beam_search=False)
        
        print("\n📝 Generated Captions by Feature Type:")
        print("-" * 50)
        for i, ((caption, confidence), label) in enumerate(zip(results, test_labels)):
            print(f"{label}: '{caption}' (confidence: {confidence:.3f})")
        
        return 0
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        print(f"\n❌ Training failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())