#!/usr/bin/env python3
"""
Demo script for LSTM caption generation functionality.
Demonstrates the complete pipeline from CNN features to generated captions.
"""

import numpy as np
import logging
from app.services.caption_generator import LSTMCaptionGenerator
from app.services.model_trainer import create_sample_training_data, ModelTrainer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def demo_caption_generation():
    """Demonstrate caption generation functionality."""
    print("=== LSTM Caption Generator Demo ===\n")
    
    # 1. Create caption generator
    print("1. Creating LSTM Caption Generator...")
    generator = LSTMCaptionGenerator(vocab_size=100, max_length=10)
    print(f"   - Vocabulary size: {generator.vocab_size}")
    print(f"   - Max caption length: {generator.max_length}")
    print(f"   - Embedding dimension: {generator.embedding_dim}")
    print(f"   - LSTM units: {generator.lstm_units}")
    
    # 2. Initialize for inference
    print("\n2. Initializing for inference...")
    generator.initialize_for_inference()
    print("   - Default vocabulary loaded")
    print("   - Model architecture built")
    print(f"   - Model loaded: {generator._model_loaded}")
    
    # 3. Generate sample CNN features
    print("\n3. Generating sample CNN features...")
    sample_features = np.random.randn(3, 2048)  # 3 sample images
    print(f"   - Generated features shape: {sample_features.shape}")
    
    # 4. Generate captions using greedy decoding
    print("\n4. Generating captions (Greedy Decoding)...")
    for i in range(3):
        features = sample_features[i:i+1]  # Keep batch dimension
        caption, confidence = generator.generate_caption_greedy(features, temperature=1.0)
        print(f"   Sample {i+1}: '{caption}' (confidence: {confidence:.3f})")
    
    # 5. Generate captions using beam search
    print("\n5. Generating captions (Beam Search)...")
    for i in range(3):
        features = sample_features[i:i+1]  # Keep batch dimension
        caption, confidence = generator.generate_caption_beam_search(features, beam_width=3)
        print(f"   Sample {i+1}: '{caption}' (confidence: {confidence:.3f})")
    
    # 6. Show model information
    print("\n6. Model Information:")
    model_info = generator.get_model_info()
    for key, value in model_info.items():
        print(f"   - {key}: {value}")
    
    print("\n=== Demo Complete ===")


def demo_vocabulary_management():
    """Demonstrate vocabulary management functionality."""
    print("\n=== Vocabulary Management Demo ===\n")
    
    # 1. Create vocabulary manager
    print("1. Creating Vocabulary Manager...")
    from app.services.caption_generator import VocabularyManager
    vocab_manager = VocabularyManager(vocab_size=50, max_length=8)
    
    # 2. Build vocabulary from sample captions
    print("\n2. Building vocabulary from sample captions...")
    sample_captions = [
        "a man sitting on a bench",
        "a woman walking in the park",
        "a dog running on the beach",
        "a cat sleeping on the couch",
        "a car driving on the road"
    ]
    
    vocab_manager.build_vocabulary(sample_captions)
    print(f"   - Built vocabulary with {len(vocab_manager.word_to_idx)} words")
    print(f"   - Sample words: {list(vocab_manager.word_to_idx.keys())[:10]}")
    
    # 3. Convert text to sequences
    print("\n3. Converting text to sequences...")
    test_texts = ["a man walking", "a dog sitting"]
    sequences = vocab_manager.text_to_sequences(test_texts)
    print(f"   - Input texts: {test_texts}")
    print(f"   - Sequences: {sequences}")
    
    # 4. Convert sequences back to text
    print("\n4. Converting sequences back to text...")
    reconstructed_texts = vocab_manager.sequences_to_text(sequences)
    print(f"   - Reconstructed texts: {reconstructed_texts}")
    
    print("\n=== Vocabulary Demo Complete ===")


def demo_training_utilities():
    """Demonstrate model training utilities."""
    print("\n=== Training Utilities Demo ===\n")
    
    # 1. Create sample training data
    print("1. Creating sample training data...")
    features, captions = create_sample_training_data(num_samples=20)
    print(f"   - Features shape: {features.shape}")
    print(f"   - Number of captions: {len(captions)}")
    print(f"   - Sample captions: {captions[:3]}")
    
    # 2. Create trainer
    print("\n2. Creating model trainer...")
    generator = LSTMCaptionGenerator(vocab_size=50, max_length=8)
    trainer = ModelTrainer(generator)
    
    # 3. Prepare training data
    print("\n3. Preparing training data...")
    vocabulary = trainer.prepare_training_data(features, captions)
    print(f"   - Vocabulary size: {len(vocabulary.word_to_idx)}")
    print(f"   - Max length: {vocabulary.max_length}")
    
    # 4. Create data generator
    print("\n4. Creating data generator...")
    from app.services.model_trainer import DataGenerator
    data_gen = DataGenerator(features, captions, vocabulary, batch_size=4)
    print(f"   - Number of batches: {len(data_gen)}")
    print(f"   - Number of sequences: {len(data_gen.sequences)}")
    
    if len(data_gen) > 0:
        batch_inputs, batch_targets = data_gen[0]
        print(f"   - Batch input shapes: {[inp.shape for inp in batch_inputs]}")
        print(f"   - Batch target shape: {batch_targets.shape}")
    
    print("\n=== Training Utilities Demo Complete ===")


if __name__ == "__main__":
    try:
        demo_caption_generation()
        demo_vocabulary_management()
        demo_training_utilities()
        
    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        raise