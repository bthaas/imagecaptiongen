#!/usr/bin/env python3
"""
Simple script to verify Flickr8k dataset structure.
"""

import os
from pathlib import Path

def verify_dataset():
    """Verify that the dataset was downloaded correctly."""
    
    data_dir = Path("data")
    
    print("🔍 Verifying Flickr8k dataset structure...")
    print(f"📁 Looking in: {data_dir.absolute()}")
    
    if not data_dir.exists():
        print("❌ Data directory not found!")
        print("   Please create: backend/data/")
        return False
    
    # Check for Images directory
    images_dir = data_dir / "Images"
    if images_dir.exists():
        image_files = list(images_dir.glob("*.jpg"))
        print(f"✅ Found Images directory with {len(image_files)} images")
    else:
        print("❌ Images directory not found")
        return False
    
    # Check for captions file
    captions_file = data_dir / "captions.txt"
    if captions_file.exists():
        with open(captions_file, 'r') as f:
            lines = len(f.readlines())
        print(f"✅ Found captions.txt with {lines} lines")
    else:
        print("❌ captions.txt not found")
        return False
    
    print("🎉 Dataset verification complete!")
    print("📊 Dataset Summary:")
    print(f"   - Images: {len(image_files)}")
    print(f"   - Captions: {lines}")
    print("✅ Ready for training!")
    
    return True

if __name__ == "__main__":
    verify_dataset()