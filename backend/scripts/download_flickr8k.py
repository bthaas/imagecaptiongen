#!/usr/bin/env python3
"""
Script to download and prepare Flickr8k dataset for training.
"""

import os
import sys
import zipfile
import subprocess
from pathlib import Path

def check_kaggle_setup():
    """Check if Kaggle API is properly set up."""
    try:
        import kaggle
        print("✅ Kaggle API is installed")
        return True
    except ImportError:
        print("❌ Kaggle API not installed. Run: pip install kaggle")
        return False
    except OSError as e:
        if "credentials" in str(e).lower():
            print("❌ Kaggle credentials not found.")
            print("   1. Go to https://www.kaggle.com/account")
            print("   2. Click 'Create New API Token'")
            print("   3. Place kaggle.json in ~/.kaggle/")
            print("   4. Run: chmod 600 ~/.kaggle/kaggle.json")
            return False
        raise

def download_dataset():
    """Download Flickr8k dataset using Kaggle API."""
    
    # Create data directory
    data_dir = Path("../data")
    data_dir.mkdir(exist_ok=True)
    
    print(f"📁 Created data directory: {data_dir.absolute()}")
    
    try:
        # Download dataset
        print("📥 Downloading Flickr8k dataset...")
        cmd = [
            "kaggle", "datasets", "download", 
            "-d", "adityajn105/flickr8k", 
            "-p", str(data_dir),
            "--unzip"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Dataset downloaded successfully!")
            return True
        else:
            print(f"❌ Download failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        return False

def verify_dataset():
    """Verify that the dataset was downloaded correctly."""
    
    data_dir = Path("../data")
    expected_files = [
        "Images",
        "captions.txt"
    ]
    
    print("🔍 Verifying dataset structure...")
    
    missing_files = []
    for file_path in expected_files:
        full_path = data_dir / file_path
        if not full_path.exists():
            missing_files.append(file_path)
        else:
            if file_path == "Images":
                image_count = len(list(full_path.glob("*.jpg")))
                print(f"   ✅ Found {image_count} images")
            else:
                print(f"   ✅ Found {file_path}")
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    
    print("✅ Dataset verification complete!")
    return True

def main():
    """Main function to download and verify Flickr8k dataset."""
    
    print("🚀 Flickr8k Dataset Downloader")
    print("=" * 40)
    
    # Check if Kaggle API is set up
    if not check_kaggle_setup():
        print("\n💡 Alternative: Download manually from:")
        print("   https://www.kaggle.com/datasets/adityajn105/flickr8k")
        print("   Then extract to: ../data/")
        return 1
    
    # Download dataset
    if not download_dataset():
        return 1
    
    # Verify dataset
    if not verify_dataset():
        return 1
    
    print("\n🎉 Flickr8k dataset is ready for training!")
    print(f"📁 Dataset location: {Path('../data').absolute()}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())