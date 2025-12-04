#!/usr/bin/env python3
"""Data preprocessing script for DVC pipeline"""

import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
from src.utils.config import Config
import json

def create_data_splits():
    """Create train/val/test splits from raw data"""
    raw_dir = Config.RAW_DATA_DIR
    processed_dir = Config.PROCESSED_DATA_DIR
    
    # Create processed directory structure
    for split in ['train', 'val', 'test']:
        for class_name in ['Infected', 'Uninfected']:
            (processed_dir / split / class_name).mkdir(parents=True, exist_ok=True)
    
    # Process each class
    for class_name in ['Infected', 'Uninfected']:
        class_dir = raw_dir / class_name
        if not class_dir.exists():
            continue
            
        # Get all image files
        image_files = list(class_dir.glob('*.png')) + list(class_dir.glob('*.jpg'))
        
        # Split data: 80% train, 10% val, 10% test
        train_files, temp_files = train_test_split(image_files, test_size=0.2, random_state=42)
        val_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=42)
        
        # Copy files to respective directories
        splits = {'train': train_files, 'val': val_files, 'test': test_files}
        
        for split, files in splits.items():
            dest_dir = processed_dir / split / class_name
            for file_path in files:
                shutil.copy2(file_path, dest_dir / file_path.name)
    
    # Save split statistics
    stats = {}
    for split in ['train', 'val', 'test']:
        stats[split] = {}
        for class_name in ['Infected', 'Uninfected']:
            count = len(list((processed_dir / split / class_name).glob('*')))
            stats[split][class_name] = count
    
    with open(processed_dir / 'split_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    print("Data preprocessing completed!")
    print(f"Statistics: {stats}")

if __name__ == "__main__":
    create_data_splits()