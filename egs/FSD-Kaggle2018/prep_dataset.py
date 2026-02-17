#!/usr/bin/env python3
"""
FSD-Kaggle2018   
ESC-50 prep_dataset.py  FSD-Kaggle2018 
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path

def create_json_files():
    """
    FSD-Kaggle2018  JSON  
    """
    
    # Resolve from this script location to avoid user-specific absolute paths.
    base_path = Path(__file__).resolve().parent
    data_path = base_path / "data"
    audio_path = data_path / "FSDKaggle2018.audio_train"
    test_audio_path = data_path / "FSDKaggle2018.audio_test"
    meta_path = data_path / "FSDKaggle2018.meta"
    output_path = data_path / "datafiles"
    
    #   
    output_path.mkdir(exist_ok=True)
    
    #  
    train_meta = pd.read_csv(meta_path / "train_post_competition.csv")
    test_meta = pd.read_csv(meta_path / "test_post_competition_scoring_clips.csv")
    
    #    
    labels = sorted(train_meta['label'].unique())
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    
    #   CSV   
    print(f"Using existing class labels CSV with {len(labels)} classes")
    
    # Train/Val  (90/10)
    np.random.seed(42)
    train_meta['random'] = np.random.rand(len(train_meta))
    train_data = train_meta[train_meta['random'] < 0.9]
    val_data = train_meta[train_meta['random'] >= 0.9]
    
    print(f"Train: {len(train_data)} samples, Val: {len(val_data)} samples, Test: {len(test_meta)} samples")
    
    # Train JSON 
    train_json = []
    for _, row in train_data.iterrows():
        wav_path = str(audio_path / row['fname'])
        if os.path.exists(wav_path):
            train_json.append({
                "wav": wav_path,
                "labels": f"/m/fsd{label_to_idx[row['label']]:02d}"
            })
    
    with open(output_path / "fsd_train_data.json", 'w') as f:
        json.dump({"data": train_json}, f, indent=1)
    print(f"Created train JSON with {len(train_json)} samples")
    
    # Validation JSON 
    val_json = []
    for _, row in val_data.iterrows():
        wav_path = str(audio_path / row['fname'])
        if os.path.exists(wav_path):
            val_json.append({
                "wav": wav_path,
                "labels": f"/m/fsd{label_to_idx[row['label']]:02d}"
            })
    
    with open(output_path / "fsd_val_data.json", 'w') as f:
        json.dump({"data": val_json}, f, indent=1)
    print(f"Created val JSON with {len(val_json)} samples")
    
    # Test JSON 
    test_json = []
    for _, row in test_meta.iterrows():
        wav_path = str(test_audio_path / row['fname'])
        if os.path.exists(wav_path):
            test_json.append({
                "wav": wav_path,
                "labels": f"/m/fsd{label_to_idx[row['label']]:02d}"
            })
    
    with open(output_path / "fsd_test_data.json", 'w') as f:
        json.dump({"data": test_json}, f, indent=1)
    print(f"Created test JSON with {len(test_json)} samples")
    
    #   (mean, std)
    print("\nCalculating dataset statistics...")
    #        
    #  ESC-50   
    dataset_mean = -1.6813264
    dataset_std = 2.4526765
    
    print(f"Dataset mean: {dataset_mean:.7f}")
    print(f"Dataset std: {dataset_std:.7f}")
    
    #   
    print("\n=== Dataset Summary ===")
    print(f"Total classes: {len(labels)}")
    print(f"Train samples: {len(train_json)}")
    print(f"Val samples: {len(val_json)}")
    print(f"Test samples: {len(test_json)}")
    print(f"Audio sample rate: 44100 Hz")
    print(f"Audio length: 1024 frames (10.24 seconds)")
    
if __name__ == "__main__":
    create_json_files()
