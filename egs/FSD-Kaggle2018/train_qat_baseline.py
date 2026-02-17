#!/usr/bin/env python3
"""
FSD-Kaggle2018 QAT  
run_fp.sh   QAT 
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
import numpy as np
import argparse
import random

sys.path.append('../../src')
import dataloader
import models
from models import get_qat_model
from traintest import train, validate

def set_seed(seed):
    """   """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_qat_fsd(seed=42):
    """FSD-Kaggle2018 QAT  - run_fp.sh  """
    
    print(f"\n{'='*80}")
    print(f"FSD-Kaggle2018 QAT Training Start (Seed: {seed})")
    print(f"{'='*80}")
    
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. FP  
    print(f"\n[1] Loading FSD-Kaggle2018 FP model...")
    audio_model = models.ASTModel(
        label_dim=41,  # FSD-Kaggle2018 41 
        fstride=10,
        tstride=10,
        input_fdim=128,
        input_tdim=1024,  # FSD-Kaggle2018 10.24  (1024 frames)
        imagenet_pretrain=True,
        audioset_pretrain=True,
        model_size='base384'
    )
    
    # FP   
    fp_ckpt_path = "./exp/test-fsdkaggle2018-f10-t10-impTrue-aspTrue-b24-lr1e-5/models/final_audio_model.pth"
    
    if not os.path.exists(fp_ckpt_path):
        print(f" FP model not found at: {fp_ckpt_path}")
        return
    
    state_dict = torch.load(fp_ckpt_path, map_location=device)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    audio_model.load_state_dict(new_state_dict)
    print(f" FP model loaded from: {fp_ckpt_path}")
    
    # 2. QAT wrapper  (train_qat_baseline.py )
    print("\n[2] Converting to QAT model...")
    
    # QAT config  
    qat_config_path = './qat_config.yml'
    if not os.path.exists(qat_config_path):
        # ESC-50 config 
        esc50_config_path = '../esc50/qat_config.yml'
        if os.path.exists(esc50_config_path):
            import shutil
            shutil.copy(esc50_config_path, qat_config_path)
            print(f"   Copied QAT config from ESC-50")
        else:
            print(" QAT config not found!")
            return
    
    with open(qat_config_path, 'r') as f:
        qat_config = yaml.safe_load(f)
    
    #   (weight activation)
    bit_w = qat_config.get('wq_bitw', 8)
    bit_a = qat_config.get('aq_bitw', 8)
    
    qat_args = argparse.Namespace(**qat_config)
    qat_model = get_qat_model(audio_model, qat_args)
    print(f" QAT model created (W{bit_w}A{bit_a})")
    
    # FP    (QAT  )
    del audio_model
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(" FP model memory released")
    
    # 3. DataParallel
    if not isinstance(qat_model, nn.DataParallel):
        qat_model = nn.DataParallel(qat_model)
    qat_model = qat_model.to(device)
    
    # 4. Scale  
    print("\n[3] Initializing scale parameters...")
    qat_model.train()
    dummy_input = torch.randn(1, 128, 1024).to(device)  # FSD: (1, 128, 1024)
    with torch.no_grad():
        _ = qat_model(dummy_input)
    
    scale_count = sum(1 for name, _ in qat_model.named_parameters() if '.s' in name)
    print(f" {scale_count} scale factors initialized")
    
    # 5.   - run_sc.sh !
    args = argparse.Namespace(
        #  
        data_train='./data/datafiles/fsd_train_data.json',  # run_fp.sh 
        data_val='./data/datafiles/fsd_val_data.json',
        data_eval='./data/datafiles/fsd_test_data.json',
        label_csv='./data/fsd_class_labels_indices.csv',
        
        #  
        n_class=41,
        dataset='fsdkaggle2018',
        model='ast',
        fstride=10,
        tstride=10,
        
        #   - run_sc.sh  !
        n_epochs=25,  # ESC-50 style: epoch=25
        batch_size=24,  # ESC-50 style: batch_size=48
        lr=1e-5,  # ESC-50 style: lr=1e-5
        optim='adam',
        num_workers=8,
        save_model=True,
        
        # Augmentation - run_sc.sh 
        freqm=24,  # ESC-50 style: freqm=24
        timem=96,  # ESC-50 style: timem=96
        mixup=0,  # ESC-50 style: mixup=0
        bal='none',
        noise=True,  # run_fp.sh: noise=True
        dataset_mean=-1.6813264,
        dataset_std=2.4526765,
        #   - run_fp.sh 
        audio_length=1024,
        
        # Loss  - run_sc.sh 
        loss='CE',  # run_fp.sh: loss=CE
        metrics='acc',
        loss_fn=nn.CrossEntropyLoss(),  # CrossEntropy loss
        
        #  - run_sc.sh 
        warmup=True,  # run_fp.sh: warmup=True
        lrscheduler_start=10,  # run_fp.sh: lrscheduler_start=10
        lrscheduler_step=5,  # run_fp.sh: lrscheduler_step=5
        lrscheduler_decay=0.5,  # run_fp.sh: lrscheduler_decay=0.5
        
        # 
        n_print_steps=100,
        exp_dir=f'./exp/qat_fsd_W{bit_w}A{bit_a}_baseline',  #  
        wa=True,  # Weight averaging on
        wa_start=6,
        wa_end=30
    )
    
    #  
    os.makedirs(args.exp_dir, exist_ok=True)
    os.makedirs(os.path.join(args.exp_dir, 'models'), exist_ok=True)
    
    # 6.   
    print("\n[4] Creating data loaders...")
    audio_conf = {
        'num_mel_bins': 128,
        'target_length': args.audio_length,
        'freqm': args.freqm,
        'timem': args.timem,
        'mixup': args.mixup,
        'dataset': args.dataset,
        'mode': 'train',
        'mean': args.dataset_mean,
        'std': args.dataset_std,
        'noise': args.noise,
        'sample_rate': 16000  # 16kHz  
    }
    
    val_audio_conf = audio_conf.copy()
    val_audio_conf.update({'freqm': 0, 'timem': 0, 'mixup': 0, 'mode': 'evaluation', 'noise': False, 'sample_rate': 16000})
    
    train_loader = torch.utils.data.DataLoader(
        dataloader.AudiosetDataset(args.data_train, audio_conf=audio_conf, label_csv=args.label_csv),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        dataloader.AudiosetDataset(args.data_val, audio_conf=val_audio_conf, label_csv=args.label_csv),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True
    )
    
    eval_loader = torch.utils.data.DataLoader(
        dataloader.AudiosetDataset(args.data_eval, audio_conf=val_audio_conf, label_csv=args.label_csv),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True
    )
    
    # 7. 
    print(f"\n[5] Starting QAT training for {args.n_epochs} epochs...")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Learning rate: {args.lr}")
    print(f"   Loss: {args.loss}")
    print(f"   Mixup: {args.mixup}")
    print(f"   SpecAugment: freq={args.freqm}, time={args.timem}")
    print(f"   Noise augmentation: {args.noise}")
    print(f"   Exp dir: {args.exp_dir}")
    
    #   
    with open(os.path.join(args.exp_dir, 'config.txt'), 'w') as f:
        f.write("QAT Training Configuration\n")
        f.write("="*50 + "\n")
        for key, value in vars(args).items():
            if key != 'loss_fn':  # loss_fn  
                f.write(f"{key}: {value}\n")
    
    train(qat_model, train_loader, val_loader, args)
    
    # 8.  
    print("\n[6] Final evaluation...")
    
    # Best model 
    best_model_path = os.path.join(args.exp_dir, 'models', 'final_audio_model.pth')
    if os.path.exists(best_model_path):
        sd = torch.load(best_model_path, map_location=device)
        qat_model.load_state_dict(sd)
        print(f" Loaded best model from: {best_model_path}")
    
    qat_model.eval()
    
    # Validation set 
    stats, _ = validate(qat_model, val_loader, args, args.n_epochs)
    val_acc = stats[0]['acc'] if isinstance(stats, list) and len(stats) > 0 else 0.0
    
    # Test set 
    stats, _ = validate(qat_model, eval_loader, args, args.n_epochs)
    test_acc = stats[0]['acc'] if isinstance(stats, list) and len(stats) > 0 else 0.0
    
    print(f"\n{'='*80}")
    print(f"QAT Training Complete!")
    print(f"{'='*80}")
    print(f"Validation Accuracy: {val_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    
    #  
    with open(os.path.join(args.exp_dir, 'results.txt'), 'w') as f:
        f.write(f"FSD-Kaggle2018 QAT Results\n")
        f.write(f"{'='*80}\n")
        f.write(f"Validation Accuracy: {val_acc:.4f}\n")
        f.write(f"Test Accuracy: {test_acc:.4f}\n")
        f.write(f"Training epochs: {args.n_epochs}\n")
        f.write(f"Learning rate: {args.lr}\n")
        f.write(f"Batch size: {args.batch_size}\n")
        f.write(f"Loss: {args.loss}\n")
        f.write(f"Mixup: {args.mixup}\n")
        f.write(f"SpecAugment: freq={args.freqm}, time={args.timem}\n")
        f.write(f"Noise: {args.noise}\n")
    
    print(f" Results saved to: {os.path.join(args.exp_dir, 'results.txt')}")
    
    return val_acc, test_acc

if __name__ == "__main__":
    val_acc, test_acc = train_qat_fsd()
    print(f"\n FSD-Kaggle2018 QAT training completed!")
    print(f"   Final Test Accuracy: {test_acc:.4f}")
