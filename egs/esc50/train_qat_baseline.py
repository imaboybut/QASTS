#!/usr/bin/env python3
"""
5-fold QAT  
 fold  FP   25 epochs 
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

def train_qat_model(fold, seed=42):
    """ fold  QAT 25 """
    
    print(f"\n{'='*80}")
    print(f"Fold {fold} QAT Training Start (Seed: {seed})")
    print(f"{'='*80}")
    
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. FP  
    print(f"\n[1] Loading Fold {fold} FP model...")
    audio_model = models.ASTModel(
        label_dim=50,
        fstride=10,
        tstride=10,
        input_fdim=128,
        input_tdim=512,
        imagenet_pretrain=True,
        audioset_pretrain=True,
        model_size='base384'
    )
    
    # Fold FP  
    fp_ckpt_path = f"exp/fp_fold{fold}_baseline/models/final_audio_model.pth"
    if os.path.exists(fp_ckpt_path):
        state_dict = torch.load(fp_ckpt_path, map_location="cpu")
        # DataParallel 
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        audio_model.load_state_dict(new_state_dict)
        print(f" Fold {fold} FP model loaded from: {fp_ckpt_path}")
    else:
        print(f" FP model not found: {fp_ckpt_path}")
        print(f"   Please run FP training first!")
        return None
    
    # FP   
    audio_model = audio_model.to(device)
    audio_model.eval()
    
    # Validation  FP  
    val_audio_conf = {
        'num_mel_bins': 128,
        'target_length': 512,
        'freqm': 0,
        'timem': 0,
        'mixup': 0,
        'dataset': 'esc50',
        'mode': 'evaluation',
        'mean': -6.6268077,
        'std': 5.358466,
        'noise': False,
        'sample_rate': 16000  #  16kHz 
    }
    
    val_loader = torch.utils.data.DataLoader(
        dataloader.AudiosetDataset(
            f'./data/datafiles/esc_eval_data_{fold}.json',
            audio_conf=val_audio_conf,
            label_csv='./data/esc_class_labels_indices.csv'
        ),
        batch_size=48, shuffle=False, num_workers=8, pin_memory=True
    )
    
    print(f"\n[2] Checking FP model performance...")
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_data, batch_label in val_loader:
            batch_data = batch_data.to(device)
            batch_label = batch_label.to(device)
            batch_output = audio_model(batch_data)
            _, predicted = torch.max(batch_output, 1)
            _, target = torch.max(batch_label, 1)
            correct += (predicted == target).sum().item()
            total += batch_label.size(0)
    
    fp_accuracy = correct / total
    print(f"   FP Model Accuracy: {fp_accuracy:.4f} ({correct}/{total})")
    
    # 2. QAT 
    print(f"\n[3] Converting to QAT model...")
    with open('qat_config.yml', 'r') as f:
        qat_config = yaml.safe_load(f)
    qat_args = argparse.Namespace(**qat_config)
    
    #   (weight activation    )
    
    bit_w = qat_config.get('wq_bitw', 4)
    bit_a = qat_config.get('aq_bitw', 4)
    bit_num = min(bit_w, bit_a)
    
    audio_model = get_qat_model(audio_model, qat_args)
    print(f" QAT conversion completed (W{bit_w}A{bit_a})")
    
    # 3. DataParallel
    if not isinstance(audio_model, nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)
    audio_model = audio_model.to(device)
    
    # 4. Scale  
    print(f"\n[4] Initializing scale parameters...")
    audio_model.train()
    dummy_input = torch.randn(1, 512, 128).to(device)
    with torch.no_grad():
        _ = audio_model(dummy_input)
    
    scale_count = sum(1 for name, _ in audio_model.named_parameters() if '.s' in name)
    print(f" {scale_count} scale factors initialized")
    
    # 5.   (fold  )
    args = argparse.Namespace(
        data_train=f'./data/datafiles/esc_train_data_{fold}.json',
        data_val=f'./data/datafiles/esc_eval_data_{fold}.json',
        label_csv='./data/esc_class_labels_indices.csv',
        n_class=50,
        dataset='esc50',
        model='ast',
        fstride=10,
        tstride=10,
        n_epochs=25,
        batch_size=24,
        lr=5e-5,  # QAT learning rate
        optim='adam',
        num_workers=8,
        save_model=True,
        freqm=24,
        timem=96,
        mixup=0,
        bal='none',
        noise=False,
        dataset_mean=-6.6268077,
        dataset_std=5.358466,
        audio_length=512,
        loss='CE',
        metrics='acc',
        loss_fn=nn.CrossEntropyLoss(),
        warmup=False,
        lrscheduler_start=5,
        lrscheduler_step=1,
        lrscheduler_decay=0.85,
        n_print_steps=100,
        exp_dir=f'./exp/qat_fold{fold}_W{bit_w}A{bit_a}_baseline',  #  
        wa=False
    )
    
    #  
    os.makedirs(args.exp_dir, exist_ok=True)
    os.makedirs(os.path.join(args.exp_dir, 'models'), exist_ok=True)
    
    # 6.  
    print(f"\n[5] Preparing data loaders...")
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
        'sample_rate': 16000  #  16kHz 
    }
    
    val_audio_conf = audio_conf.copy()
    val_audio_conf.update({'freqm': 0, 'timem': 0, 'mixup': 0, 'mode': 'evaluation', 'noise': False})
    
    train_loader = torch.utils.data.DataLoader(
        dataloader.AudiosetDataset(args.data_train, audio_conf=audio_conf, label_csv=args.label_csv),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        dataloader.AudiosetDataset(args.data_val, audio_conf=val_audio_conf, label_csv=args.label_csv),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True
    )
    
    print(f"   Train: {len(train_loader.dataset)} samples")
    print(f"   Val: {len(val_loader.dataset)} samples")
    
    # 7. 
    print(f"\n[6] Starting 25-epoch QAT training for Fold {fold}...")
    print(f"   Learning rate: {args.lr}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Save path: {args.exp_dir}")
    
    train(audio_model, train_loader, val_loader, args)
    
    # 8.  
    print(f"\n[7] Final performance check...")
    audio_model.eval()
    stats, _ = validate(audio_model, val_loader, args, 25)
    
    if isinstance(stats, list) and len(stats) > 0:
        final_acc = stats[0].get('acc', 0.0)
    else:
        final_acc = 0.0
    
    print(f"\n{'='*80}")
    print(f"Fold {fold} QAT Training Completed")
    print(f"FP Accuracy: {fp_accuracy:.4f}  QAT Accuracy: {final_acc:.4f}")
    print(f"{'='*80}")
    
    return final_acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='5-fold QAT Training')
    parser.add_argument('--fold', type=int, default=0, help='Specific fold to train (0 for all)')
    args = parser.parse_args()
    
    if args.fold > 0:
        #  fold 
        print(f"Training only Fold {args.fold}")
        acc = train_qat_model(args.fold)
        print(f"\n Fold {args.fold} QAT completed: {acc:.2%}")
    else:
        # 5 fold  
        print("Training all 5 folds")
        results = {}
        for fold in range(1, 6):
            acc = train_qat_model(fold)
            results[f'fold{fold}'] = acc
            print(f"\n Fold {fold} QAT completed: {acc:.2%}")
        
        #  
        print(f"\n{'='*80}")
        print("5-Fold QAT Training Summary")
        print(f"{'='*80}")
        for fold, acc in results.items():
            print(f"{fold}: {acc:.2%}")
        avg_acc = np.mean(list(results.values()))
        print(f"Average: {avg_acc:.2%}")
        print(f"{'='*80}")