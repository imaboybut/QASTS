#!/usr/bin/env python3
"""
FSD-Kaggle2018 QAT Model Soup  (ESC-50  )
- ESC-50 train_qasts.py   FSD 
- FSD-Kaggle2018  
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

RATE_SET_HZ = [12000, 16000, 20000, 24000, 28000]
CANDIDATES_PER_RATE = 2
TOTAL_CANDIDATES = len(RATE_SET_HZ) * CANDIDATES_PER_RATE


def get_candidate_sample_rate(candidate_idx, rate_mode):
    """QASTS: 5 rate x 2 , Soup baseline: 16kHz ."""
    if candidate_idx < 0 or candidate_idx >= TOTAL_CANDIDATES:
        raise ValueError(
            f"candidate_idx must be in [0, {TOTAL_CANDIDATES - 1}], got {candidate_idx}"
        )
    if rate_mode == 'qasts':
        return RATE_SET_HZ[candidate_idx // CANDIDATES_PER_RATE]
    return 16000


def set_seed(seed):
    """   """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_qat_wrapper():
    """
    QAT wrapper  (  )
    FP  QAT  
    """
    print(f"\n[1] Creating QAT wrapper for FSD-Kaggle2018...")
    
    # 1.  FP   
    audio_model = models.ASTModel(
        label_dim=41,  # FSD-Kaggle2018: 41 classes
        fstride=10,
        tstride=10,
        input_fdim=128,
        input_tdim=1024,  # FSD-Kaggle2018: 1024
        imagenet_pretrain=True,
        audioset_pretrain=True,  # FSD: True (ESC-50 style)
        model_size='base384'
    )
    
    # 2. FP    (QAT wrapper  )
    fp_ckpt_path = "./exp/test-fsdkaggle2018-f10-t10-impTrue-aspTrue-b24-lr1e-5/models/final_audio_model.pth"
    if os.path.exists(fp_ckpt_path):
        state_dict = torch.load(fp_ckpt_path, map_location="cpu")
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        audio_model.load_state_dict(new_state_dict)
        print(f"   FP model loaded for QAT wrapper creation")
    
    # 3. QAT wrapper  (  )
    print("   Converting to QAT wrapper...")
    with open('qat_config.yml', 'r') as f:
        qat_config = yaml.safe_load(f)
    qat_args = argparse.Namespace(**qat_config)
    qat_model = get_qat_model(audio_model, qat_args)
    
    print(f" QAT wrapper created with quantization modules")
    return qat_model

def load_qat_weights(qat_model):
    """
    QAT   wrapper 
    """
    print(f"\n[2] Loading QAT weights...")
    
    #  scale factors   dummy forward pass
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    qat_model = qat_model.to(device)
    qat_model.train()
    dummy_input = torch.randn(1, 128, 1024).to(device)  # FSD: 1024
    with torch.no_grad():
        _ = qat_model(dummy_input)
    print("   Scale factors initialized with dummy forward pass")
    
    # qat_config.yml   
    with open('qat_config.yml', 'r') as f:
        config = yaml.safe_load(f)
    bit_w = config.get('wq_bitw', 8)
    bit_a = config.get('aq_bitw', 8)
    
    # QAT   (scale factors )
    qat_ckpt_path = f"./exp/qat_fsd_W{bit_w}A{bit_a}_baseline/models/final_audio_model.pth"
    if not os.path.exists(qat_ckpt_path):
        print(f" QAT model not found: {qat_ckpt_path}")
        return None
    
    state_dict = torch.load(qat_ckpt_path, map_location="cpu")
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    
    #    (scale factors , strict=True)
    try:
        qat_model.load_state_dict(new_state_dict, strict=True)
        print(f" QAT weights loaded (strict mode) from: {qat_ckpt_path}")
    except RuntimeError as e:
        # strict     
        print(f"   Warning: Some keys mismatched, trying non-strict load...")
        qat_model.load_state_dict(new_state_dict, strict=False)
        print(f" QAT weights loaded (non-strict mode) from: {qat_ckpt_path}")
    
    # Scale factors  
    scale_count = sum(1 for name, _ in qat_model.named_parameters() if '.s' in name)
    print(f"   Total {scale_count} scale factors in model")
    
    return qat_model

def verify_qat_performance(model, device):
    """
     QAT    (Validation + Test)
    """
    print(f"\n[3] Verifying QAT model performance...")
    
    model = model.to(device)
    model.eval()
    
    audio_conf = {
        'num_mel_bins': 128,
        'target_length': 1024,  # FSD: 1024
        'freqm': 0,
        'timem': 0,
        'mixup': 0,
        'dataset': 'fsdkaggle2018',
        'mode': 'evaluation',
        'mean': -1.6813264,  # FSD mean
        'std': 2.4526765,   # FSD std
        'noise': False,
        'sample_rate': 16000  # 16kHz 
    }
    
    # 1. Validation Set  (9,981)
    print("   Evaluating on Validation Set...")
    val_loader = torch.utils.data.DataLoader(
        dataloader.AudiosetDataset(
            './data/datafiles/fsd_val_data.json',
            audio_conf=audio_conf,
            label_csv='./data/fsd_class_labels_indices.csv'
        ),
        batch_size=48, shuffle=False, num_workers=8, pin_memory=True
    )
    
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for batch_data, batch_label in val_loader:
            batch_data = batch_data.to(device)
            batch_label = batch_label.to(device)
            batch_output = model(batch_data)
            batch_output = batch_output  # No sigmoid for CE loss  # FSD-Kaggle2018: BCE loss
            _, predicted = torch.max(batch_output, 1)
            _, target = torch.max(batch_label, 1)
            val_correct += (predicted == target).sum().item()
            val_total += batch_label.size(0)
    
    val_accuracy = val_correct / val_total
    print(f"   QAT Model Val Accuracy: {val_accuracy:.4f} ({val_correct}/{val_total})")
    
    # 2. Test Set  (11,005)
    print("   Evaluating on Test Set...")
    test_loader = torch.utils.data.DataLoader(
        dataloader.AudiosetDataset(
            './data/datafiles/fsd_test_data.json',  # Test set
            audio_conf=audio_conf,
            label_csv='./data/fsd_class_labels_indices.csv'
        ),
        batch_size=48, shuffle=False, num_workers=8, pin_memory=True
    )
    
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for batch_data, batch_label in test_loader:
            batch_data = batch_data.to(device)
            batch_label = batch_label.to(device)
            batch_output = model(batch_data)
            batch_output = batch_output  # No sigmoid for CE loss
            _, predicted = torch.max(batch_output, 1)
            _, target = torch.max(batch_label, 1)
            test_correct += (predicted == target).sum().item()
            test_total += batch_label.size(0)
    
    test_accuracy = test_correct / test_total
    print(f"   QAT Model Test Accuracy: {test_accuracy:.4f} ({test_correct}/{test_total})")
    
    return val_accuracy, test_accuracy

def train_candidate_with_diversity(model, candidate_idx, num_epochs=3, rate_mode='qasts'):
    """
    Diversity  candidate  
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # qat_config.yml   
    with open('qat_config.yml', 'r') as f:
        config = yaml.safe_load(f)
    bit_w = config.get('wq_bitw', 8)
    bit_a = config.get('aq_bitw', 8)
    
    # Candidate   ( )
    candidate_seed = 42 + candidate_idx  #  candidate  
    np.random.seed(candidate_seed)
    
    # Diversity 
    lr = np.random.uniform(1e-6, 1e-5)
    sample_rate = get_candidate_sample_rate(candidate_idx, rate_mode)
    
    print(f"\n[4-{candidate_idx}] Training candidate {candidate_idx}")
    print(f"   Seed: {candidate_seed}")
    print(f"   Learning rate: {lr:.2e}")
    print(f"   Sample rate: {sample_rate} Hz")
    print(f"   Mode: {rate_mode}")
    
    # QASTS scale , Soup baseline scale  
    scale_params_frozen = 0
    if rate_mode == 'qasts':
        for name, param in model.named_parameters():
            if name.endswith('.s'):
                param.requires_grad = False
                scale_params_frozen += 1
    
    print(f"   Scale factors frozen: {scale_params_frozen} parameters")
    
    # :    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Trainable parameters: {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.2f}%)")
    
    # DataParallel
    if not isinstance(model, nn.DataParallel):
        model = nn.DataParallel(model)
    model = model.to(device)
    
    #  
    exp_tag = 'fixscale' if rate_mode == 'qasts' else 'norate'
    args = argparse.Namespace(
        data_train='./data/datafiles/fsd_train_data.json',
        data_val='./data/datafiles/fsd_val_data.json',
        label_csv='./data/fsd_class_labels_indices.csv',
        n_class=41,  # FSD-Kaggle2018: 41 classes
        dataset='fsdkaggle2018',
        model='ast',
        fstride=10,
        tstride=10,
        n_epochs=num_epochs,
        batch_size=24,  # FSD-Kaggle2018 batch size (same as run_fp.sh)
        lr=lr,  # Diversity:  LR
        optim='adam',
        num_workers=8,
        save_model=True,
        freqm=24,  # ESC-50 style freqm
        timem=96,  # ESC-50 style timem
        mixup=0,  # ESC-50 style: no mixup
        bal='none',
        noise=False,  # ESC-50 style: noise=False for soup training
        dataset_mean=-1.6813264,
        dataset_std=2.4526765,
        audio_length=1024,  # FSD-Kaggle2018: 1024
        loss='CE',  # FSD-Kaggle2018: CE
        metrics='acc',
        loss_fn=nn.CrossEntropyLoss(),
        warmup=False,
        lrscheduler_start=3,
        lrscheduler_step=1,
        lrscheduler_decay=0.85,
        n_print_steps=100,
        exp_dir=f'./exp/soup_candidate{candidate_idx}_W{bit_w}A{bit_a}_{exp_tag}',
        wa=False,
        target_sample_rate=sample_rate
    )
    
    #  
    os.makedirs(args.exp_dir, exist_ok=True)
    os.makedirs(os.path.join(args.exp_dir, 'models'), exist_ok=True)
    
    # Hyperparameter   
    log_file = os.path.join(args.exp_dir, 'training_log.txt')
    with open(log_file, 'w') as f:
        f.write(f"Candidate {candidate_idx} Training Log\n")
        f.write("="*50 + "\n")
        f.write(f"Dataset: FSD-Kaggle2018\n")
        f.write(f"Seed: {candidate_seed}\n")
        f.write(f"Learning Rate: {lr:.2e}\n")
        f.write(f"Sample Rate: {sample_rate} Hz\n")
        f.write(f"Rate Mode: {rate_mode}\n")
        f.write(f"Epochs: {num_epochs}\n")
        f.write(f"Batch Size: {args.batch_size}\n")
        f.write(f"Scale factors frozen: {scale_params_frozen}\n")
        f.write("="*50 + "\n")
    print(f"   Log saved to: {log_file}")
    
    #   (sampling rate diversity )
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
        'sample_rate': sample_rate  # dataloader.py   
    }
    
    val_audio_conf = audio_conf.copy()
    val_audio_conf.update({'freqm': 0, 'timem': 0, 'mixup': 0, 'mode': 'evaluation', 'noise': False})  # Val training  sample_rate
    
    train_loader = torch.utils.data.DataLoader(
        dataloader.AudiosetDataset(args.data_train, audio_conf=audio_conf, label_csv=args.label_csv),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        dataloader.AudiosetDataset(args.data_val, audio_conf=val_audio_conf, label_csv=args.label_csv),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True
    )
    
    # 
    train(model, train_loader, val_loader, args)
    
    #    16kHz  (Validation + Test)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[5] Testing candidate {candidate_idx} with 16kHz audio...")
    
    test_audio_conf = {
        'num_mel_bins': 128,
        'target_length': 1024,  # FSD-Kaggle2018: 1024
        'freqm': 0,
        'timem': 0,
        'mixup': 0,
        'dataset': 'fsdkaggle2018',
        'mode': 'evaluation',
        'mean': -1.6813264,  # FSD-Kaggle2018 mean
        'std': 2.4526765,   # FSD-Kaggle2018 std
        'noise': False,
        'sample_rate': 16000  # 16kHz 
    }
    
    # Validation set 
    print("   Evaluating on Validation Set (16kHz)...")
    val_loader = torch.utils.data.DataLoader(
        dataloader.AudiosetDataset(
            './data/datafiles/fsd_val_data.json',
            audio_conf=test_audio_conf,
            label_csv='./data/fsd_class_labels_indices.csv'
        ),
        batch_size=48, shuffle=False, num_workers=8, pin_memory=True
    )
    
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for batch_data, batch_label in val_loader:
            batch_data = batch_data.to(device)
            batch_label = batch_label.to(device)
            batch_output = model(batch_data)
            batch_output = batch_output  # No sigmoid for CE loss
            _, predicted = torch.max(batch_output, 1)
            _, target = torch.max(batch_label, 1)
            val_correct += (predicted == target).sum().item()
            val_total += batch_label.size(0)
    
    val_accuracy = val_correct / val_total
    print(f"   Candidate {candidate_idx} Val Accuracy (16kHz): {val_accuracy:.4f}")
    
    # Test set 
    print("   Evaluating on Test Set (16kHz)...")
    test_loader = torch.utils.data.DataLoader(
        dataloader.AudiosetDataset(
            './data/datafiles/fsd_test_data.json',  # Test set
            audio_conf=test_audio_conf,
            label_csv='./data/fsd_class_labels_indices.csv'
        ),
        batch_size=48, shuffle=False, num_workers=8, pin_memory=True
    )
    
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for batch_data, batch_label in test_loader:
            batch_data = batch_data.to(device)
            batch_label = batch_label.to(device)
            batch_output = model(batch_data)
            batch_output = batch_output  # No sigmoid for CE loss
            _, predicted = torch.max(batch_output, 1)
            _, target = torch.max(batch_label, 1)
            test_correct += (predicted == target).sum().item()
            test_total += batch_label.size(0)
    
    test_accuracy = test_correct / test_total
    print(f"   Candidate {candidate_idx} Test Accuracy (16kHz): {test_accuracy:.4f}")
    
    #     
    with open(log_file, 'a') as f:
        f.write(f"\nFinal Results (16kHz):\n")
        f.write(f"Val Accuracy: {val_accuracy:.4f}\n")
        f.write(f"Test Accuracy: {test_accuracy:.4f}\n")
    
    print(f" Candidate {candidate_idx} training completed!")
    return model

def create_model_soup(candidate_paths):
    """
     candidate    (Model Soup)
    """
    print(f"\n[5] Creating Model Soup from {len(candidate_paths)} candidates...")
    
    if len(candidate_paths) == 0:
        print(" No candidate models found!")
        return None
    
    soup_state_dict = {}
    
    #  candidate  
    all_state_dicts = []
    for i, path in enumerate(candidate_paths):
        model_path = os.path.join(path, 'models', 'final_audio_model.pth')
        if os.path.exists(model_path):
            print(f"   Loading candidate {i}: {os.path.basename(path)}")
            state_dict = torch.load(model_path, map_location='cpu')
            all_state_dicts.append(state_dict)
        else:
            print(f"   Warning: Model not found: {model_path}")
    
    if len(all_state_dicts) == 0:
        print(" No valid candidate models found!")
        return None
    
    #  
    print(f"   Computing average of {len(all_state_dicts)} models...")
    for key in all_state_dicts[0].keys():
        weight_list = [state_dict[key] for state_dict in all_state_dicts]
        soup_state_dict[key] = torch.stack(weight_list).mean(dim=0)
    
    print(f" Model Soup created with {len(all_state_dicts)} models")
    return soup_state_dict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate_idx', type=int, default=0, help='Candidate index (0-9)')
    parser.add_argument('--num_epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'soup'],
                        help='train: train single candidate, soup: create model soup')
    parser.add_argument('--rate_mode', type=str, default='qasts', choices=['qasts', 'soup'],
                        help='qasts: 5-rate diversity+frozen scales, soup: fixed-16k baseline')
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if args.mode == 'train':
        # Single candidate training
        print("="*80)
        print(f"Training Candidate {args.candidate_idx} for FSD-Kaggle2018")
        print("="*80)
        
        # 1. QAT wrapper 
        qat_model = create_qat_wrapper()
        
        # 2. QAT  
        qat_model = load_qat_weights(qat_model)
        if qat_model is None:
            return
        
        # 3.  
        val_acc, test_acc = verify_qat_performance(qat_model, device)
        print(f"\n   QAT Baseline Summary:")
        print(f"   - Validation: {val_acc:.4f}")
        print(f"   - Test: {test_acc:.4f}")
        
        # 4. Candidate 
        train_candidate_with_diversity(
            qat_model,
            args.candidate_idx,
            args.num_epochs,
            rate_mode=args.rate_mode
        )
        
    else:
        # Model Soup creation
        print("="*80)
        print("Creating Model Soup")
        print("="*80)
        
        # qat_config.yml   
        with open('qat_config.yml', 'r') as f:
            config = yaml.safe_load(f)
        bit_w = config.get('wq_bitw', 8)
        bit_a = config.get('aq_bitw', 8)
        
        candidate_paths = []
        exp_tag = 'fixscale' if args.rate_mode == 'qasts' else 'norate'
        for i in range(10):  # 10 candidates
            path = f'./exp/soup_candidate{i}_W{bit_w}A{bit_a}_{exp_tag}'
            if os.path.exists(path):
                candidate_paths.append(path)
        
        soup_state_dict = create_model_soup(candidate_paths)
        if soup_state_dict:
            # Save soup
            soup_dir = f'./exp/soup_final_W{bit_w}A{bit_a}_{exp_tag}'
            os.makedirs(soup_dir, exist_ok=True)
            os.makedirs(os.path.join(soup_dir, 'models'), exist_ok=True)
            soup_path = os.path.join(soup_dir, 'models', 'soup_model.pth')
            torch.save(soup_state_dict, soup_path)
            print(f" Model Soup saved to: {soup_path}")
            
            # Model Soup 16kHz 
            print("\n[6] Testing Model Soup with 16kHz audio...")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # QAT wrapper   soup 
            qat_model = create_qat_wrapper()
            
            # DataParallel  
            if not isinstance(qat_model, nn.DataParallel):
                qat_model = nn.DataParallel(qat_model)
            
            # Device  (DataParallel )
            qat_model = qat_model.to(device)
            
            #  dummy forward pass (  device )
            qat_model.train()
            # dtype float32  
            dummy_input = torch.randn(2, 128, 1024, dtype=torch.float32).to(device)
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=False):  # autocast 
                    _ = qat_model(dummy_input)
            
            # Soup weights 
            qat_model.load_state_dict(soup_state_dict, strict=False)
            qat_model.eval()
            
            # 16kHz 
            test_audio_conf = {
                'num_mel_bins': 128,
                'target_length': 1024,
                'freqm': 0,
                'timem': 0,
                'mixup': 0,
                'dataset': 'fsdkaggle2018',
                'mode': 'evaluation',
                'mean': -1.6813264,
                'std': 2.4526765,
                'noise': False,
                'sample_rate': 16000  # 16kHz 
            }
            
            test_loader = torch.utils.data.DataLoader(
                dataloader.AudiosetDataset(
                    './data/datafiles/fsd_val_data.json',
                    audio_conf=test_audio_conf,
                    label_csv='./data/fsd_class_labels_indices.csv'
                ),
                batch_size=48, shuffle=False, num_workers=8, pin_memory=True
            )
            
            correct = 0
            total = 0
            with torch.no_grad():
                for batch_data, batch_label in test_loader:
                    batch_data = batch_data.to(device)
                    batch_label = batch_label.to(device)
                    batch_output = qat_model(batch_data)
                    batch_output = batch_output  # No sigmoid for CE loss
                    _, predicted = torch.max(batch_output, 1)
                    _, target = torch.max(batch_label, 1)
                    correct += (predicted == target).sum().item()
                    total += batch_label.size(0)
            
            soup_accuracy = correct / total
            print(f"   Model Soup Test Accuracy (16kHz): {soup_accuracy:.4f}")
            
            #  
            with open(os.path.join(soup_dir, 'soup_result.txt'), 'w') as f:
                f.write("Model Soup Results\n")
                f.write("="*50 + "\n")
                f.write(f"Number of candidates: {len(candidate_paths)}\n")
                f.write(f"Test Accuracy (16kHz): {soup_accuracy:.4f}\n")
                f.write("="*50 + "\n")
            
            print(f" Model Soup evaluation completed!")

if __name__ == "__main__":
    main()
