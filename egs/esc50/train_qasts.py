#!/usr/bin/env python3
"""
QAT   Model Soup 
- QAT wrapper    
- 10 candidate   LR/sampling rate fine-tuning
- Model Soup   
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


def get_candidate_sample_rate(candidate_idx):
    """ : {12,16,20,24,28}kHz   2 ."""
    if candidate_idx < 0 or candidate_idx >= TOTAL_CANDIDATES:
        raise ValueError(
            f"candidate_idx must be in [0, {TOTAL_CANDIDATES - 1}], got {candidate_idx}"
        )
    rate_group = candidate_idx // CANDIDATES_PER_RATE
    return RATE_SET_HZ[rate_group]


def set_seed(seed):
    """   """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_qat_wrapper(fold):
    """
    QAT wrapper  (  )
    FP  QAT  
    """
    print(f"\n[1] Creating QAT wrapper for Fold {fold}...")
    
    # 1.  FP   
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
    
    # 2. FP    (QAT wrapper  )
    fp_ckpt_path = f"exp/fp_fold{fold}_baseline/models/final_audio_model.pth"
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

def load_qat_weights(qat_model, fold):
    """
    QAT   wrapper 
    """
    print(f"\n[2] Loading QAT weights for Fold {fold}...")
    
    #  scale factors   dummy forward pass
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    qat_model = qat_model.to(device)
    qat_model.train()
    dummy_input = torch.randn(1, 512, 128).to(device)
    with torch.no_grad():
        _ = qat_model(dummy_input)
    print("   Scale factors initialized with dummy forward pass")
    
    # qat_config.yml bit  
    with open('qat_config.yml', 'r') as f:
        qat_config = yaml.safe_load(f)
    bit_w = qat_config.get('wq_bitw', 8)
    bit_a = qat_config.get('aq_bitw', 8)
    
    # QAT   (scale factors )
    qat_ckpt_path = f"exp/qat_fold{fold}_W{bit_w}A{bit_a}_baseline/models/final_audio_model.pth"
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

def verify_qat_performance(model, fold, device):
    """
     QAT   
    """
    print(f"\n[3] Verifying QAT model performance...")
    
    model = model.to(device)
    model.eval()
    
    # Validation    ( sample rate )
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
        'sample_rate': 16000  # FP  : 16kHz 
    }
    print(f"   Validation Sample Rate: 16kHz (ESC-50 default)")
    
    val_loader = torch.utils.data.DataLoader(
        dataloader.AudiosetDataset(
            f'./data/datafiles/esc_eval_data_{fold}.json',
            audio_conf=val_audio_conf,
            label_csv='./data/esc_class_labels_indices.csv'
        ),
        batch_size=48, shuffle=False, num_workers=8, pin_memory=True
    )
    
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_data, batch_label in val_loader:
            batch_data = batch_data.to(device)
            batch_label = batch_label.to(device)
            batch_output = model(batch_data)
            _, predicted = torch.max(batch_output, 1)
            _, target = torch.max(batch_label, 1)
            correct += (predicted == target).sum().item()
            total += batch_label.size(0)
    
    accuracy = correct / total
    print(f"   QAT Model Accuracy: {accuracy:.4f} ({correct}/{total})")
    
    return accuracy

def train_candidate_with_diversity(model, fold, candidate_idx, num_epochs=8):
    """
    Diversity  candidate  
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Candidate   ( )
    candidate_seed = 14444 + candidate_idx  #  candidate  
    np.random.seed(candidate_seed)
    
    # Diversity 
    lr = np.random.uniform(5e-6, 5e-5)
    sample_rate = get_candidate_sample_rate(candidate_idx)
    
    print(f"\n[4-{candidate_idx}] Training candidate {candidate_idx} (Fixed Scale)")
    print(f"   Seed: {candidate_seed}")
    print(f"   Learning rate: {lr:.2e}")
    print(f"   Training Sample rate: {sample_rate} Hz (44.1kHz  {sample_rate} Hz)")
    print(f"   Validation Sample rate: {sample_rate} Hz (same as training)")
    
    # Stage 3 (QASTS): learned quantization scale 
    scale_params_frozen = 0
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
    args = argparse.Namespace(
        data_train=f'./data/datafiles/esc_train_data_{fold}.json',
        data_val=f'./data/datafiles/esc_eval_data_{fold}.json',
        label_csv='./data/esc_class_labels_indices.csv',
        n_class=50,
        dataset='esc50',
        model='ast',
        fstride=10,
        tstride=10,
        n_epochs=num_epochs,
        batch_size=24,
        lr=lr,  # Diversity:  LR
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
        lrscheduler_start=3,
        lrscheduler_step=1,
        lrscheduler_decay=0.85,
        n_print_steps=100,
        exp_dir=f'./exp/soup_fold{fold}_candidate{candidate_idx}_fixscale',
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
        f.write(f"Fold: {fold}\n")
        f.write(f"Seed: {candidate_seed}\n")
        f.write(f"Learning Rate: {lr:.2e}\n")
        f.write(f"Sample Rate: {sample_rate} Hz\n")
        f.write(f"Rate Group: {(candidate_idx // CANDIDATES_PER_RATE) + 1}/{len(RATE_SET_HZ)}\n")
        f.write(f"Epochs: {num_epochs}\n")
        f.write(f"Batch Size: {args.batch_size}\n")
        f.write(f"Frozen scale params: {scale_params_frozen}\n")
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
    val_audio_conf.update({'freqm': 0, 'timem': 0, 'mixup': 0, 'mode': 'evaluation', 'noise': False})
    
    train_loader = torch.utils.data.DataLoader(
        dataloader.AudiosetDataset(args.data_train, audio_conf=audio_conf, label_csv=args.label_csv),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        dataloader.AudiosetDataset(args.data_val, audio_conf=val_audio_conf, label_csv=args.label_csv),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True
    )
    
    # 
    print(f"   Starting {num_epochs}-epoch fine-tuning...")
    train(model, train_loader, val_loader, args)
    
    #   (   sample rate)
    model.eval()
    stats, _ = validate(model, val_loader, args, num_epochs)
    final_acc = stats[0].get('acc', 0.0) if isinstance(stats, list) and len(stats) > 0 else 0.0
    
    print(f"   Candidate {candidate_idx} accuracy @ {sample_rate}Hz: {final_acc:.4f}")
    
    # 16kHz  ( )
    print(f"\n   Testing with 16kHz (standard evaluation)...")
    test_audio_conf = {
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
        'sample_rate': 16000  # 16kHz 
    }
    
    test_loader = torch.utils.data.DataLoader(
        dataloader.AudiosetDataset(
            f'./data/datafiles/esc_eval_data_{fold}.json',
            audio_conf=test_audio_conf,
            label_csv='./data/esc_class_labels_indices.csv'
        ),
        batch_size=48, shuffle=False, num_workers=8, pin_memory=True
    )
    
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_data, batch_label in test_loader:
            batch_data = batch_data.to(device)
            batch_label = batch_label.to(device)
            batch_output = model(batch_data)
            _, predicted = torch.max(batch_output, 1)
            _, target = torch.max(batch_label, 1)
            correct += (predicted == target).sum().item()
            total += batch_label.size(0)
    
    test_acc_16k = correct / total
    print(f"   Candidate {candidate_idx} accuracy @ 16kHz: {test_acc_16k:.4f}")
    
    #  
    with open(log_file, 'a') as f:
        f.write(f"\nTraining Results:\n")
        f.write(f"Final Accuracy @ {sample_rate}Hz: {final_acc:.4f}\n")
        f.write(f"Test Accuracy @ 16kHz: {test_acc_16k:.4f}\n")
        f.write(f"Difference: {test_acc_16k - final_acc:+.4f}\n")
    
    return args.exp_dir, test_acc_16k, sample_rate

def create_model_soup(candidate_paths, fold):
    """
     candidate    (Model Soup)
    """
    print(f"\n[5] Creating Model Soup from {len(candidate_paths)} candidates...")
    
    if len(candidate_paths) == 0:
        print(" No candidate models found!")
        return None
    
    # qat_config.yml bit  
    with open('qat_config.yml', 'r') as f:
        qat_config = yaml.safe_load(f)
    bit_w = qat_config.get('wq_bitw', 8)
    bit_a = qat_config.get('aq_bitw', 8)
    
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
    
    #    key     
    print(f"   Computing average of {len(all_state_dicts)} models...")
    for key in all_state_dicts[0].keys():
        #    key  
        weight_list = [state_dict[key] for state_dict in all_state_dicts]
        #  
        soup_state_dict[key] = torch.stack(weight_list).mean(dim=0)
    
    # Soup   (bit  )
    soup_dir = f'./exp/soup_fold{fold}_W{bit_w}A{bit_a}_final_fixscale'
    os.makedirs(soup_dir, exist_ok=True)
    os.makedirs(os.path.join(soup_dir, 'models'), exist_ok=True)
    
    soup_path = os.path.join(soup_dir, 'models', 'soup_model.pth')
    torch.save(soup_state_dict, soup_path)
    
    print(f" Model Soup saved to: {soup_path}")
    print(f"   Averaged {len(all_state_dicts)} models")
    
    return soup_state_dict

def evaluate_soup_model(soup_state_dict, fold):
    """
    Model Soup  
    """
    print(f"\n[6] Evaluating Model Soup performance...")
    print(f"   Test Sample Rate: 16kHz (44.1kHz  16kHz, ESC-50 default)")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # QAT wrapper   soup  
    qat_model = create_qat_wrapper(fold)
    
    # soup_state_dict key   
    # DataParallel   module. prefix 
    new_state_dict = {}
    for k, v in soup_state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    
    # Scale factors   dummy forward pass
    qat_model = qat_model.to(device)
    qat_model.train()
    dummy_input = torch.randn(1, 512, 128).to(device)
    with torch.no_grad():
        _ = qat_model(dummy_input)
    print("   Scale factors initialized")
    
    #  soup  
    qat_model.load_state_dict(new_state_dict, strict=False)
    print("   Soup weights loaded")
    
    if not isinstance(qat_model, nn.DataParallel):
        qat_model = nn.DataParallel(qat_model)
    qat_model = qat_model.to(device)
    qat_model.eval()
    
    # Validation  
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
        'sample_rate': 16000  # Model Soup  : 16kHz 
    }
    
    val_loader = torch.utils.data.DataLoader(
        dataloader.AudiosetDataset(
            f'./data/datafiles/esc_eval_data_{fold}.json',
            audio_conf=val_audio_conf,
            label_csv='./data/esc_class_labels_indices.csv'
        ),
        batch_size=48, shuffle=False, num_workers=8, pin_memory=True
    )
    
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_data, batch_label in val_loader:
            batch_data = batch_data.to(device)
            batch_label = batch_label.to(device)
            batch_output = qat_model(batch_data)
            _, predicted = torch.max(batch_output, 1)
            _, target = torch.max(batch_label, 1)
            correct += (predicted == target).sum().item()
            total += batch_label.size(0)
    
    soup_accuracy = correct / total
    print(f"   Model Soup Accuracy: {soup_accuracy:.4f} ({correct}/{total})")
    
    return soup_accuracy

def train_soup_for_fold(fold, num_candidates=10, seed=42):
    """
     fold   Model Soup 
    """
    print(f"\n{'='*80}")
    print(f"Fold {fold} Model Soup Training")
    print(f"{'='*80}")
    if num_candidates != TOTAL_CANDIDATES:
        raise ValueError(
            f"QASTS requires exactly {TOTAL_CANDIDATES} candidates "
            f"(5 rates x 2), got {num_candidates}"
        )
    
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. QAT wrapper 
    qat_model = create_qat_wrapper(fold)
    
    # 2. QAT  
    qat_model = load_qat_weights(qat_model, fold)
    if qat_model is None:
        return None
    
    # 3. QAT  
    qat_accuracy = verify_qat_performance(qat_model, fold, device)
    
    # 4. Candidate  
    candidate_paths = []
    candidate_accuracies = []
    candidate_rates = []  #  candidate  sample rate 
    
    for i in range(num_candidates):
        #  candidate QAT  
        candidate_model = create_qat_wrapper(fold)
        candidate_model = load_qat_weights(candidate_model, fold)
        
        # QAT     (  )
        print(f"\n   [Pre-training Test] Candidate {i} - Testing loaded QAT model...")
        pre_train_acc = verify_qat_performance(candidate_model, fold, device)
        print(f"   QAT baseline for candidate {i}: {pre_train_acc:.4f}")
        
        # Diversity  fine-tuning (16kHz   )
        exp_dir, acc_16k, sample_rate = train_candidate_with_diversity(
            candidate_model, fold, i, num_epochs=8
        )
        candidate_paths.append(exp_dir)
        candidate_accuracies.append(acc_16k)  # 16kHz  
        candidate_rates.append(sample_rate)
    
    # 5. Model Soup 
    soup_state_dict = create_model_soup(candidate_paths, fold)
    
    # 6. Soup  
    soup_accuracy = evaluate_soup_model(soup_state_dict, fold)
    
    # # 7. Candidate    (  ,  )
    print(f"\n[7] Cleaning up candidate model files...")
    for path in candidate_paths:
        model_file = os.path.join(path, 'models', 'final_audio_model.pth')
        if os.path.exists(model_file):
            os.remove(model_file)
            print(f"   Deleted model: {model_file}")
        else:
            print(f"   Model not found: {model_file}")
    
    # 8.  
    print(f"\n{'='*80}")
    print(f"Fold {fold} Results Summary")
    print(f"{'='*80}")
    print(f"Original QAT Accuracy @ 16kHz: {qat_accuracy:.4f}")
    print(f"Candidate Rates (Hz): {candidate_rates}")
    print(f"Candidate Accuracies @ 16kHz: {[f'{acc:.4f}' for acc in candidate_accuracies]}")
    print(f"Model Soup Accuracy @ 16kHz: {soup_accuracy:.4f}")
    print(f"Improvement: {(soup_accuracy - qat_accuracy)*100:+.2f}%")
    
    # qat_config.yml bit   (summary )
    with open('qat_config.yml', 'r') as f:
        qat_config = yaml.safe_load(f)
    bit_w = qat_config.get('wq_bitw', 8)
    bit_a = qat_config.get('aq_bitw', 8)
    
    #     (bit  )
    summary_file = f'./exp/soup_fold{fold}_W{bit_w}A{bit_a}_final_fixscale/summary.txt'
    with open(summary_file, 'w') as f:
        f.write(f"{'='*80}\n")
        f.write(f"Fold {fold} Results Summary\n")
        f.write(f"{'='*80}\n")
        f.write(f"QASTS Rate Set (Hz): {RATE_SET_HZ}\n")
        f.write(f"Candidates per Rate: {CANDIDATES_PER_RATE}\n")
        f.write(f"Candidate Rates (Hz): {candidate_rates}\n")
        f.write(f"Original QAT Accuracy: {qat_accuracy:.4f}\n")
        f.write(f"Candidate Accuracies: {[f'{acc:.4f}' for acc in candidate_accuracies]}\n")
        f.write(f"Model Soup Accuracy: {soup_accuracy:.4f}\n")
        f.write(f"Improvement: {(soup_accuracy - qat_accuracy)*100:.2f}%\n")
        f.write(f"\nDetailed Candidate Results:\n")
        f.write(f"{'-'*80}\n")
        for i, acc in enumerate(candidate_accuracies):
            f.write(f"Candidate {i}: {acc:.4f}\n")
        f.write(f"{'-'*80}\n")
        f.write(f"Average Candidate Accuracy: {np.mean(candidate_accuracies):.4f}\n")
        f.write(f"Best Candidate: {np.argmax(candidate_accuracies)} (Accuracy: {max(candidate_accuracies):.4f})\n")
        f.write(f"Worst Candidate: {np.argmin(candidate_accuracies)} (Accuracy: {min(candidate_accuracies):.4f})\n")
    
    print(f" Summary saved to: {summary_file}")
    
    return {
        'fold': fold,
        'qat_acc': qat_accuracy,
        'candidate_accs': candidate_accuracies,
        'soup_acc': soup_accuracy
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model Soup Training for QAT Models')
    parser.add_argument('--fold', type=int, default=0, help='Specific fold to train (0 for all)')
    parser.add_argument('--num_candidates', type=int, default=10, help='Number of candidate models')
    args = parser.parse_args()
    
    if args.fold > 0:
        #  fold 
        result = train_soup_for_fold(args.fold, args.num_candidates)
        print(f"\n Fold {args.fold} Model Soup completed")
    else:
        # 5 fold  
        all_results = []
        for fold in range(1, 6):
            result = train_soup_for_fold(fold, args.num_candidates)
            all_results.append(result)
        
        #   
        print(f"\n{'='*80}")
        print("5-Fold Model Soup Summary")
        print(f"{'='*80}")
        for res in all_results:
            print(f"Fold {res['fold']}: QAT {res['qat_acc']:.4f}  Soup {res['soup_acc']:.4f}")
        
        avg_qat = np.mean([r['qat_acc'] for r in all_results])
        avg_soup = np.mean([r['soup_acc'] for r in all_results])
        print(f"\nAverage: QAT {avg_qat:.4f}  Soup {avg_soup:.4f}")
        print(f"Average Improvement: {(avg_soup - avg_qat)*100:.2f}%")
