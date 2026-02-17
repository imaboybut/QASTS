#!/usr/bin/env python3
"""
QAT 5-fold   
 QAT     
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
import numpy as np
import argparse

sys.path.append('../../src')
import dataloader
import models
from models import get_qat_model

def evaluate_qat_fold(fold):
    """
     fold QAT  
    """
    print(f"\n{'='*80}")
    print(f"Evaluating QAT Model - Fold {fold}")
    print(f"{'='*80}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # 1. QAT config 
    with open('qat_config.yml', 'r') as f:
        qat_config = yaml.safe_load(f)
    bit_w = qat_config.get('wq_bitw', 8)
    bit_a = qat_config.get('aq_bitw', 8)
    print(f"Quantization: W{bit_w}A{bit_a}")
    
    # 2. QAT wrapper 
    print(f"\n[1] Creating QAT wrapper...")
    
    # FP   
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
    
    # FP    (QAT wrapper  )
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
        print(f"   FP model loaded from: {fp_ckpt_path}")
    else:
        print(f"   Warning: FP model not found: {fp_ckpt_path}")
    
    # QAT wrapper 
    qat_args = argparse.Namespace(**qat_config)
    qat_model = get_qat_model(audio_model, qat_args)
    print(f"   QAT wrapper created")
    
    # 3. Scale factors  (dummy forward)
    print(f"\n[2] Initializing scale factors...")
    qat_model = qat_model.to(device)
    qat_model.train()
    dummy_input = torch.randn(1, 512, 128).to(device)
    with torch.no_grad():
        _ = qat_model(dummy_input)
    print("   Scale factors initialized")
    
    # 4. QAT  
    print(f"\n[3] Loading QAT weights...")
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
    
    #   
    try:
        qat_model.load_state_dict(new_state_dict, strict=True)
        print(f" QAT weights loaded from: {qat_ckpt_path}")
    except RuntimeError as e:
        print(f"   Warning: Some keys mismatched, trying non-strict load...")
        qat_model.load_state_dict(new_state_dict, strict=False)
        print(f" QAT weights loaded (non-strict)")
    
    # Scale factors 
    scale_count = sum(1 for name, _ in qat_model.named_parameters() if '.s' in name)
    print(f"   Total {scale_count} scale factors in model")
    
    # 5.   
    print(f"\n[4] Evaluating on test set...")
    qat_model.eval()
    
    # DataParallel 
    if torch.cuda.device_count() > 1:
        print(f"   Using {torch.cuda.device_count()} GPUs")
        qat_model = nn.DataParallel(qat_model)
    qat_model = qat_model.to(device)
    
    # 6. Test   
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
        'sample_rate': 16000  # 16kHz (ESC-50 standard)
    }
    
    test_loader = torch.utils.data.DataLoader(
        dataloader.AudiosetDataset(
            f'./data/datafiles/esc_eval_data_{fold}.json',
            audio_conf=test_audio_conf,
            label_csv='./data/esc_class_labels_indices.csv'
        ),
        batch_size=48, shuffle=False, num_workers=8, pin_memory=True
    )
    
    print(f"   Test samples: {len(test_loader.dataset)}")
    print(f"   Sample rate: 16kHz (44.1kHz  16kHz)")
    
    # 7.  
    correct = 0
    total = 0
    class_correct = np.zeros(50)
    class_total = np.zeros(50)
    
    with torch.no_grad():
        for batch_idx, (batch_data, batch_label) in enumerate(test_loader):
            batch_data = batch_data.to(device)
            batch_label = batch_label.to(device)
            
            # Forward pass
            batch_output = qat_model(batch_data)
            
            # 
            _, predicted = torch.max(batch_output, 1)
            _, target = torch.max(batch_label, 1)
            
            #  
            correct += (predicted == target).sum().item()
            total += batch_label.size(0)
            
            #  
            for i in range(len(target)):
                label = target[i].item()
                class_correct[label] += (predicted[i] == target[i]).item()
                class_total[label] += 1
            
            #   
            if (batch_idx + 1) % 5 == 0:
                print(f"   Batch [{batch_idx+1}/{len(test_loader)}] - "
                      f"Running accuracy: {correct/total:.4f}")
    
    # 8.  
    accuracy = correct / total
    print(f"\n{'='*60}")
    print(f"Fold {fold} Results")
    print(f"{'='*60}")
    print(f"Overall Accuracy: {accuracy:.4f} ({correct}/{total})")
    print(f"Per-class Accuracy: {np.mean(class_correct/class_total):.4f}")
    
    # /  
    class_acc = class_correct / class_total
    best_classes = np.argsort(class_acc)[-5:][::-1]
    worst_classes = np.argsort(class_acc)[:5]
    
    print(f"\nTop 5 classes:")
    for idx in best_classes:
        print(f"  Class {idx}: {class_acc[idx]:.4f}")
    
    print(f"\nBottom 5 classes:")
    for idx in worst_classes:
        print(f"  Class {idx}: {class_acc[idx]:.4f}")
    
    return accuracy, class_acc

def evaluate_all_folds():
    """
    5 fold  
    """
    print("\n" + "="*80)
    print("QAT 5-Fold Evaluation")
    print("="*80)
    
    all_accuracies = []
    all_class_acc = []
    
    for fold in range(1, 6):
        result = evaluate_qat_fold(fold)
        if result is not None:
            accuracy, class_acc = result
            all_accuracies.append(accuracy)
            all_class_acc.append(class_acc)
        else:
            print(f" Failed to evaluate Fold {fold}")
    
    if len(all_accuracies) > 0:
        #   
        print("\n" + "="*80)
        print("5-Fold Summary")
        print("="*80)
        
        for i, acc in enumerate(all_accuracies, 1):
            print(f"Fold {i}: {acc:.4f}")
        
        print(f"\nMean Accuracy: {np.mean(all_accuracies):.4f}")
        print(f"Std Accuracy: {np.std(all_accuracies):.4f}")
        print(f"Max Accuracy: {np.max(all_accuracies):.4f} (Fold {np.argmax(all_accuracies)+1})")
        print(f"Min Accuracy: {np.min(all_accuracies):.4f} (Fold {np.argmin(all_accuracies)+1})")
        
        #   
        mean_class_acc = np.mean(all_class_acc, axis=0)
        print(f"\nMean Per-class Accuracy: {np.mean(mean_class_acc):.4f}")
        
        #  
        with open('qat_config.yml', 'r') as f:
            qat_config = yaml.safe_load(f)
        bit_w = qat_config.get('wq_bitw', 8)
        bit_a = qat_config.get('aq_bitw', 8)
        
        result_file = f'qat_W{bit_w}A{bit_a}_evaluation_results.txt'
        with open(result_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write(f"QAT W{bit_w}A{bit_a} 5-Fold Evaluation Results\n")
            f.write("="*80 + "\n\n")
            
            for i, acc in enumerate(all_accuracies, 1):
                f.write(f"Fold {i}: {acc:.4f}\n")
            
            f.write(f"\nMean Accuracy: {np.mean(all_accuracies):.4f}\n")
            f.write(f"Std Accuracy: {np.std(all_accuracies):.4f}\n")
            f.write(f"Max Accuracy: {np.max(all_accuracies):.4f} (Fold {np.argmax(all_accuracies)+1})\n")
            f.write(f"Min Accuracy: {np.min(all_accuracies):.4f} (Fold {np.argmin(all_accuracies)+1})\n")
            f.write(f"\nMean Per-class Accuracy: {np.mean(mean_class_acc):.4f}\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("Class-wise Performance (averaged across folds)\n")
            f.write("="*80 + "\n")
            
            sorted_classes = np.argsort(mean_class_acc)[::-1]
            for idx in sorted_classes:
                f.write(f"Class {idx:2d}: {mean_class_acc[idx]:.4f}\n")
        
        print(f"\n Results saved to: {result_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='QAT Model Evaluation')
    parser.add_argument('--fold', type=int, default=0, 
                        help='Specific fold to evaluate (0 for all)')
    args = parser.parse_args()
    
    if args.fold > 0:
        #  fold 
        print(f"Evaluating Fold {args.fold}...")
        result = evaluate_qat_fold(args.fold)
        if result is not None:
            accuracy, _ = result
            print(f"\n Fold {args.fold} QAT Accuracy: {accuracy:.4f}")
    else:
        #  fold 
        evaluate_all_folds()