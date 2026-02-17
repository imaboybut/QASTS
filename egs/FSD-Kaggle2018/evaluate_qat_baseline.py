#!/usr/bin/env python3
"""
FSD-Kaggle2018 QAT   ( )
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
from traintest import validate

def evaluate_trained_qat():
    """  QAT  """
    
    print(f"\n{'='*80}")
    print(f"FSD-Kaggle2018 QAT Model Evaluation")
    print(f"{'='*80}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # QAT config 
    with open('./qat_config.yml', 'r') as f:
        qat_config = yaml.safe_load(f)
    
    bit_w = qat_config.get('wq_bitw', 8)
    bit_a = qat_config.get('aq_bitw', 8)
    print(f"QAT Configuration: W{bit_w}A{bit_a}")
    
    # 1. FP    (QAT wrapper  )
    print(f"\n[1] Creating base FP model structure...")
    audio_model = models.ASTModel(
        label_dim=41,  # FSD-Kaggle2018: 41 classes
        fstride=10,
        tstride=10,
        input_fdim=128,
        input_tdim=1024,  # FSD-Kaggle2018: 1024 frames
        imagenet_pretrain=True,
        audioset_pretrain=True,  # FSD uses audioset pretrain
        model_size='base384'
    )
    
    # FP    (QAT wrapper  )
    fp_ckpt_path = "./exp/test-fsdkaggle2018-f10-t10-impTrue-aspTrue-b24-lr1e-5/models/final_audio_model.pth"
    if os.path.exists(fp_ckpt_path):
        print(f"   Loading FP weights for QAT structure...")
        state_dict = torch.load(fp_ckpt_path, map_location="cpu")
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        audio_model.load_state_dict(new_state_dict)
        print(f"    FP weights loaded")
    
    # 2. QAT wrapper 
    print(f"\n[2] Converting to QAT model...")
    qat_args = argparse.Namespace(**qat_config)
    qat_model = get_qat_model(audio_model, qat_args)
    print(f" QAT model created with quantization modules")
    
    # DataParallel  
    if not isinstance(qat_model, nn.DataParallel):
        qat_model = nn.DataParallel(qat_model)
    qat_model = qat_model.to(device)
    
    # 3. Scale factors   dummy forward pass
    print(f"\n[3] Initializing scale factors...")
    qat_model.train()  # train  
    dummy_input = torch.randn(2, 128, 1024).to(device)  # batch size 2
    with torch.no_grad():
        _ = qat_model(dummy_input)
    print("   Scale factors initialized with dummy forward pass")
    
    # Scale factors  
    scale_count = sum(1 for name, _ in qat_model.named_parameters() if '.s' in name)
    print(f"   Total {scale_count} scale factors in model")
    
    # 4. QAT   
    print(f"\n[4] Loading QAT trained weights...")
    qat_model_path = f'./exp/qat_fsd_W{bit_w}A{bit_a}_baseline/models/final_audio_model.pth'
    
    if not os.path.exists(qat_model_path):
        # best  
        qat_model_path = qat_model_path.replace('final_audio_model.pth', 'best_audio_model.pth')
    
    if not os.path.exists(qat_model_path):
        print(f" QAT model not found at: {qat_model_path}")
        print(f"   Please train QAT model first using train_qat_baseline.py")
        return None, None
    
    print(f"   Loading from: {qat_model_path}")
    state_dict = torch.load(qat_model_path, map_location=device)
    
    # strict=False  (   )
    missing_keys, unexpected_keys = qat_model.load_state_dict(state_dict, strict=False)
    print(f" QAT weights loaded from: {qat_model_path}")
    
    if missing_keys:
        print(f"   Warning: Missing keys: {len(missing_keys)}")
        if len(missing_keys) < 10:
            for key in missing_keys[:10]:
                print(f"      - {key}")
    
    if unexpected_keys:
        print(f"   Warning: Unexpected keys: {len(unexpected_keys)}")
        if len(unexpected_keys) < 10:
            for key in unexpected_keys[:10]:
                print(f"      - {key}")
    
    # 5.  
    args = argparse.Namespace(
        data_train='./data/datafiles/fsd_train_data.json',
        data_val='./data/datafiles/fsd_val_data.json',
        data_eval='./data/datafiles/fsd_test_data.json',
        label_csv='./data/fsd_class_labels_indices.csv',
        n_class=41,
        dataset='fsdkaggle2018',
        batch_size=48,
        num_workers=8,
        dataset_mean=-1.6813264,
        dataset_std=2.4526765,
        audio_length=1024,
        loss='CE',
        metrics='acc',
        loss_fn=nn.CrossEntropyLoss(),
        exp_dir=f'./exp/qat_fsd_W{bit_w}A{bit_a}_baseline'
    )
    
    # 6.    (16kHz )
    print("\n[5] Creating data loaders...")
    eval_audio_conf = {
        'num_mel_bins': 128,
        'target_length': args.audio_length,
        'freqm': 0,
        'timem': 0,
        'mixup': 0,
        'dataset': args.dataset,
        'mode': 'evaluation',
        'mean': args.dataset_mean,
        'std': args.dataset_std,
        'noise': False,
        'sample_rate': 16000  # 16kHz  (!)
    }
    
    val_loader = torch.utils.data.DataLoader(
        dataloader.AudiosetDataset(args.data_val, audio_conf=eval_audio_conf, label_csv=args.label_csv),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        dataloader.AudiosetDataset(args.data_eval, audio_conf=eval_audio_conf, label_csv=args.label_csv),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True
    )
    
    # 7.  (   )
    print("\n[6] Evaluating model...")
    qat_model.eval()
    
    # Validation set 
    print("   Evaluating on Validation Set (9,981 samples)...")
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for i, (batch_data, batch_label) in enumerate(val_loader):
            batch_data = batch_data.to(device)
            batch_label = batch_label.to(device)
            
            batch_output = qat_model(batch_data)
            
            # CE loss  
            _, predicted = torch.max(batch_output, 1)
            _, target = torch.max(batch_label, 1)
            
            val_correct += (predicted == target).sum().item()
            val_total += batch_label.size(0)
            
            if i % 50 == 0:
                print(f"      Batch {i}/{len(val_loader)}, Running Acc: {val_correct/val_total:.4f}")
    
    val_acc_manual = val_correct / val_total
    print(f"   Manual Val Accuracy: {val_acc_manual:.4f} ({val_correct}/{val_total})")
    
    # validate  
    val_stats, _ = validate(qat_model, val_loader, args, 25)
    val_acc = val_stats[0]['acc'] if isinstance(val_stats, list) and len(val_stats) > 0 else 0.0
    print(f"   Validate Function Val Accuracy: {val_acc:.4f}")
    
    # Test set 
    print("\n   Evaluating on Test Set (11,005 samples)...")
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for i, (batch_data, batch_label) in enumerate(test_loader):
            batch_data = batch_data.to(device)
            batch_label = batch_label.to(device)
            
            batch_output = qat_model(batch_data)
            
            # CE loss  
            _, predicted = torch.max(batch_output, 1)
            _, target = torch.max(batch_label, 1)
            
            test_correct += (predicted == target).sum().item()
            test_total += batch_label.size(0)
            
            if i % 50 == 0:
                print(f"      Batch {i}/{len(test_loader)}, Running Acc: {test_correct/test_total:.4f}")
    
    test_acc_manual = test_correct / test_total
    print(f"   Manual Test Accuracy: {test_acc_manual:.4f} ({test_correct}/{test_total})")
    
    # validate  
    test_stats, _ = validate(qat_model, test_loader, args, 25)
    test_acc = test_stats[0]['acc'] if isinstance(test_stats, list) and len(test_stats) > 0 else 0.0
    print(f"   Validate Function Test Accuracy: {test_acc:.4f}")
    
    # 8.    
    print(f"\n{'='*80}")
    print(f"FSD-Kaggle2018 QAT Evaluation Results")
    print(f"{'='*80}")
    print(f"Model: W{bit_w}A{bit_a}")
    print(f"Model Path: {qat_model_path}")
    print(f"Sample Rate: 16kHz (downsampled from 44.1kHz)")
    print(f"{'='*80}")
    print(f"Validation Accuracy (Manual): {val_acc_manual:.4f}")
    print(f"Validation Accuracy (validate): {val_acc:.4f}")
    print(f"Test Accuracy (Manual): {test_acc_manual:.4f}")
    print(f"Test Accuracy (validate): {test_acc:.4f}")
    
    #   
    result_file = os.path.join(args.exp_dir, 'qat_evaluation_results.txt')
    with open(result_file, 'w') as f:
        f.write(f"FSD-Kaggle2018 QAT Evaluation Results\n")
        f.write(f"{'='*80}\n")
        f.write(f"Model: W{bit_w}A{bit_a}\n")
        f.write(f"Model Path: {qat_model_path}\n")
        f.write(f"Sample Rate: 16kHz\n")
        f.write(f"{'='*80}\n")
        f.write(f"Validation Accuracy (Manual): {val_acc_manual:.4f}\n")
        f.write(f"Validation Accuracy (validate): {val_acc:.4f}\n")
        f.write(f"Test Accuracy (Manual): {test_acc_manual:.4f}\n")
        f.write(f"Test Accuracy (validate): {test_acc:.4f}\n")
        f.write(f"{'='*80}\n")
        f.write(f"Scale factors in model: {scale_count}\n")
    
    print(f"\n Results saved to: {result_file}")
    
    # :    
    total_params = sum(p.numel() for p in qat_model.parameters())
    trainable_params = sum(p.numel() for p in qat_model.parameters() if p.requires_grad)
    print(f"\nModel Info:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Scale factors: {scale_count}")
    
    return val_acc_manual, test_acc_manual

if __name__ == "__main__":
    val_acc, test_acc = evaluate_trained_qat()
    if val_acc is not None:
        print(f"\n Evaluation completed!")
        print(f"   Final Validation Accuracy: {val_acc:.4f}")
        print(f"   Final Test Accuracy: {test_acc:.4f}")
    else:
        print(f"\n Evaluation failed - model not found")