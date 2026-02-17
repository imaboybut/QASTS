#!/usr/bin/env python3
"""
ESC-50 Model Soup Robustness 
QAT, QAT+MR, Soup, QASTS       
5-fold cross validation 
evaluate_qat_baseline.py  
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
import numpy as np
import argparse
from datetime import datetime
from tabulate import tabulate

sys.path.append('../../src')
import dataloader
import models
from models import get_qat_model

ROBUSTNESS_SAMPLE_RATES = [8000, 14000, 22000, 30000, 44100]


def first_existing_path(candidates):
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def create_qat_wrapper(fold):
    """
    QAT wrapper  (  )
    FP  QAT  
    """
    print(f"    Creating QAT wrapper for Fold {fold}...")
    
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
        print(f"      FP model loaded for QAT wrapper creation")
    
    # 3. QAT wrapper  (  )
    print("      Converting to QAT wrapper...")
    with open('qat_config.yml', 'r') as f:
        qat_config = yaml.safe_load(f)
    
    #      
    qat_args = argparse.Namespace(**qat_config)
    qat_model = get_qat_model(audio_model, qat_args)
    
    return qat_model

def load_model(model_path, fold, device):
    """
      - evaluate_qat_baseline.py  
    """
    print(f"    Loading: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"       Model not found: {model_path}")
        return None
    
    # 1. QAT wrapper 
    qat_model = create_qat_wrapper(fold)
    
    # 2. device   train  scale factors 
    qat_model = qat_model.to(device)
    qat_model.train()
    dummy_input = torch.randn(1, 512, 128).to(device)
    with torch.no_grad():
        _ = qat_model(dummy_input)
    print("      Scale factors initialized")
    
    # 3.   
    state_dict = torch.load(model_path, map_location="cpu")
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    
    #   ( strict=True )
    try:
        qat_model.load_state_dict(new_state_dict, strict=True)
        print("       Model weights loaded (strict=True)")
    except RuntimeError as e:
        print(f"      Warning: Some keys mismatched, using strict=False")
        qat_model.load_state_dict(new_state_dict, strict=False)
        print("      Model weights loaded (strict=False)")
    
    # Scale factors 
    scale_count = sum(1 for name, _ in qat_model.named_parameters() if '.s' in name)
    print(f"      Total {scale_count} scale factors in model")
    
    # 4.   
    qat_model.eval()
    
    # 5. DataParallel  ( )
    if torch.cuda.device_count() > 1:
        print(f"      Using {torch.cuda.device_count()} GPUs")
        qat_model = nn.DataParallel(qat_model)
    qat_model = qat_model.to(device)
    
    return qat_model

def evaluate_at_sample_rate(model, sample_rate, fold, device):
    """ sample rate  """
    
    eval_audio_conf = {
        'num_mel_bins': 128,
        'target_length': 512,  # ESC-50: 512
        'freqm': 0,
        'timem': 0,
        'mixup': 0,
        'dataset': 'esc50',
        'mode': 'evaluation',
        'mean': -6.6268077,  # ESC-50 mean
        'std': 5.358466,     # ESC-50 std
        'noise': False,
        'sample_rate': sample_rate
    }
    
    data_loader = torch.utils.data.DataLoader(
        dataloader.AudiosetDataset(
            f'./data/datafiles/esc_eval_data_{fold}.json',
            audio_conf=eval_audio_conf,
            label_csv='./data/esc_class_labels_indices.csv'
        ),
        batch_size=48, shuffle=False, num_workers=8, pin_memory=True
    )
    
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_data, batch_label in data_loader:
            batch_data = batch_data.to(device)
            batch_label = batch_label.to(device)
            
            batch_output = model(batch_data)
            _, predicted = torch.max(batch_output, 1)
            _, target = torch.max(batch_label, 1)
            
            correct += (predicted == target).sum().item()
            total += batch_label.size(0)
    
    accuracy = correct / total
    return accuracy, correct, total

def evaluate_robustness_fold(fold, bit_w, bit_a):
    """
     fold robustness 
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*80}")
    print(f"Evaluating Fold {fold} - W{bit_w}A{bit_a}")
    print(f"Device: {device}")
    print(f"{'='*80}")

    sample_rates = ROBUSTNESS_SAMPLE_RATES
    results = {}

    path_map = {
        'baseline': first_existing_path([
            f"./exp/qat_fold{fold}_W{bit_w}A{bit_a}_baseline/models/final_audio_model.pth",
            f"./exp/qat_fold{fold}_W{bit_w}A{bit_a}_baseline/models/best_audio_model.pth",
            f"./exp/qat_fold{fold}_baseline/models/final_audio_model.pth",
        ]),
        'qat_mr': first_existing_path([
            f"./exp/seq_qat_fold{fold}_W{bit_w}A{bit_a}_multirate/models/final_audio_model.pth",
            f"./exp/seq_qat_fold{fold}_multirate/models/final_audio_model.pth",
        ]),
        'soup': first_existing_path([
            f"./exp/soup_fold{fold}_W{bit_w}A{bit_a}_final_norate/models/soup_model.pth",
            f"./exp/soup_fold{fold}_final_norate/models/soup_model.pth",
            f"./models/{bit_w}{bit_a}_soup/fold{fold}_soup_model.pth",
            "./models/44_soup/fold{fold}_soup_model.pth".format(fold=fold),
        ]),
        'qasts': first_existing_path([
            f"./exp/soup_fold{fold}_W{bit_w}A{bit_a}_final_fixscale/models/soup_model.pth",
            f"./models/{bit_w}{bit_a}_ours/fold{fold}_soup_model.pth",
            "./models/44_ours/fold{fold}_soup_model.pth".format(fold=fold),
        ]),
    }

    display_name = {
        'baseline': 'QAT',
        'qat_mr': 'QAT+MR',
        'soup': 'Soup',
        'qasts': 'QASTS',
    }

    for model_key in ['baseline', 'qat_mr', 'soup', 'qasts']:
        model_path = path_map[model_key]
        if model_path is None:
            print(f"\n[{display_name[model_key]}] Model not found")
            continue

        print(f"\n[{display_name[model_key]}] Evaluating {model_path}")
        qat_model = load_model(model_path, fold, device)
        if qat_model is None:
            continue

        test_results = []
        for sr in sample_rates:
            print(f"  Testing at {sr} Hz...", end="")
            test_acc, test_correct, test_total = evaluate_at_sample_rate(
                qat_model, sr, fold, device
            )
            print(f" Accuracy: {test_acc:.4f} ({test_correct}/{test_total})")
            test_results.append(test_acc)

        results[model_key] = test_results
        del qat_model
        torch.cuda.empty_cache()

    return results, sample_rates

def main():
    parser = argparse.ArgumentParser(description='Evaluate ESC-50 Model Robustness')
    parser.add_argument('--bit_w', type=int, default=None, 
                        help='Weight quantization bits (default: from qat_config.yml)')
    parser.add_argument('--bit_a', type=int, default=None,
                        help='Activation quantization bits (default: from qat_config.yml)')
    parser.add_argument('--fold', type=int, default=None,
                        help='Specific fold to evaluate (1-5, default: all folds)')
    args = parser.parse_args()
    
    #   
    if args.bit_w is None or args.bit_a is None:
        with open('./qat_config.yml', 'r') as f:
            qat_config = yaml.safe_load(f)
        bit_w = args.bit_w or qat_config.get('wq_bitw', 8)
        bit_a = args.bit_a or qat_config.get('aq_bitw', 8)
    else:
        bit_w = args.bit_w
        bit_a = args.bit_a
    
    print(f"Using bit configuration: W{bit_w}A{bit_a}")
    
    # robustness  
    os.makedirs('robustness', exist_ok=True)
    
    # 
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Fold 
    if args.fold:
        folds = [args.fold]
    else:
        folds = range(1, 6)  # 1-5
    
    all_results = {}
    
    for fold in folds:
        results, sample_rates = evaluate_robustness_fold(fold, bit_w, bit_a)
        if results:
            all_results[fold] = results
            
            # Fold  
            result_file = f"robustness/fold{fold}_W{bit_w}A{bit_a}_comparison_{timestamp}.txt"
            with open(result_file, 'w') as f:
                f.write(f"ESC-50 Robustness Evaluation - Fold {fold}\n")
                f.write(f"Configuration: W{bit_w}A{bit_a}\n")
                f.write(f"Timestamp: {timestamp}\n")
                f.write(f"Test Set: 400 samples per fold\n")
                f.write(f"{'='*80}\n\n")
                
                #   
                table_data = []
                for i, sr in enumerate(sample_rates):
                    row = [f"{sr/1000:.0f}k" if sr < 44100 else "44.1k"]

                    for model_type in ['baseline', 'qat_mr', 'soup', 'qasts']:
                        if model_type in results:
                            row.append(f"{results[model_type][i]:.4f}")
                        else:
                            row.append("-")

                    #   (QASTS - Baseline)
                    if 'qasts' in results and 'baseline' in results:
                        diff = results['qasts'][i] - results['baseline'][i]
                        row.append(f"{diff:+.4f}")
                    else:
                        row.append("-")
                    
                    table_data.append(row)
                
                #  
                headers = ["Sample Rate", "QAT", "QAT+MR", "Soup", "QASTS", "QASTS-QAT"]
                f.write("Test Set Results:\n")
                f.write(tabulate(table_data, headers=headers, tablefmt="grid"))
                f.write("\n\n")
                
                # 
                f.write("Statistics:\n")
                f.write("="*80 + "\n")
                
                name_map = {
                    'baseline': 'QAT',
                    'qat_mr': 'QAT+MR',
                    'soup': 'Soup',
                    'qasts': 'QASTS',
                }
                for model_type in ['baseline', 'qat_mr', 'soup', 'qasts']:
                    if model_type in results:
                        accs = results[model_type]
                        f.write(f"{name_map[model_type]}:\n")
                        f.write(f"  Mean: {np.mean(accs):.4f}\n")
                        f.write(f"  Std:  {np.std(accs):.4f}\n")
                        f.write(f"  Best: {max(accs):.4f} at {sample_rates[accs.index(max(accs))]/1000:.0f}kHz\n")
                        f.write(f"  Worst: {min(accs):.4f} at {sample_rates[accs.index(min(accs))]/1000:.0f}kHz\n\n")
                
                # 
                if len(results) > 1:
                    f.write("COMPARISON:\n")
                    all_stds = {}
                    for model_type in ['baseline', 'qat_mr', 'soup', 'qasts']:
                        if model_type in results:
                            all_stds[name_map[model_type]] = np.std(results[model_type])
                    
                    if all_stds:
                        most_robust = min(all_stds.items(), key=lambda x: x[1])
                        f.write(f"  Most Robust: {most_robust[0]} (std={most_robust[1]:.4f})\n")
                    
                    if 'qasts' in results and 'baseline' in results:
                        improvements = [results['qasts'][i] - results['baseline'][i] for i in range(len(sample_rates))]
                        f.write(f"  Avg Improvement (QASTS - QAT): {np.mean(improvements):.4f}\n")
            
            print(f"\n Fold {fold} results saved to: {result_file}")
    
    #  fold   ( fold  )
    if len(all_results) > 1:
        avg_file = f"robustness/all_folds_W{bit_w}A{bit_a}_average_{timestamp}.txt"
        with open(avg_file, 'w') as f:
            f.write(f"ESC-50 Robustness Evaluation - Average across {len(all_results)} folds\n")
            f.write(f"Configuration: W{bit_w}A{bit_a}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"{'='*80}\n\n")
            
            #     
            avg_results = {}
            for model_type in ['baseline', 'qat_mr', 'soup', 'qasts']:
                fold_accs = []
                for fold, results in all_results.items():
                    if model_type in results:
                        fold_accs.append(results[model_type])
                
                if fold_accs:
                    #  sample rate 
                    avg_results[model_type] = np.mean(fold_accs, axis=0).tolist()
            
            #  
            if avg_results:
                table_data = []
                for i, sr in enumerate(sample_rates):
                    row = [f"{sr/1000:.0f}k" if sr < 44100 else "44.1k"]
                    
                    for model_type in ['baseline', 'qat_mr', 'soup', 'qasts']:
                        if model_type in avg_results:
                            row.append(f"{avg_results[model_type][i]:.4f}")
                        else:
                            row.append("-")
                    
                    if 'qasts' in avg_results and 'baseline' in avg_results:
                        diff = avg_results['qasts'][i] - avg_results['baseline'][i]
                        row.append(f"{diff:+.4f}")
                    else:
                        row.append("-")
                    
                    table_data.append(row)
                
                headers = ["Sample Rate", "QAT", "QAT+MR", "Soup", "QASTS", "QASTS-QAT"]
                f.write("Average Results across all folds:\n")
                f.write(tabulate(table_data, headers=headers, tablefmt="grid"))
                f.write("\n")
        
        print(f"\n Average results saved to: {avg_file}")
    
    print(f"\n{'='*80}")
    print(f" All evaluations completed!")
    print(f"Results saved in ./robustness/ folder")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
