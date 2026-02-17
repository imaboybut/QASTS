#!/usr/bin/env python3
"""
FSD-Kaggle2018 Model Soup Robustness 
QAT, QAT+MR, Soup, QASTS       
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


def first_pth_in_dir(dir_path):
    if not os.path.isdir(dir_path):
        return None
    pth_files = sorted([f for f in os.listdir(dir_path) if f.endswith('.pth')])
    if not pth_files:
        return None
    return os.path.join(dir_path, pth_files[0])


def evaluate_at_sample_rate(qat_model, sample_rate, data_path, label_csv, device):
    """ sample rate  """
    
    eval_audio_conf = {
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
        'sample_rate': sample_rate
    }
    
    data_loader = torch.utils.data.DataLoader(
        dataloader.AudiosetDataset(data_path, audio_conf=eval_audio_conf, label_csv=label_csv),
        batch_size=48, shuffle=False, num_workers=8, pin_memory=True
    )
    
    qat_model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_data, batch_label in data_loader:
            batch_data = batch_data.to(device)
            batch_label = batch_label.to(device)
            
            batch_output = qat_model(batch_data)
            _, predicted = torch.max(batch_output, 1)
            _, target = torch.max(batch_label, 1)
            
            correct += (predicted == target).sum().item()
            total += batch_label.size(0)
    
    accuracy = correct / total
    return accuracy, correct, total

def load_qat_soup_model(model_path, bit_w, bit_a, device):
    """QAT Model Soup """
    
    print(f"  Loading: {model_path}")
    
    # 1. FP   
    audio_model = models.ASTModel(
        label_dim=41,
        fstride=10,
        tstride=10,
        input_fdim=128,
        input_tdim=1024,
        imagenet_pretrain=True,
        audioset_pretrain=True,
        model_size='base384'
    )
    
    # FP   ()
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
    
    # 2. QAT config  ( config )
    with open('./qat_config.yml', 'r') as f:
        qat_config = yaml.safe_load(f)
    
    #   
    qat_config['wq_bitw'] = bit_w
    qat_config['aq_bitw'] = bit_a
    
    # 3. QAT wrapper 
    qat_args = argparse.Namespace(**qat_config)
    qat_model = get_qat_model(audio_model, qat_args)
    
    # DataParallel 
    if not isinstance(qat_model, nn.DataParallel):
        qat_model = nn.DataParallel(qat_model)
    qat_model = qat_model.to(device)
    
    # 4. Scale factors 
    qat_model.train()
    dummy_input = torch.randn(2, 128, 1024).to(device)
    with torch.no_grad():
        _ = qat_model(dummy_input)
    
    # 5. Model weights 
    if not os.path.exists(model_path):
        print(f"     Model not found: {model_path}")
        return None
    
    state_dict = torch.load(model_path, map_location=device)
    qat_model.load_state_dict(state_dict, strict=False)
    print(f"     Model loaded")
    
    return qat_model

def evaluate_robustness(specified_bit_w=None, specified_bit_a=None):
    """
      
    specified_bit_w, specified_bit_a:   
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if specified_bit_w is not None and specified_bit_a is not None:
        bit_configs = [(specified_bit_w, specified_bit_a)]
    else:
        with open('./qat_config.yml', 'r') as f:
            qat_config = yaml.safe_load(f)
        bit_configs = [(qat_config.get('wq_bitw', 8), qat_config.get('aq_bitw', 8))]

    sample_rates = ROBUSTNESS_SAMPLE_RATES
    test_data_path = './data/datafiles/fsd_test_data.json'
    label_csv = './data/fsd_class_labels_indices.csv'

    os.makedirs('robustness', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    display_name = {
        'baseline': 'QAT',
        'qat_mr': 'QAT+MR',
        'soup': 'Soup',
        'qasts': 'QASTS',
    }

    for bit_w, bit_a in bit_configs:
        print(f"\n{'='*80}")
        print(f"Evaluating W{bit_w}A{bit_a} Models")
        print(f"{'='*80}")

        path_map = {
            'baseline': first_existing_path([
                f"./exp/qat_fsd_W{bit_w}A{bit_a}_baseline/models/final_audio_model.pth",
                f"./exp/qat_fsd_W{bit_w}A{bit_a}_baseline/models/best_audio_model.pth",
            ]),
            'qat_mr': first_existing_path([
                f"./exp/seq_qat_fsd_W{bit_w}A{bit_a}_multirate/models/final_audio_model.pth",
                "./exp/seq_qat_fsd_multirate/models/final_audio_model.pth",
            ]),
            'soup': first_existing_path([
                f"./exp/soup_final_W{bit_w}A{bit_a}_norate/models/soup_model.pth",
                first_pth_in_dir(f"./models/{bit_w}{bit_a}_soup"),
            ]),
            'qasts': first_existing_path([
                f"./exp/soup_final_W{bit_w}A{bit_a}_fixscale/models/soup_model.pth",
                first_pth_in_dir(f"./models/{bit_w}{bit_a}_ours"),
            ]),
        }

        results = {}

        for model_key in ['baseline', 'qat_mr', 'soup', 'qasts']:
            model_path = path_map[model_key]
            if model_path is None:
                print(f"\n[{display_name[model_key]}] Model not found, skipping...")
                continue

            print(f"\n[{display_name[model_key]}] Loading: {model_path}")
            qat_model = load_qat_soup_model(model_path, bit_w, bit_a, device)
            if qat_model is None:
                continue

            test_results = []
            for sr in sample_rates:
                print(f"  Testing at {sr} Hz...", end="")
                test_acc, test_correct, test_total = evaluate_at_sample_rate(
                    qat_model, sr, test_data_path, label_csv, device
                )
                print(f" Accuracy: {test_acc:.4f}")
                test_results.append(test_acc)

            results[model_key] = test_results
            del qat_model
            torch.cuda.empty_cache()

        if results:
            result_file = f"robustness/W{bit_w}A{bit_a}_comparison_{timestamp}.txt"
            with open(result_file, 'w') as f:
                f.write(f"Robustness Evaluation Comparison\n")
                f.write(f"Configuration: W{bit_w}A{bit_a}\n")
                f.write(f"Timestamp: {timestamp}\n")
                f.write(f"Test Set: 11,005 samples\n")
                f.write(f"{'='*80}\n\n")
                
                #   
                table_data = []
                for i, sr in enumerate(sample_rates):
                    row = [f"{sr/1000:.0f}k" if sr < 44100 else "44.1k"]

                    for model_key in ['baseline', 'qat_mr', 'soup', 'qasts']:
                        if model_key in results:
                            row.append(f"{results[model_key][i]:.4f}")
                        else:
                            row.append("-")

                    #   (QASTS - QAT)
                    if 'qasts' in results and 'baseline' in results:
                        diff = results['qasts'][i] - results['baseline'][i]
                        row.append(f"{diff:+.4f}")
                    else:
                        row.append("-")

                    table_data.append(row)

                headers = ["Sample Rate", "QAT", "QAT+MR", "Soup", "QASTS", "QASTS-QAT"]
                f.write("Test Set Results:\n")
                f.write(tabulate(table_data, headers=headers, tablefmt="grid"))
                f.write("\n\n")

                f.write("Statistics:\n")
                f.write("="*80 + "\n")

                for model_key in ['baseline', 'qat_mr', 'soup', 'qasts']:
                    if model_key in results:
                        accs = results[model_key]
                        f.write(f"{display_name[model_key]}:\n")
                        f.write(f"  Mean: {np.mean(accs):.4f}\n")
                        f.write(f"  Std:  {np.std(accs):.4f}\n")
                        f.write(f"  Best: {max(accs):.4f} at {sample_rates[accs.index(max(accs))]/1000:.0f}kHz\n")
                        f.write(f"  Worst: {min(accs):.4f} at {sample_rates[accs.index(min(accs))]/1000:.0f}kHz\n\n")

                if len(results) > 1:
                    f.write("COMPARISON:\n")
                    all_stds = {
                        display_name[k]: np.std(v) for k, v in results.items()
                    }
                    most_robust = min(all_stds.items(), key=lambda x: x[1])
                    f.write(f"  Most Robust: {most_robust[0]} (std={most_robust[1]:.4f})\n")

                    if 'qasts' in results and 'baseline' in results:
                        qasts_gain = np.mean([
                            results['qasts'][i] - results['baseline'][i]
                            for i in range(len(sample_rates))
                        ])
                        f.write(f"  Avg Improvement (QASTS - QAT): {qasts_gain:.4f}\n")

            print(f"\n Results saved to: {result_file}")

    print(f"\n{'='*80}")
    print(f" All evaluations completed!")
    print(f"Results saved in ./robustness/ folder")
    print(f"{'='*80}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate Model Robustness')
    parser.add_argument('--bit_w', type=int, default=None, 
                        help='Weight quantization bits (default: from qat_config.yml)')
    parser.add_argument('--bit_a', type=int, default=None,
                        help='Activation quantization bits (default: from qat_config.yml)')
    args = parser.parse_args()
    
    if (args.bit_w is None) != (args.bit_a is None):
        print("Error: Both --bit_w and --bit_a must be specified together or both left unspecified")
        exit(1)
    
    evaluate_robustness(args.bit_w, args.bit_a)
