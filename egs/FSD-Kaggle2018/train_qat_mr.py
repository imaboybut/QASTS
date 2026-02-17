#!/usr/bin/env python3
"""
FSD-Kaggle2018 QAT+MR baseline.

Single QAT model is continuously fine-tuned with the Stage-3 budget:
- sample rates: {12,16,20,24,28} kHz, each repeated 2 times (10 stages)
- 3 epochs per stage (total 30 epochs)
- lr sampled from [1e-6, 1e-5] for each stage
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
from traintest import train

RATE_SET_HZ = [12000, 16000, 20000, 24000, 28000]
CANDIDATES_PER_RATE = 2
STAGE_RATE_PLAN = RATE_SET_HZ * CANDIDATES_PER_RATE


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_qat_wrapper():
    audio_model = models.ASTModel(
        label_dim=41,
        fstride=10,
        tstride=10,
        input_fdim=128,
        input_tdim=1024,
        imagenet_pretrain=True,
        audioset_pretrain=True,
        model_size='base384',
    )

    fp_ckpt_path = "./exp/test-fsdkaggle2018-f10-t10-impTrue-aspTrue-b24-lr1e-5/models/final_audio_model.pth"
    if os.path.exists(fp_ckpt_path):
        state_dict = torch.load(fp_ckpt_path, map_location="cpu")
        new_state_dict = {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}
        audio_model.load_state_dict(new_state_dict)

    with open('qat_config.yml', 'r') as f:
        qat_config = yaml.safe_load(f)
    qat_args = argparse.Namespace(**qat_config)
    return get_qat_model(audio_model, qat_args)


def load_qat_weights(qat_model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    qat_model = qat_model.to(device)
    qat_model.train()
    with torch.no_grad():
        _ = qat_model(torch.randn(1, 128, 1024, device=device))

    with open('qat_config.yml', 'r') as f:
        qat_config = yaml.safe_load(f)
    bit_w = qat_config.get('wq_bitw', 8)
    bit_a = qat_config.get('aq_bitw', 8)

    ckpt_path = f"./exp/qat_fsd_W{bit_w}A{bit_a}_baseline/models/final_audio_model.pth"
    if not os.path.exists(ckpt_path):
        print(f" QAT checkpoint not found: {ckpt_path}")
        return None, bit_w, bit_a

    state_dict = torch.load(ckpt_path, map_location="cpu")
    new_state_dict = {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}
    qat_model.load_state_dict(new_state_dict, strict=False)
    return qat_model, bit_w, bit_a


def evaluate_at_sample_rate(model, sample_rate, data_path, device):
    eval_conf = {
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
        'sample_rate': sample_rate,
    }
    loader = torch.utils.data.DataLoader(
        dataloader.AudiosetDataset(
            data_path,
            audio_conf=eval_conf,
            label_csv='./data/fsd_class_labels_indices.csv',
        ),
        batch_size=48,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_data, batch_label in loader:
            batch_data = batch_data.to(device)
            batch_label = batch_label.to(device)
            out = model(batch_data)
            _, pred = torch.max(out, 1)
            _, target = torch.max(batch_label, 1)
            correct += (pred == target).sum().item()
            total += batch_label.size(0)
    return correct / total if total > 0 else 0.0


def sequential_finetune(model, bit_w, bit_a, num_stages=10, epochs_per_stage=3, seed=42):
    set_seed(seed)
    if num_stages != len(STAGE_RATE_PLAN):
        raise ValueError(
            f"num_stages must be {len(STAGE_RATE_PLAN)} for paper setting, got {num_stages}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not isinstance(model, nn.DataParallel):
        model = nn.DataParallel(model)
    model = model.to(device)

    exp_root = f'./exp/seq_qat_fsd_W{bit_w}A{bit_a}_multirate'
    os.makedirs(exp_root, exist_ok=True)
    os.makedirs(os.path.join(exp_root, 'models'), exist_ok=True)
    log_file = os.path.join(exp_root, 'training_log.txt')

    with open(log_file, 'w') as f:
        f.write("FSD QAT+MR sequential fine-tuning\n")
        f.write("=" * 60 + "\n")
        f.write(f"Stages: {num_stages}, Epochs per stage: {epochs_per_stage}\n")
        f.write(f"Stage rate plan (Hz): {STAGE_RATE_PLAN}\n")
        f.write(f"Bit config: W{bit_w}A{bit_a}\n")
        f.write("=" * 60 + "\n")

    common_args = dict(
        data_train='./data/datafiles/fsd_train_data.json',
        data_val='./data/datafiles/fsd_val_data.json',
        label_csv='./data/fsd_class_labels_indices.csv',
        n_class=41,
        dataset='fsdkaggle2018',
        model='ast',
        fstride=10,
        tstride=10,
        batch_size=24,
        optim='adam',
        num_workers=8,
        save_model=True,
        freqm=24,
        timem=96,
        mixup=0,
        bal='none',
        noise=False,
        dataset_mean=-1.6813264,
        dataset_std=2.4526765,
        audio_length=1024,
        loss='CE',
        metrics='acc',
        loss_fn=nn.CrossEntropyLoss(),
        warmup=False,
        lrscheduler_start=3,
        lrscheduler_step=1,
        lrscheduler_decay=0.85,
        n_print_steps=100,
        wa=False,
    )

    for stage in range(num_stages):
        sample_rate = STAGE_RATE_PLAN[stage]
        lr = np.random.uniform(1e-6, 1e-5)
        exp_dir = os.path.join(exp_root, f'stage{stage}')
        os.makedirs(exp_dir, exist_ok=True)
        os.makedirs(os.path.join(exp_dir, 'models'), exist_ok=True)

        args = argparse.Namespace(
            **common_args,
            n_epochs=epochs_per_stage,
            lr=lr,
            exp_dir=exp_dir,
            target_sample_rate=sample_rate,
        )

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
            'sample_rate': sample_rate,
        }
        val_audio_conf = audio_conf.copy()
        val_audio_conf.update({'freqm': 0, 'timem': 0, 'mixup': 0, 'mode': 'evaluation', 'noise': False})

        train_loader = torch.utils.data.DataLoader(
            dataloader.AudiosetDataset(args.data_train, audio_conf=audio_conf, label_csv=args.label_csv),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        val_loader = torch.utils.data.DataLoader(
            dataloader.AudiosetDataset(args.data_val, audio_conf=val_audio_conf, label_csv=args.label_csv),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        print(f"[Stage {stage}] sr={sample_rate}Hz, epochs={epochs_per_stage}, lr={lr:.2e}")
        with open(log_file, 'a') as f:
            f.write(f"\n[Stage {stage}] sample_rate={sample_rate}, lr={lr:.2e}\n")

        train(model, train_loader, val_loader, args)

        stage_model_path = os.path.join(exp_dir, 'models', 'final_audio_model.pth')
        torch.save(model.state_dict(), stage_model_path)

    final_path = os.path.join(exp_root, 'models', 'final_audio_model.pth')
    torch.save(model.state_dict(), final_path)
    print(f"\n QAT+MR training complete. Final model: {final_path}")

    eval_rates = [16000, 8000, 14000, 22000, 30000, 44100]
    with open(log_file, 'a') as f:
        f.write("\n[Final Evaluation]\n")
    for sr in eval_rates:
        val_acc = evaluate_at_sample_rate(model, sr, './data/datafiles/fsd_val_data.json', device)
        test_acc = evaluate_at_sample_rate(model, sr, './data/datafiles/fsd_test_data.json', device)
        print(f"  sr={sr}Hz: val={val_acc:.4f}, test={test_acc:.4f}")
        with open(log_file, 'a') as f:
            f.write(f"sr={sr}Hz: val={val_acc:.4f}, test={test_acc:.4f}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FSD QAT+MR sequential fine-tuning')
    parser.add_argument('--stages', type=int, default=10, help='Number of stages (paper: 10)')
    parser.add_argument('--epochs', type=int, default=3, help='Epochs per stage (paper: 3)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    qat_model = create_qat_wrapper()
    qat_model, bit_w, bit_a = load_qat_weights(qat_model)
    if qat_model is None:
        raise SystemExit(1)

    sequential_finetune(
        model=qat_model,
        bit_w=bit_w,
        bit_a=bit_a,
        num_stages=args.stages,
        epochs_per_stage=args.epochs,
        seed=args.seed,
    )

