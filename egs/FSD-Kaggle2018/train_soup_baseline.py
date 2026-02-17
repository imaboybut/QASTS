#!/usr/bin/env python3
"""
Wrapper for Soup baseline training on FSD-Kaggle2018.

This wrapper calls train_qasts.py with rate_mode='soup' so that
the CLI name matches ESC-50 folder structure.
"""

import argparse
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description='FSD Soup baseline wrapper')
    parser.add_argument('--candidate_idx', type=int, default=0, help='Candidate index (0-9)')
    parser.add_argument('--num_epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'soup'],
                        help='train: train single candidate, soup: create model soup')
    args = parser.parse_args()

    cmd = [
        sys.executable,
        'train_qasts.py',
        '--candidate_idx', str(args.candidate_idx),
        '--num_epochs', str(args.num_epochs),
        '--mode', args.mode,
        '--rate_mode', 'soup',
    ]
    raise SystemExit(subprocess.call(cmd))


if __name__ == '__main__':
    main()

