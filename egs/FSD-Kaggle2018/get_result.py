#!/usr/bin/env python3
"""
Simple result collector for FSD-Kaggle2018 folder structure parity with ESC-50.

Prints the latest result files under robustness/ and exp/ if available.
"""

from pathlib import Path


def latest_file(pattern):
    files = sorted(Path('.').glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def main():
    paths = [
        latest_file('robustness/*.txt'),
        latest_file('exp/**/results.txt'),
        latest_file('exp/**/summary.txt'),
    ]
    printed = False
    for p in paths:
        if p is not None:
            print(p)
            printed = True
    if not printed:
        print('No result files found.')


if __name__ == '__main__':
    main()

