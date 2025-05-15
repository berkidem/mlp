"""
This script documents how we took the original occupation_sequences.pkl dataset
and split it into train.pkl, val.pkl, test.pkl splits.
TLDR we shuffled the names, took the first 1000 for val, rest for train.

Process:
- navigate to this directory (data/)
- run this script
python preprocess.py
- this creates train.pkl, val.pkl, test.pkl in the same directory
"""

import pandas as pd
import numpy as np
import pickle
import random
import os


def split_and_save_sequences(
    sequences_path="occupation_sequences.pkl",
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    seed=42,
):
    """Split sequences into train/val/test sets and save them"""
    assert (
        abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-10
    ), "Ratios must sum to 1"

    with open(sequences_path, "rb") as f:
        sequences = pickle.load(f)

    # Shuffle sequences with fixed seed for reproducibility
    random.seed(seed)
    random.shuffle(sequences)

    # to each sequence, add an end token, -1, numpy array
    sequences = [np.insert(seq, 0, 1016) for seq in sequences]

    # Calculate split indices
    n = len(sequences)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    # Split the data
    train_sequences = sequences[:train_end]
    val_sequences = sequences[train_end:val_end]
    test_sequences = sequences[val_end:]

    # Create output directory if it doesn't exist
    os.makedirs("data", exist_ok=True)

    # Save the splits as pickle files
    with open("train_sequences.pkl", "wb") as f:
        pickle.dump(train_sequences, f)

    with open("val_sequences.pkl", "wb") as f:
        pickle.dump(val_sequences, f)

    with open("test_sequences.pkl", "wb") as f:
        pickle.dump(test_sequences, f)

    # keeping the following in case I want to use the original code
    # Also save as text files for compatibility with original code structure
    # Each line is a space-separated sequence of occupation codes
    # with open('data/train.txt', 'w') as f:
    #     for seq in train_sequences:
    #         f.write(' '.join(map(str, seq)) + '\n')

    # with open('data/val.txt', 'w') as f:
    #     for seq in val_sequences:
    #         f.write(' '.join(map(str, seq)) + '\n')

    # with open('data/test.txt', 'w') as f:
    #     for seq in test_sequences:
    #         f.write(' '.join(map(str, seq)) + '\n')

    print(f"Split {n} sequences into:")
    print(f"  Train: {len(train_sequences)} sequences ({train_ratio*100:.1f}%)")
    print(f"  Validation: {len(val_sequences)} sequences ({val_ratio*100:.1f}%)")
    print(f"  Test: {len(test_sequences)} sequences ({test_ratio*100:.1f}%)")


split_and_save_sequences()