
#!/usr/bin/env python3

"""
===========================================================================================
Author:         yxshag
Created:        22-06-2026
GitHub:         https://github.com/yxshag

Project:        heart-byte
File:           preprocessing.py
Description:    loads the dataframe(df)
                Steps:
                1. Bandpass filter -- strip out baseline wander (slow drift) and
                    high-frequency noise, keep the band that actually contains real
                    heartbeat signal.
                2. Per-record amplitude normalization -- so the model isn't thrown
                    off by scale differences between recordings/machines.
                3. Split into train (folds 1-8) / val (fold 9) / test (fold 10),
                    using the exact split we already verified has zero patient leakage.
                4. Save everything as compressed .npz files so we don't have to
                    re-do this expensive step every time we experiment with a model.
===========================================================================================
"""


import ast
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
from scipy.signal import butter, filtfilt
from tqdm import tqdm

from data_utils import *


#Paths and Constants-DO NOT CHANGE

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "physionet.org" / "files" / "ptb-xl" / "1.0.3"
OUT_DIR = PROJECT_ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

#Ideal constants for the bandpass filter- recommended not to change for ideal results

SAMPLING_RATE = 100  # Hz, since we're using records100/
LOWCUT_HZ = 0.5      #below this is baseline wander
HIGHCUT_HZ = 40.0    #above this is noise

SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]


def bandpass_filter(signal, fs=SAMPLING_RATE, lowcut=LOWCUT_HZ, highcut=HIGHCUT_HZ, order=4):
    """
    Apply a Butterworth band-pass filter to a multi-lead ECG signal.

    The filter removes:
        - Low-frequency baseline wander (e.g., respiration or electrode drift)
        - High-frequency noise (e.g., muscle activity and electrical interference)

    while preserving the frequency band that contains the clinically
    relevant ECG waveform.

    Args:
        signal (np.ndarray):
            ECG signal of shape (timesteps, n_leads).
        fs (float):
            Sampling frequency in Hz.
        lowcut (float):
            Lower cutoff frequency in Hz.
        highcut (float):
            Upper cutoff frequency in Hz.
        order (int):
            Order of the Butterworth filter. Higher orders produce a
            steeper frequency response.

    Returns:
        np.ndarray:
            Filtered ECG signal with the same shape as the input.
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band")

    filtered = np.zeros_like(signal)
    for lead in range(signal.shape[1]):
        #filtfilt does filters the signal backwards and forward so that the time sensitive features still remain
        filtered[:, lead] = filtfilt(b, a, signal[:, lead])
    return filtered


def normalize(signal):
    """
    Per-record, per-lead z-score normalization: subtract the mean, divide
    by the standard deviation. Puts every recording on the same scale.
    This is standard mathematical formula for normalisation.
    """
    mean = signal.mean(axis=0, keepdims=True)
    std = signal.std(axis=0, keepdims=True)
    std[std == 0] = 1.0  # guard against divide-by-zero on a flat/dead lead
    return (signal - mean) / std


def labels_to_multihot(superclass_list):
    """Convert a list like ['NORM', 'STTC'] into a fixed-order vector.[0,1,0,0,1]"""
    vec = np.zeros(len(SUPERCLASSES), dtype=np.float32)
    for cls in superclass_list:
        if cls in SUPERCLASSES:
            vec[SUPERCLASSES.index(cls)] = 1.0
    return vec


def process_split(df_split, split_name):
    """Load, filter, normalize every waveform in this split, return Features, label arrays."""
    Features = []
    label = []

    for ecg_id, row in tqdm(df_split.iterrows(), total=len(df_split), desc=f"Processing {split_name}"):
        record_path = DATA_ROOT / row["filename_lr"]
        signal, _ = wfdb.rdsamp(str(record_path))  # shape: (1000, 12) for 10s @ 100Hz

        signal = bandpass_filter(signal)
        signal = normalize(signal)

        Features.append(signal.astype(np.float32))
        label.append(labels_to_multihot(row["diagnostic_superclass"]))

    Features = np.stack(Features)  # (n_records, 1000, 12)
    label = np.stack(label)  # (n_records, 5)
    return Features, label


def main():
    """Load, split the data based on strat_folds, process the data, save the data in compressed files."""
    df = load_processed_metadata()

    # Drop any record with no diagnostic superclass at all
    # for this classification task.
    before = len(df)
    df = df[df["diagnostic_superclass"].apply(len) > 0]
    print(f"Dropped {before - len(df)} records with no diagnostic superclass label.")

    train_df = df[df["strat_fold"].isin(range(1, 9))]
    val_df = df[df["strat_fold"] == 9]
    test_df = df[df["strat_fold"] == 10]

    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    #Save 3 different files for 3 different pre-processed data of train, val and testing

    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        X, y = process_split(split_df, split_name)
        out_path = OUT_DIR / f"{split_name}.npz"
        np.savez_compressed(out_path, X=X, y=y, ecg_ids=split_df.index.values)
        print(f"Saved {split_name}: X={X.shape}, y={y.shape} -> {out_path}")

    print("\nLabel columns (in order):", SUPERCLASSES)
    print("Done.")


if __name__ == "__main__":
    main()
