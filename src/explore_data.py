#!/usr/bin/env python3

"""
================================================================================
Author:         yxshag
Created:        20-06-2026
GitHub:         https://github.com/yxshag

Project:        heart-byte
File:           explore_data.py
Description:    loads the data that was downloaded and does a sanity check to
                ensure that the ecg's are downloaded without any issues.
================================================================================
"""

import ast
import os
from pathlib import Path
from data_utils import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wfdb

#Paths-SHOULD NOT BE CHANGED
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "physionet.org" / "files" / "ptb-xl" / "1.0.3"


def summarize(df):
    print(f"Total records: {len(df)}")
    print(f"Unique patients: {df['patient_id'].nunique()}")
    print()

    print("Records per patient (top of distribution):")
    print(df["patient_id"].value_counts().describe())
    print()

    print("Diagnostic superclass frequency (a record can have more than one):")
    all_classes = df["diagnostic_superclass"].explode()
    print(all_classes.value_counts())
    print()

    print("Train/val/test fold distribution (PTB-XL's official `strat_fold` column,")
    print("already stratified to avoid patient leakage across folds):")
    print(df["strat_fold"].value_counts().sort_index())


def plot_sample_waveform(df, n_samples=1):
    """Plot one or more raw 12-lead ECG waveforms as a sanity check."""
    sample_ids = df.sample(n_samples, random_state=42).index

    for ecg_id in sample_ids:
        row = df.loc[ecg_id]
        record_path = DATA_ROOT / row["filename_lr"]  # low-res (100Hz) path
        signal, fields = wfdb.rdsamp(str(record_path)) #signal->ECG values, fields->metadata explaining which angle of the ecg it is 

        fig, axes = plt.subplots(12, 1, figsize=(10, 14), sharex=True)#creates 12 plots for 12-lead ecg
        for i, lead_name in enumerate(fields["sig_name"]):
            axes[i].plot(signal[:, i], linewidth=0.8)
            axes[i].set_ylabel(lead_name, rotation=0, labelpad=20)
            axes[i].set_yticks([])
        axes[-1].set_xlabel("Sample index (100 Hz)")
        fig.suptitle(
            f"ECG {ecg_id} | superclass(es): {row['diagnostic_superclass']}"
        )
        plt.tight_layout()

        out_path = PROJECT_ROOT / "results" / f"sample_ecg_{ecg_id}.png"
        plt.savefig(out_path, dpi=120)
        plt.close(fig)
        print(f"Saved sample waveform plot: {out_path}")


if __name__ == "__main__":
    df, scp_df = load_metadata()
    df = add_diagnostic_superclass(df, scp_df)

    summarize(df)
    plot_sample_waveform(df, n_samples=2)
