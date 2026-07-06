#!/usr/bin/env python3

"""
==========================================================================================================================
Author:         yxshag
Created:        23-06-2026
GitHub:         https://github.com/yxshag

Project:        heart-byte
File:           train_baseline.py
Description:    1. Extracts 63 hand-crafted features per record:
                    - Rhythm features from lead II via neurokit2: heart rate, HRV (SDNN), beat count
                    - Per-lead stats across all 12 leads: mean, std, min, max, range (5 * 12 = 60 features)
                    - NaN fallback: if peak detection fails, fills with column median
                2. Trains 5 independent XGBoost binary classifiers (one per class)
                    - n_estimators=300,i max_depth=4, lr=0.05, early stopping on val AUC
                3. Evaluates on test set: AUROC + AUPRC per class
                X->Features, y->labels
                4. Change parameters in lines 128 to 134 to change how the model trains
==========================================================================================================================
"""

from pathlib import Path

import numpy as np
import neurokit2 as nk
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm
from xgboost import XGBClassifier

#Paths-DO NOT CHANGE

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

SAMPLING_RATE = 100
SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]

# Lead II is the classic choice for rhythm/HR analysis -- it's the lead
# most peak-detection algorithms (including neurokit2's defaults) are
# tuned against, and the standard ordering in PTB-XL's 12-lead arrays
# is: I, II, III, aVR, aVL, aVF, V1-V6 -- so lead II is index 1.
LEAD_II_INDEX = 1


def extract_features_one_record(signal):
    """
    signal shape: (1000, 12) -- 10 seconds at 100Hz, 12 leads.
    Returns a 1D feature vector combining:
      - rhythm features from lead II (heart rate, HRV, QRS-adjacent timing)
      - simple statistical features across all 12 leads
    """
    features = []
    lead_ii = signal[:, LEAD_II_INDEX]
    try:
        #getting R peaks
        _, info = nk.ecg_peaks(lead_ii, sampling_rate=SAMPLING_RATE)
        r_peaks = info["ECG_R_Peaks"]

        if len(r_peaks) >= 2:
            rr_intervals = np.diff(r_peaks) / SAMPLING_RATE  # seconds between beats
            heart_rate = 60.0 / rr_intervals.mean()
            hrv_sdnn = rr_intervals.std()  # standard deviation of RR intervals
            n_beats = len(r_peaks)
        else:
            #too less beats
            heart_rate, hrv_sdnn, n_beats = np.nan, np.nan, len(r_peaks)
    except Exception:
        # if it fails to find peaks, go back to default values
        heart_rate, hrv_sdnn, n_beats = np.nan, np.nan, 0

    features.extend([heart_rate, hrv_sdnn, n_beats])

    # --- Simple per-lead statistical features across all 12 leads ---
    # mean, std, min, max, and "peak-to-peak range" per lead
    for lead in range(signal.shape[1]):
        lead_signal = signal[:, lead]
        features.extend([
            lead_signal.mean(),
            lead_signal.std(),
            lead_signal.min(),
            lead_signal.max(),
            lead_signal.max() - lead_signal.min(),
        ])

    return np.array(features, dtype=np.float32)


def extract_features(X, split_name):
    """X shape: (n_records, 1000, 12) -> returns (n_records, n_features)"""
    feature_list = []
    for i in tqdm(range(X.shape[0]), desc=f"Extracting features ({split_name})"):
        feature_list.append(extract_features_one_record(X[i]))
    features = np.stack(feature_list)

    #Replace nan's with column median
    col_median = np.nanmedian(features, axis=0)
    nan_mask = np.isnan(features)
    features[nan_mask] = np.take(col_median, np.where(nan_mask)[1])

    return features


def load_split(split_name):
    data = np.load(PROCESSED_DIR / f"{split_name}.npz")
    return data["X"], data["y"]


def train_and_evaluate():
    X_train, y_train = load_split("train")
    X_val, y_val = load_split("val")
    X_test, y_test = load_split("test")

    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    feat_train = extract_features(X_train, "train")
    feat_val = extract_features(X_val, "val")
    feat_test = extract_features(X_test, "test")

    print(f"\nFeature vector length: {feat_train.shape[1]}")

    # make 5 different models for each of superclasses
    models = {}
    test_results = {}

    for i, cls_name in enumerate(SUPERCLASSES):
        model = XGBClassifier(
            n_estimators=300,        #300 trees
            max_depth=4,      
            learning_rate=0.05,
            eval_metric="auc",
            early_stopping_rounds=20,
        )
        #fit the model according to training and validation data
        model.fit(
            feat_train, y_train[:, i],
            eval_set=[(feat_val, y_val[:, i])],
            verbose=False,
        )
        models[cls_name] = model

        test_probs = model.predict_proba(feat_test)[:, 1]
        auroc = roc_auc_score(y_test[:, i], test_probs)
        auprc = average_precision_score(y_test[:, i], test_probs)
        prevalence = y_test[:, i].mean()

        test_results[cls_name] = {
            "AUROC": auroc,
            "AUPRC": auprc,
            "test_prevalence": prevalence,
        }

        print(
            f"{cls_name:6s} | AUROC: {auroc:.3f} | AUPRC: {auprc:.3f} "
            f"| test prevalence: {prevalence:.1%}"
        )

    return test_results


if __name__ == "__main__":
    results = train_and_evaluate()

    out_path = RESULTS_DIR / "baseline_results.txt"
    with open(out_path, "w") as f:
        f.write("Baseline model (hand-crafted features + XGBoost)\n")
        f.write("=" * 55 + "\n\n")
        for cls_name, metrics in results.items():
            f.write(
                f"{cls_name:6s} | AUROC: {metrics['AUROC']:.3f} "
                f"| AUPRC: {metrics['AUPRC']:.3f} "
                f"| test prevalence: {metrics['test_prevalence']:.1%}\n"
            )
    print(f"\nSaved results to {out_path}")
