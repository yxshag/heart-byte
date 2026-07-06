#!/usr/bin/env python3

"""
=======================================================================================================
Author:         yxshag
Created:        03-07-2026
GitHub:         https://github.com/yxshag

Project:        heart-byte
File:           tune_thresholds_cnn.py
Description:    Per-class F1-optimal threshold tuning for the CNN, mirroring the process
                already run for the XGBoost baseline.

                Tunes thresholds on the VAL set, then reports precision/recall/F1 and
                confusion matrices on the TEST set using those thresholds -- same
                tune-on-val / report-on-test discipline as before, so results stay
                comparable to the baseline's threshold-tuning step.
=======================================================================================================
"""
import json
import numpy as np
from sklearn.metrics import precision_recall_curve, f1_score, confusion_matrix

from cnn_inference_utils import CLASSES, load_split, load_model, get_predictions, get_device


def best_f1_threshold(y_true, y_prob):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-12)#1e-12 is so that the denominator is never 0
    #precision recall curve sweeps all the values of thresholds and the last one is the value for threshold=inf, so we are going to ignore that
    best_idx = np.argmax(f1s[:-1])
    return thresholds[best_idx], f1s[best_idx]


def main():
    device = get_device()
    print(f"Using device: {device}")

    model = load_model(device=device)

    X_val, y_val = load_split("val")
    X_test, y_test = load_split("test")

    val_probs = get_predictions(model, X_val, device)
    test_probs = get_predictions(model, X_test, device)

    thresholds = {}
    print("\n" + "=" * 65)
    print("F1-optimal thresholds (tuned on VAL set)")
    print("=" * 65)

    for i, cls in enumerate(CLASSES):
        thresh, val_f1 = best_f1_threshold(y_val[:, i], val_probs[:, i])
        thresholds[cls] = float(thresh)
        print(f"{cls:6s} | threshold: {thresh:.3f} | val F1 at threshold: {val_f1:.3f}")

    with open("results/cnn_thresholds.json", "w") as f:
        json.dump(thresholds, f, indent=2)
    print("\nSaved thresholds to results/cnn_thresholds.json")

    print("\n" + "=" * 65)
    print("TEST set performance using tuned thresholds")
    print("=" * 65)

    for i, cls in enumerate(CLASSES):
        thresh = thresholds[cls]
        y_true = y_test[:, i]
        y_pred = (test_probs[:, i] >= thresh).astype(int)

        f1 = f1_score(y_true, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)

        print(f"\n{cls} (threshold={thresh:.3f})")
        print(f"  Precision: {precision:.3f}  Recall: {recall:.3f}  F1: {f1:.3f}")
        print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")


if __name__ == "__main__":
    main()
