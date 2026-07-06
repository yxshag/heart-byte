#!/usr/bin/env python3

"""
======================================================================================================
Author:         yxshag
Created:        25-06-2026
GitHub:         https://github.com/yxshag

Project:        heart-byte
File:           explore_data.py
Description:    The default 0.5 probability threshold is a bad fit for imbalanced
                classes like HYP (12%  prevalence) -- the model can get away with
                almost never predicting "positive" and still look fine on AUROC.

                This script finds a better per-class threshold using the VALIDATION
                set only (never the test set, to keep the final test numbers honest),
                then reports what test performance looks like at that chosen threshold.

                Two threshold-selection strategies are shown side by side:
                - "F1-optimal": the threshold that maximizes F1 score (balances
                    precision and recall equally) on validation data
                - "Youden's J": the threshold that maximizes (sensitivity + specificity - 1),
                    a classic choice in clinical diagnostics
======================================================================================================
"""

import numpy as np
from sklearn.metrics import f1_score, roc_curve, precision_score, recall_score
from xgboost import XGBClassifier

from train_baseline import extract_features, load_split, SUPERCLASSES, RESULTS_DIR
from error_analysis import train_models

def find_f1_optimal_threshold(y_true, y_prob):
    """
    Check the f1 value for all thresholds, return the best one.
    The reason we dont do binary search in this even though its way faster is
    because f1 values doesnt increase linearly with threshold.
    """
    candidate_thresholds = np.linspace(0.01, 0.99, 99)
    best_threshold, best_f1 = 0.5, -1
    for t in candidate_thresholds:
        preds = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_threshold = f1, t
    return best_threshold


def find_youden_threshold(y_true, y_prob):
    """
    Youden's J statistic = sensitivity + specificity - 1, maximized.
    Equivalent to finding the point on the ROC curve furthest from the
    diagonal (the "no better than random" line).
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j_scores = tpr - fpr  # sensitivity - (1 - specificity) = tpr - fpr
    best_idx = np.argmax(j_scores)
    return thresholds[best_idx]


def evaluate_at_threshold(y_true, y_prob, threshold):
    """
    Evaluate value of threshold at particular given value of threshold.
    """
    preds = (y_prob >= threshold).astype(int)
    precision = precision_score(y_true, preds, zero_division=0)
    recall = recall_score(y_true, preds, zero_division=0)
    f1 = f1_score(y_true, preds, zero_division=0)
    return precision, recall, f1


def main():
    X_train, y_train = load_split("train")
    X_val, y_val = load_split("val")
    X_test, y_test = load_split("test")

    feat_train = extract_features(X_train, "train")
    feat_val = extract_features(X_val, "val")
    feat_test = extract_features(X_test, "test")

    models = train_models(feat_train, y_train, feat_val, y_val)

    report_lines = ["THRESHOLD TUNING-chosen on validation set, evaluated on test set"]
    report_lines.append("=" * 75)

    for i, cls_name in enumerate(SUPERCLASSES):
        model = models[cls_name]

        # Notice that we are in no way using the test dataframe to find the value of threshold
        val_prob = model.predict_proba(feat_val)[:, 1]
        f1_threshold = find_f1_optimal_threshold(y_val[:, i], val_prob)
        youden_threshold = find_youden_threshold(y_val[:, i], val_prob)

        # Now apply the threshold values to test to report the true values
        test_prob = model.predict_proba(feat_test)[:, 1]
        test_prevalence = y_test[:, i].mean()

        report_lines.append(f"\n{cls_name} (test prevalence: {test_prevalence:.1%})")
        report_lines.append("-" * 50)

        for label, threshold in [
            ("Default (0.5)", 0.5),
            ("F1-optimal", f1_threshold),
            ("Youden's J", youden_threshold),
        ]:
            precision, recall, f1 = evaluate_at_threshold(y_test[:, i], test_prob, threshold)
            report_lines.append(
                f"  {label:18s} (t={threshold:.2f}) | "
                f"Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f}"
            )

    report_text = "\n".join(report_lines)
    print(report_text)

    out_path = RESULTS_DIR / "threshold_tuning.txt"
    with open(out_path, "w") as f:
        f.write(report_text)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
