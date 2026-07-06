#!/usr/bin/env python3

"""
=======================================================================================================
Author:         yxshag
Created:        05-07-2026
GitHub:         https://github.com/yxshag

Project:        heart-byte
File:           generate_xgb_errors.py
Description:    Retrains the XGBoost baseline (same as train_baseline.py) and saves
                per-case false positive / false negative rows to xgb_test_errors.csv,
                in the same format as cnn_test_errors.csv, so the two can be compared
                directly by cross_model_error_comparison.py.
=======================================================================================================
"""

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from cnn_error_analysis import load_test_ids
from train_baseline import extract_features, load_split, SUPERCLASSES, PROCESSED_DIR

# F1-optimal thresholds HARDCODED IN from results of threshold_tuning.py
THRESHOLDS = {"NORM": 0.44, "MI": 0.22, "STTC": 0.26, "CD": 0.42, "HYP": 0.15}
ID_KEY_IN_NPZ = "ecg_ids"

def main():
    """Do the exact same thing as cnn_error_analysis.py but do it with the xgboost model instead of cnn model"""
    X_train, y_train = load_split("train")
    X_val, y_val = load_split("val")
    X_test, y_test = load_split("test")
    test_ids = load_test_ids()

    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    feat_train = extract_features(X_train, "train")
    feat_val = extract_features(X_val, "val")
    feat_test = extract_features(X_test, "test")

    all_rows = []

    for i, cls_name in enumerate(SUPERCLASSES):
        model = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            eval_metric="auc",
            early_stopping_rounds=20,
        )
        model.fit(
            feat_train, y_train[:, i],
            eval_set=[(feat_val, y_val[:, i])],
            verbose=False,
        )

        test_probs = model.predict_proba(feat_test)[:, 1]
        thresh = THRESHOLDS[cls_name]
        y_true = y_test[:, i]
        y_pred = (test_probs >= thresh).astype(int)

        fp_mask = (y_true == 0) & (y_pred == 1)
        fn_mask = (y_true == 1) & (y_pred == 0)
        
        for mask, error_type in [(fp_mask, "FP"), (fn_mask, "FN")]:
            idx = np.where(mask)[0]
            for j in idx:
                all_rows.append({
                    "class": cls_name,
                    "error_type": error_type,
                    "ecg_id": test_ids[j],
                    "predicted_prob": float(test_probs[j]),
                })

        n_fp, n_fn = fp_mask.sum(), fn_mask.sum()
        print(f"{cls_name:6s} | threshold: {thresh:.2f} | FP: {n_fp} | FN: {n_fn}")

    errors_df = pd.DataFrame(all_rows)
    errors_df.to_csv("results/xgb_test_errors.csv", index=False)
    print(f"\nSaved {len(errors_df)} FP/FN rows to results/xgb_test_errors.csv")


if __name__ == "__main__":
    main()
