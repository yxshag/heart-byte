#!/usr/bin/env python3

"""
=======================================================================================================
Author:         yxshag
Created:        04-07-2026
GitHub:         https://github.com/yxshag

Project:        heart-byte
File:           cnn_error_analysis.py
Description:    This script evaluates a trained CNN model on the PTB-XL test dataset using 
                class-specific decision thresholds. It computes per-class confusion matrices and 
                isolates individual False Positive (FP) and False Negative (FN) error cases. These 
                error cases are mapped back to their original patient metadata to extract demographics 
                like age and sex. The individual errors are saved to a CSV file 
                (results/cnn_test_errors.csv) for detailed traceability. Finally, it outputs a 
                stratified breakdown of errors across different age groups and genders to highlight 
                potential subgroup biases in model performance.
=======================================================================================================
"""

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from cnn_inference_utils import CLASSES, load_split, load_model, get_predictions, get_device
import json

# ----------------------Paths-DO NOT CHANGE-----------------------------------
THRESHOLDS_PATH = "results/cnn_thresholds.json"
PTBXL_METADATA_PATH = "data/physionet.org/files/ptb-xl/1.0.3/ptbxl_database.csv" 
ID_KEY_IN_NPZ = "ecg_ids"                          
ID_COLUMN_IN_METADATA = "ecg_id"                     
# -----------------------------------------------------------------------------


def load_test_ids():
    data = np.load("data/processed/test.npz")
    if ID_KEY_IN_NPZ not in data:
        raise KeyError(
            f"'{ID_KEY_IN_NPZ}' not found in test.npz (keys present: {list(data.keys())}). "
            "Update ID_KEY_IN_NPZ in this script, or add ids to your preprocessing "
            "step so error cases can be traced back to specific patients."
        )
    return data[ID_KEY_IN_NPZ]


def main():
    device = get_device()
    print(f"Using device: {device}")

    with open(THRESHOLDS_PATH) as f:
        thresholds = json.load(f)

    model = load_model(device=device)
    X_test, y_test = load_split("test")
    test_probs = get_predictions(model, X_test, device)
    test_ids = load_test_ids()

    meta = pd.read_csv(PTBXL_METADATA_PATH)
    meta = meta.set_index(ID_COLUMN_IN_METADATA)

    all_fp_fn_rows = []

    print("\n" + "=" * 65)
    print("Confusion matrices + FP/FN breakdown (CNN, test set)")
    print("=" * 65)

    for i, cls in enumerate(CLASSES):
        thresh = thresholds[cls]
        y_true = y_test[:, i]
        y_prob = test_probs[:, i]
        y_pred = (y_prob >= thresh).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        print(f"\n{cls} (threshold={thresh:.3f})")
        print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")

        fp_mask = (y_true == 0) & (y_pred == 1)
        fn_mask = (y_true == 1) & (y_pred == 0)

        for mask, error_type in [(fp_mask, "FP"), (fn_mask, "FN")]:
            idx = np.where(mask)[0]
            for j in idx:
                all_fp_fn_rows.append({
                    "class": cls,
                    "error_type": error_type,
                    "ecg_id": test_ids[j],
                    "predicted_prob": float(y_prob[j]),
                })

    errors_df = pd.DataFrame(all_fp_fn_rows)
    errors_df.to_csv("results/cnn_test_errors.csv", index=False)
    print(f"\nSaved {len(errors_df)} FP/FN rows to results/cnn_test_errors.csv")

    # age/sex based analysis
    errors_df = errors_df.merge(
        meta[["age", "sex"]], left_on="ecg_id", right_index=True, how="left"
    )

    print("\n" + "=" * 65)
    print("Age / sex subgroup breakdown of errors, per class")
    print("=" * 65)

    # Build age bins consistent with a typical PTB-XL age-subgroup analysis
    bins = [0, 40, 55, 70, 120]
    labels = ["<40", "40-54", "55-69", "70+"]
    errors_df["age_group"] = pd.cut(errors_df["age"], bins=bins, labels=labels)

    for cls in CLASSES:
        cls_errors = errors_df[errors_df["class"] == cls]
        if cls_errors.empty:
            continue
        print(f"\n{cls}")
        print("  By sex:")
        print(cls_errors.groupby(["sex", "error_type"]).size().unstack(fill_value=0).to_string().replace("\n", "\n  "))
        print("  By age group:")
        print(cls_errors.groupby(["age_group", "error_type"], observed=False).size().unstack(fill_value=0).to_string().replace("\n", "\n  "))


if __name__ == "__main__":
    main()
