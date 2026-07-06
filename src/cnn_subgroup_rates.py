#!/usr/bin/env python3

"""
=======================================================================================================
Author:         yxshag
Created:        04-07-2026
GitHub:         https://github.com/yxshag

Project:        heart-byte
File:           cnn_subgroup_rates.py
Description:    Subgroup-normalized error analysis for the CNN: recall and precision
                per age group / sex, per class - not just raw FP/FN counts.
=======================================================================================================
"""

import numpy as np
import pandas as pd
import json

from cnn_inference_utils import CLASSES, load_split, load_model, get_predictions, get_device

# ---------------------Paths-DO NOT CHANGE-----------------------------------
THRESHOLDS_PATH = "results/cnn_thresholds.json"
PTBXL_METADATA_PATH = "data/physionet.org/files/ptb-xl/1.0.3/ptbxl_database.csv" 
ID_KEY_IN_NPZ = "ecg_ids"                                
ID_COLUMN_IN_METADATA = "ecg_id"                       
AGE_BINS = [0, 40, 55, 70, 120]
AGE_LABELS = ["<40", "40-54", "55-69", "70+"]
# -----------------------------------------------------------------------------


def load_test_ids():
    data = np.load("data/processed/test.npz")
    if ID_KEY_IN_NPZ not in data:
        raise KeyError(
            f"'{ID_KEY_IN_NPZ}' not found in test.npz (keys present: {list(data.keys())}). "
            "Update ID_KEY_IN_NPZ in this script to match your preprocessing output."
        )
    return data[ID_KEY_IN_NPZ]


def rate_table(df, group_col, cls):
    """Compute recall, precision, and support for every label grouped by a particular column"""
    rows = []
    for group_val, g in df.groupby(group_col, observed=False):
        tp = ((g.y_true == 1) & (g.y_pred == 1)).sum()
        fn = ((g.y_true == 1) & (g.y_pred == 0)).sum()
        fp = ((g.y_true == 0) & (g.y_pred == 1)).sum()
        tn = ((g.y_true == 0) & (g.y_pred == 0)).sum()

        recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        n_positive = tp + fn 

        rows.append({
            "class": cls,
            "group": group_val,
            "n_total": len(g),
            "n_positive_cases": n_positive,
            "recall": round(recall, 3) if n_positive > 0 else None,
            "precision": round(precision, 3) if (tp + fp) > 0 else None,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        })
    return pd.DataFrame(rows)


def main():
    device = get_device()
    print(f"Using device: {device}")

    with open(THRESHOLDS_PATH) as f:
        thresholds = json.load(f)

    model = load_model(device=device)
    X_test, y_test = load_split("test")
    test_probs = get_predictions(model, X_test, device)
    test_ids = load_test_ids()

    meta = pd.read_csv(PTBXL_METADATA_PATH).set_index(ID_COLUMN_IN_METADATA)
    demo = meta.loc[test_ids, ["age", "sex"]].reset_index(drop=True)
    demo["age_group"] = pd.cut(demo["age"], bins=AGE_BINS, labels=AGE_LABELS)

    all_age_rows = []
    all_sex_rows = []

    for i, cls in enumerate(CLASSES):
        thresh = thresholds[cls]
        y_pred = (test_probs[:, i] >= thresh).astype(int)
        y_true = y_test[:, i]

        df = demo.copy()
        df["y_true"] = y_true
        df["y_pred"] = y_pred

        all_age_rows.append(rate_table(df, "age_group", cls))
        all_sex_rows.append(rate_table(df, "sex", cls))

    age_df = pd.concat(all_age_rows, ignore_index=True)
    sex_df = pd.concat(all_sex_rows, ignore_index=True)

    age_df.to_csv("results/cnn_subgroup_rates_by_age.csv", index=False)
    sex_df.to_csv("results/cnn_subgroup_rates_by_sex.csv", index=False)

    print("\n" + "=" * 75)
    print("Recall / precision by AGE GROUP, per class")
    print("=" * 75)
    for cls in CLASSES:
        print(f"\n{cls}")
        sub = age_df[age_df["class"] == cls][["group", "n_positive_cases", "recall", "precision"]]
        print(sub.to_string(index=False))

    print("\n" + "=" * 75)
    print("Recall / precision by SEX, per class")
    print("=" * 75)
    for cls in CLASSES:
        print(f"\n{cls}")
        sub = sex_df[sex_df["class"] == cls][["group", "n_positive_cases", "recall", "precision"]]
        print(sub.to_string(index=False))

    print("\nSaved full tables to results/cnn_subgroup_rates_by_age.csv and results/cnn_subgroup_rates_by_sex.csv")
    print("(includes tp/fp/fn/tn and n_total per subgroup for further slicing)")


if __name__ == "__main__":
    main()
