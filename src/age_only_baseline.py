#!/usr/bin/env python3

"""
==============================================================================================
Author:         yxshag
Created:        27-06-2026
GitHub:         https://github.com/yxshag

Project:        heart-byte
File:           age_only_baseline.py
Description:    Sanity check: the error analysis showed recall climbing steeply with
                age for MI/STTC/CD/HYP. This raises a real question -- is the hand-
                crafted-feature model learning genuine ECG waveform patterns, or is
                it largely just exploiting "older patient -> more likely abnormal"?

                This script trains the simplest possible model using ONLY age (and
                sex) as input -- no waveform data at all -- and compares its AUROC
                to the full 63-feature baseline. If age-only gets close to the real
                model, that's a strong signal the real model isn't adding much beyond
                what age already implies.
==============================================================================================
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

from train_baseline import SUPERCLASSES, PROCESSED_DIR, PROJECT_ROOT, RESULTS_DIR

#Paths-DO NOT CHANGE

DATA_ROOT = PROJECT_ROOT / "data" / "physionet.org" / "files" / "ptb-xl" / "1.0.3"


def load_split_with_demographics(split_name):
    """
    Load y labels + ecg_ids from the processed .npz, then pull age/sex
    back in from the original metadata CSV by aligning on ecg_id.
    """
    data = np.load(PROCESSED_DIR / f"{split_name}.npz")
    y = data["y"]
    ecg_ids = data["ecg_ids"]

    meta = pd.read_csv(DATA_ROOT / "ptbxl_database.csv", index_col="ecg_id")
    demo = meta.loc[ecg_ids, ["age", "sex"]].copy()

    # fill missing age values with median age value so that we dont need to skip and data
    demo["age"] = demo["age"].fillna(demo["age"].median())

    X_demo = demo[["age", "sex"]].values.astype(np.float32)
    return X_demo, y


def main():
    X_train, y_train = load_split_with_demographics("train")
    X_val, y_val = load_split_with_demographics("val")
    X_test, y_test = load_split_with_demographics("test")

    print(f"Age-only feature matrix shape: {X_train.shape} (just age + sex, 2 columns)")

    report_lines = ["AGE-ONLY SANITY CHECK -- age + sex as the ONLY features"]
    report_lines.append("Compare these AUROC numbers to the full 63-feature baseline.")
    report_lines.append("=" * 70)
    report_lines.append(f"{'Class':6s} | {'Age-only AUROC':16s} | {'Full baseline AUROC':20s} | Gap")
    report_lines.append("-" * 70)

    #These are values AUROC values from the 63 parameter model, no point of re-runing the whole script so we hardcoded it
    full_baseline_auroc = {
        "NORM": 0.849, "MI": 0.791, "STTC": 0.794, "CD": 0.820, "HYP": 0.721,
    }

    for i, cls_name in enumerate(SUPERCLASSES):
        model = LogisticRegression(max_iter=1000)#logistic regression model that is based on weighted sum
        model.fit(X_train, y_train[:, i])#readjusts the weights with training data

        test_prob = model.predict_proba(X_test)[:, 1]
        age_only_auroc = roc_auc_score(y_test[:, i], test_prob)
        age_only_auprc = average_precision_score(y_test[:, i], test_prob)

        full_auroc = full_baseline_auroc[cls_name]
        gap = full_auroc - age_only_auroc

        report_lines.append(
            f"{cls_name:6s} | {age_only_auroc:16.3f} | {full_auroc:20.3f} | {gap:+.3f}"
        )
        report_lines.append(f"       (age-only AUPRC: {age_only_auprc:.3f})")

    report_lines.append("\nInterpretation:")
    report_lines.append("  - A small gap means the 63-feature model isn't adding much beyond")
    report_lines.append("    what age/sex alone already tell you -- the waveform features may")
    report_lines.append("    be substantially proxying for age rather than capturing disease-")
    report_lines.append("    specific signal shape.")
    report_lines.append("  - A large gap means the waveform features are contributing real,")
    report_lines.append("    age-independent information.")

    report_text = "\n".join(report_lines)
    print("\n" + report_text)

    out_path = RESULTS_DIR / "age_only_sanity_check.txt"
    with open(out_path, "w") as f:
        f.write(report_text)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
