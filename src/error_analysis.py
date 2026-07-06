#!/usr/bin/env python3

"""
================================================================================
Author:         yxshag
Created:        24-06-2026
GitHub:         https://github.com/yxshag

Project:        heart-byte
File:           error_analysis.py
Description:    For each diagnostic class, this script:
                    1. Builds a confusion matrix at a chosen probability threshold
                    2. Shows the model's most confident WRONG predictions (both false
                        positives and false negatives) -- the cases most worth looking at
                    3. Breaks down performance by age and sex, to check the model isn't
                        silently much worse for some subgroup
                    4. Saves everything to results/ as text + a plot

                Reuses extract_features() and SUPERCLASSES from train_baseline.py so
                the feature definitions stay in exactly one place.
================================================================================
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier

from train_baseline import (
    extract_features, load_split, SUPERCLASSES, PROCESSED_DIR, PROJECT_ROOT
)

#Paths and Constants-DO NOT CHANGE

DATA_ROOT = PROJECT_ROOT / "data" / "physionet.org" / "files" / "ptb-xl" / "1.0.3"
RESULTS_DIR = PROJECT_ROOT / "results"


#This can be changed as per user preference
THRESHOLD = 0.5  # probability cutoff for calling a prediction "positive"
N_WORST_TO_SHOW = 5


def load_test_metadata(test_ecg_ids):
    """loads the age/sex related info as that was ommited in the npz files"""
    meta_path = DATA_ROOT / "ptbxl_database.csv"
    df = pd.read_csv(meta_path, index_col="ecg_id")
    return df.loc[test_ecg_ids, ["age", "sex"]]


def train_models(feat_train, y_train, feat_val, y_val):
    models = {}
    for i, cls_name in enumerate(SUPERCLASSES):
        model = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            eval_metric="auc", early_stopping_rounds=20,
        )
        model.fit(
            feat_train, y_train[:, i],
            eval_set=[(feat_val, y_val[:, i])],
            verbose=False,
        )
        models[cls_name] = model
    return models


def analyze_class(cls_name, y_true, y_prob, ecg_ids, age_sex_df, report_lines):
    """
    Perform a detailed error analysis for a single diagnostic superclass.

    This function goes beyond reporting a single evaluation metric by
    examining *where* the classifier succeeds and fails. Given the ground
    truth labels and predicted probabilities for one superclass, it first
    converts probabilities into binary predictions using the global
    probability threshold. It then computes the confusion matrix along with
    precision and recall, providing an overall summary of the model's
    performance for that class.

    To help identify the most informative mistakes, the function ranks and
    reports the model's most confident false positives (cases where the model
    predicted the disease with high confidence even though it was absent) and
    the most confident false negatives (cases where the disease was present
    but the model assigned a very low probability). These are often the most
    valuable ECGs for manual inspection, as they may reveal systematic model
    weaknesses, ambiguous recordings, or potential label issues.

    Finally, the function evaluates recall across demographic subgroups by
    sex and predefined age ranges. This helps determine whether the model's
    sensitivity differs significantly across patient populations and can
    highlight potential biases or performance disparities that would not be
    apparent from overall metrics alone.

    Parameters
    ----------
    cls_name : str
        Name of the diagnostic superclass currently being analyzed.

    y_true : ndarray
        Binary ground-truth labels (0 or 1) for the test samples.

    y_prob : ndarray
        Predicted probabilities output by the classifier for the positive
        class.

    ecg_ids : ndarray
        PTB-XL ECG identifiers corresponding to each test sample. Used to
        identify the recordings associated with important errors.

    age_sex_df : pandas.DataFrame
        DataFrame indexed by ECG ID containing patient age and sex metadata
        for demographic analysis.

    report_lines : list[str]
        List used to accumulate lines of the final text report. This function
        appends its findings directly to the list rather than printing them.

    Returns
    -------
    precision : float
        Precision of the classifier for this diagnostic superclass.

    recall : float
        Recall (sensitivity) of the classifier for this diagnostic
        superclass.
    """
    y_pred = (y_prob >= THRESHOLD).astype(int) #Converts the probability to prediction

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    report_lines.append(f"\n{'=' * 60}")
    report_lines.append(f"Class: {cls_name}  (threshold={THRESHOLD})")
    report_lines.append(f"{'=' * 60}")
    report_lines.append(f"Confusion matrix:")
    report_lines.append(f"                 Predicted Negative   Predicted Positive")
    report_lines.append(f"Actual Negative  {tn:>18d}   {fp:>18d}")
    report_lines.append(f"Actual Positive  {fn:>18d}   {tp:>18d}")

    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    report_lines.append(f"Precision: {precision:.3f}  |  Recall: {recall:.3f}")

    # --- Most confident false positives: model was SURE it was present, but wasn't ---
    fp_mask = (y_pred == 1) & (y_true == 0)
    fp_indices = np.where(fp_mask)[0]
    #sort the FP indices in decreasing order and show only N worst ones
    fp_sorted = fp_indices[np.argsort(-y_prob[fp_indices])][:N_WORST_TO_SHOW]

    report_lines.append(f"\nTop {len(fp_sorted)} most confident FALSE POSITIVES "
                         f"(model said '{cls_name}' but it wasn't):")
    for idx in fp_sorted:
        ecg_id = ecg_ids[idx]
        age, sex = age_sex_df.loc[ecg_id, ["age", "sex"]]
        report_lines.append(
            f"  ecg_id={ecg_id}  predicted_prob={y_prob[idx]:.3f}  "
            f"age={age}  sex={'M' if sex == 0 else 'F'}"
        )

    # --- Most confident false negatives: model was SURE it was absent, but wasn't ---
    fn_mask = (y_pred == 0) & (y_true == 1)
    fn_indices = np.where(fn_mask)[0]
    fn_sorted = fn_indices[np.argsort(y_prob[fn_indices])][:N_WORST_TO_SHOW]

    report_lines.append(f"\nTop {len(fn_sorted)} most confident FALSE NEGATIVES "
                         f"(model missed '{cls_name}' entirely):")
    for idx in fn_sorted:
        ecg_id = ecg_ids[idx]
        age, sex = age_sex_df.loc[ecg_id, ["age", "sex"]]
        report_lines.append(
            f"  ecg_id={ecg_id}  predicted_prob={y_prob[idx]:.3f}  "
            f"age={age}  sex={'M' if sex == 0 else 'F'}"
        )

    # recall by sex
    report_lines.append(f"\nRecall by sex (of actual positives, how many were caught):")
    for sex_val, sex_label in [(0, "Male"), (1, "Female")]:
        sex_mask = (age_sex_df["sex"].values == sex_val) & (y_true == 1)
        if sex_mask.sum() > 0:
            sex_recall = y_pred[sex_mask].mean()
            report_lines.append(f"  {sex_label}: {sex_recall:.3f}  (n={sex_mask.sum()})")
        else:
            report_lines.append(f"  {sex_label}: no positive cases in test set")

    # recall by age
    report_lines.append(f"\nRecall by age group:")
    age_bins = [0, 40, 60, 80, 120]
    age_labels = ["<40", "40-60", "60-80", "80+"]
    age_groups = pd.cut(age_sex_df["age"].values, bins=age_bins, labels=age_labels)
    for label in age_labels:
        bucket_mask = (age_groups == label) & (y_true == 1)
        if bucket_mask.sum() > 0:
            bucket_recall = y_pred[bucket_mask].mean()
            report_lines.append(f"  {label}: {bucket_recall:.3f}  (n={bucket_mask.sum()})")
        else:
            report_lines.append(f"  {label}: no positive cases in test set")

    return precision, recall


def plot_confusion_matrices(all_cms, out_path):
    cmap = plt.get_cmap("YlGnBu")  # light yellow (low) -> dark blue (high), good contrast either way
    fig, axes = plt.subplots(1, len(SUPERCLASSES), figsize=(4 * len(SUPERCLASSES), 4))
    for ax, cls_name in zip(axes, SUPERCLASSES):
        cm = all_cms[cls_name]
        im = ax.imshow(cm, cmap=cmap)
        ax.set_title(cls_name)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Neg", "Pos"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["Neg", "Pos"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")

        # Pick black or white based on cell color
        vmax = cm.max()
        for i in range(2):
            for j in range(2):
                cell_value = cm[i, j]
                text_color = "white" if cell_value > vmax * 0.6 else "black"
                ax.text(j, i, str(cell_value), ha="center", va="center",
                         fontsize=13, color=text_color, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    X_train, y_train = load_split("train")
    X_val, y_val = load_split("val")
    X_test, y_test = load_split("test")

    test_data = np.load(PROCESSED_DIR / "test.npz")
    test_ecg_ids = test_data["ecg_ids"]

    feat_train = extract_features(X_train, "train")
    feat_val = extract_features(X_val, "val")
    feat_test = extract_features(X_test, "test")

    age_sex_df = load_test_metadata(test_ecg_ids)

    models = train_models(feat_train, y_train, feat_val, y_val)

    report_lines = ["ERROR ANALYSIS -- Baseline model"]
    all_cms = {}

    for i, cls_name in enumerate(SUPERCLASSES):
        y_prob = models[cls_name].predict_proba(feat_test)[:, 1]
        y_true = y_test[:, i]
        y_pred = (y_prob >= THRESHOLD).astype(int)

        all_cms[cls_name] = confusion_matrix(y_true, y_pred)
        analyze_class(cls_name, y_true, y_prob, test_ecg_ids, age_sex_df, report_lines)

    report_text = "\n".join(report_lines)
    print(report_text)

    out_txt = RESULTS_DIR / "error_analysis.txt"
    with open(out_txt, "w") as f:
        f.write(report_text)
    print(f"\nSaved full report to {out_txt}")

    out_plot = RESULTS_DIR / "confusion_matrices.png"
    plot_confusion_matrices(all_cms, out_plot)
    print(f"Saved confusion matrix plot to {out_plot}")


if __name__ == "__main__":
    main()
