#!/usr/bin/env python3

"""
=======================================================================================================
Author:         yxshag
Created:        05-07-2026
GitHub:         https://github.com/yxshag

Project:        heart-byte
File:           cross_model_error_comparison.py
Description:    Cross model error comparison: Compares the errors made by xgboost and the cnn models.
                Checks if both the models are making completely different mistakes which is a good
                enough reason to believe that either models are ineffective in particular cases or 
                if the models are making similar mistakes which would lead us to believe that there
                is no need for an ensemble of both the models.
=======================================================================================================
"""
import pandas as pd

# --------------------Paths-DO NOT CHANGE--------------------------------------
CNN_ERRORS_PATH = "results/cnn_test_errors.csv"
XGB_ERRORS_PATH = "results/xgb_test_errors.csv" 
CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
# -----------------------------------------------------------------------------


def main():
    cnn_errors = pd.read_csv(CNN_ERRORS_PATH)
    xgb_errors = pd.read_csv(XGB_ERRORS_PATH)

    print("=" * 70)
    print("Cross-model error overlap, per class")
    print("=" * 70)

    summary_rows = []

    for cls in CLASSES:
        cnn_ids = set(cnn_errors.loc[cnn_errors["class"] == cls, "ecg_id"])
        xgb_ids = set(xgb_errors.loc[xgb_errors["class"] == cls, "ecg_id"])

        both = cnn_ids & xgb_ids
        cnn_only = cnn_ids - xgb_ids
        xgb_only = xgb_ids - cnn_ids

        total_error_cases = cnn_ids | xgb_ids
        overlap_pct = len(both) / len(cnn_ids) * 100 if cnn_ids else 0.0

        print(f"\n{cls}")
        print(f"  CNN errors:       {len(cnn_ids)}")
        print(f"  XGBoost errors:   {len(xgb_ids)}")
        print(f"  Both models wrong:      {len(both)}")
        print(f"  CNN wrong only:         {len(cnn_only)}")
        print(f"  XGBoost wrong only:     {len(xgb_only)}")
        print(f"  Overlap percent of cnn(as cnn has lesser errors):     {overlap_pct:.1f}%")

        summary_rows.append({
            "class": cls,
            "cnn_errors": len(cnn_ids),
            "xgb_errors": len(xgb_ids),
            "both_wrong": len(both),
            "cnn_only_wrong": len(cnn_only),
            "xgb_only_wrong": len(xgb_only),
            "overlap_pct_of_union": round(overlap_pct, 1),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv("results/cross_model_error_overlap.csv", index=False)
    print("\nSaved summary to cross_model_error_overlap.csv")


if __name__ == "__main__":
    main()