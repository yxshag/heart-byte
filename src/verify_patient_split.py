#!/usr/bin/env python3

"""
================================================================================
Author:         yxshag
Created:        20-06-2026
GitHub:         https://github.com/yxshag

Project:        heart-byte
File:           verify_patient_split.py
Description:    Verifies that there is no leakage of patient data among the
                train, validate and testing folds of the PTB-XL's official 
                strat_fold colums.
================================================================================
"""


from pathlib import Path
import pandas as pd

#Paths-DO NOT CHANGE

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "physionet.org" / "files" / "ptb-xl" / "1.0.3"


def load_metadata():
    #returns df
    meta_path = DATA_ROOT / "ptbxl_database.csv"
    if not meta_path.exists():
        raise FileNotFoundError(f"Couldn't find {meta_path}. Did you run download_data.sh?")
    return pd.read_csv(meta_path, index_col="ecg_id")


def check_patient_fold_consistency(df):
    """
    For every patient, collect the set of distinct folds their records
    appear in. If that set ever has more than 1 element, that patient
    is split across folds which shows a leak.
    
    Returns:
        a list of leaked patients if any
    """
    folds_per_patient = df.groupby("patient_id")["strat_fold"].nunique()

    leaked_patients = folds_per_patient[folds_per_patient > 1]

    print(f"Total unique patients: {len(folds_per_patient)}")
    print(f"Patients whose records span more than one fold: {len(leaked_patients)}")

    if len(leaked_patients) == 0:
        print("\n PASS: every patient's records sit entirely within a single fold.")
        print("   Safe to split by fold without patient leakage.")
    else:
        print("\n FAIL: the following patients have records spread across multiple folds:")
        for patient_id, n_folds in leaked_patients.items():
            folds = sorted(df.loc[df["patient_id"] == patient_id, "strat_fold"].unique())
            print(f"   patient_id={patient_id} -> folds {folds}")

    return leaked_patients


def check_train_test_overlap(df, train_folds=range(1, 9), val_fold=9, test_fold=10):
    """
    Double-check the specific split we want to use: no patient_id should
    appear in more than one of {train, val, test}.
    """
    train_patients = set(df.loc[df["strat_fold"].isin(train_folds), "patient_id"])
    val_patients = set(df.loc[df["strat_fold"] == val_fold, "patient_id"])
    test_patients = set(df.loc[df["strat_fold"] == test_fold, "patient_id"])

    train_val_overlap = train_patients & val_patients
    train_test_overlap = train_patients & test_patients
    val_test_overlap = val_patients & test_patients

    print(f"\nTrain patients (folds {list(train_folds)}): {len(train_patients)}")
    print(f"Val patients (fold {val_fold}): {len(val_patients)}")
    print(f"Test patients (fold {test_fold}): {len(test_patients)}")

    print(f"\nTrain/Val patient overlap:  {len(train_val_overlap)}")
    print(f"Train/Test patient overlap: {len(train_test_overlap)}")
    print(f"Val/Test patient overlap:   {len(val_test_overlap)}")

    if not (train_val_overlap or train_test_overlap or val_test_overlap):
        print("\nPASS: the train/val/test split (folds 1-8 / 9 / 10) has zero patient overlap.")
    else:
        print("\nFAIL: overlap detected in the intended train/val/test split. Investigate above.")


if __name__ == "__main__":
    df = load_metadata()
    check_patient_fold_consistency(df)
    check_train_test_overlap(df)
