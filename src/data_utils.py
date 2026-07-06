#!/usr/bin/env python3

"""
================================================================================
Author:         yxshag
Created:        27-06-2026
GitHub:         https://github.com/yxshag

Project:        heart-byte
File:           data_uils.py
Description:    Contains the important functions that are used very regularly.
================================================================================
"""

from pathlib import Path
import ast

import pandas as pd

# ---------------------------------------------------------------------
# Paths (DO NOT CHANGE)
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "physionet.org" / "files" / "ptb-xl" / "1.0.3"


def load_metadata():
    """
    Load the PTB-XL metadata and SCP statements.

    Returns:
        df (pd.DataFrame): PTB-XL metadata.
        scp_df (pd.DataFrame): SCP statements table.
    """

    meta_path = DATA_ROOT / "ptbxl_database.csv"
    scp_path = DATA_ROOT / "scp_statements.csv"

    if not meta_path.exists():
        raise FileNotFoundError(
            f"Could not find {meta_path}\n"
            "Did you run download_data.sh first?"
        )

    df = pd.read_csv(meta_path, index_col="ecg_id")
    df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)

    scp_df = pd.read_csv(scp_path, index_col=0)

    return df, scp_df


def add_diagnostic_superclass(df, scp_df):
    """
    Map each ECG's SCP codes to one or more diagnostic superclasses.

    Superclasses:
        - NORM
        - MI
        - STTC
        - CD
        - HYP

    Args:
        df (pd.DataFrame): PTB-XL metadata.
        scp_df (pd.DataFrame): SCP statements.

    Returns:
        pd.DataFrame: DataFrame with a new column
        'diagnostic_superclass'.
    """

    diag_scp = scp_df[scp_df["diagnostic"] == 1]

    def codes_to_superclasses(scp_codes_dict):
        classes = set()

        for code in scp_codes_dict.keys():
            if code in diag_scp.index:
                superclass = diag_scp.loc[code, "diagnostic_class"]

                if isinstance(superclass, str):
                    classes.add(superclass)

        return list(classes)

    df = df.copy()
    df["diagnostic_superclass"] = df["scp_codes"].apply(
        codes_to_superclasses
    )

    return df


def load_processed_metadata():
    """
    Convenience function that loads the metadata and immediately adds
    diagnostic superclasses.

    Returns:
        pd.DataFrame
    """

    df, scp_df = load_metadata()
    df = add_diagnostic_superclass(df, scp_df)

    return df