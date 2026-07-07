# ECG Arrhythmia Classifier

A machine learning project for detecting cardiac abnormalities from
12-lead ECG signals, using the [PTB-XL](https://physionet.org/content/ptb-xl/)
dataset. The project classifies each ECG into five diagnostic
superclasses — **NORM** (normal), **MI** (myocardial infarction),
**STTC** (ST/T-wave changes), **CD** (conduction disturbance), and
**HYP** (hypertrophy).

Two approaches are implemented and compared:
1. A hand-crafted-feature baseline (rhythm/statistical features +
   XGBoost)
2. A 1D CNN trained directly on raw ECG waveforms

## Dataset

This project uses the 100Hz version of PTB-XL (~1.8GB). Run the
provided download script to fetch it into `data/`:

```bash
bash download_data.sh
```

## Project structure

```
ecg-classifier/
├── data/
│   ├── raw/                       # PTB-XL dataset (fetched by         download_data.sh)
│   └── processed/                 # preprocessed train/val/test .npz files
├── src/
│   ├── data_utils.py                    # shared data loading helpers
│   ├── explore_data.py                  # data exploration
│   ├── verify_patient_split.py          # patient-level leakage check
│   ├── preprocessing.py                 # filtering, normalization, train/val/test split
│   ├── train_baseline.py                # hand-crafted features + XGBoost
│   ├── age_only_baseline.py             # age-only sanity check
│   ├── error_analysis.py                # baseline error analysis
│   ├── threshold_tuning.py              # baseline threshold tuning
│   ├── train_cnn.py                     # 1D CNN on raw waveforms
│   ├── cnn_inference_utils.py           # shared CNN loading/inference helpers
│   ├── tune_thresholds_cnn.py           # per-class threshold tuning for the CNN
│   ├── calibration_check_cnn.py         # calibration check for the CNN
│   ├── cnn_error_analysis.py            # error analysis for the CNN
│   ├── cnn_subgroup_rates.py            # age/sex subgroup performance analysis
│   ├── generate_xgb_errors.py           # error extraction for the baseline
│   └── cross_model_error_comparison.py  # compares CNN vs. baseline errors
├── download_data.sh
└── README.md
```

## Setup

```bash
git clone https://github.com/yxshag/heart-byte
cd heartbyte
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Reproducing the results

Run the scripts in this order:

```bash
# 0. Download the raw PTB-XL data and create models and results directories
bash download_data.sh
mkdir results
mkdir models

# 1. Explore the data and verify no patient-level leakage in the fold split
python src/explore_data.py
python src/verify_patient_split.py

# 2. Preprocess: filtering, normalization, train/val/test split
python src/preprocessing.py

# 3. Train and evaluate the baseline (hand-crafted features + XGBoost)
python src/train_baseline.py
python src/age_only_baseline.py       # sanity check: waveform features vs. age alone
python src/error_analysis.py
python src/threshold_tuning.py

# 4. Train the CNN on raw waveforms
python src/train_cnn.py

# 5. Tune classification thresholds for the CNN
python src/tune_thresholds_cnn.py

# 6. Check CNN calibration
python src/calibration_check_cnn.py

# 7. Run error and subgroup analysis for the CNN
python src/cnn_error_analysis.py
python src/cnn_subgroup_rates.py

# 8. Compare CNN and baseline errors
python src/generate_xgb_errors.py
python src/cross_model_error_comparison.py
```

Each script prints its results to the terminal and saves any relevant
outputs (thresholds, plots, CSVs) to the results folder.

For the full methodology, results breakdown, and design decisions
behind this project, see the accompanying write-up.
