# ECG Arrhythmia Classification on PTB-XL: Technical Documentation

**A comparative study of hand-crafted-feature and deep-learning approaches
to multi-label ECG diagnostic classification**

---

## Abstract

This project builds and evaluates two approaches to multi-label ECG
diagnostic classification on the PTB-XL dataset: (1) a hand-crafted-feature
baseline using rhythm and statistical features with per-class XGBoost
classifiers, and (2) a 1D convolutional neural network (CNN) trained
directly on raw waveform data. Both models predict the presence of five
diagnostic superclasses — Normal (NORM), Myocardial Infarction (MI),
ST/T-wave Changes (STTC), Conduction Disturbance (CD), and Hypertrophy
(HYP) — from 10-second, 12-lead ECG recordings. The pipeline includes
patient-level leakage verification, per-class threshold tuning, error
analysis, demographic subgroup analysis, model calibration assessment,
and a cross-model error comparison to evaluate the viability of
ensembling. The CNN outperforms the baseline on every diagnostic class
while remaining well-calibrated, with the most significant caveats being
persistent difficulty on the Hypertrophy class and an age-dependent drop
in precision for Normal-class predictions.

---

## 1. Introduction

### 1.1 Problem Statement

Given a 10-second, 12-lead ECG recording, the goal is to predict which of
five diagnostic categories are present. This is a **multi-label**
problem — a single recording may exhibit more than one condition
simultaneously — which rules out a simple softmax classifier in favor of
five independent binary decisions (implemented as five sigmoid outputs
trained with `BCEWithLogitsLoss`, or as five separate binary classifiers
in the baseline).

### 1.2 Diagnostic Classes

| Code | Meaning |
|---|---|
| NORM | Normal rhythm |
| MI | Myocardial infarction (heart attack) |
| STTC | ST/T-wave changes (ischemia-related) |
| CD | Conduction disturbance |
| HYP | Hypertrophy (enlarged heart muscle) |

---

## 2. Dataset

**Source:** [PTB-XL](https://physionet.org/content/ptb-xl/1.0.3/), PhysioNet
**License:** Creative Commons Attribution 4.0 (CC-BY) — open access, no
credentialing required.

**Composition:** 21,799 ECG records from 18,869 unique patients. Each
record is 12-lead, 10 seconds, sampled at either 100Hz or 500Hz. This
project uses **the 100Hz version only** — sufficient signal resolution
for superclass-level diagnosis at a fraction of the storage and compute
cost of the 500Hz version.

**Key data files:**

| File | Contents |
|---|---|
| `ptbxl_database.csv` | One row per ECG: `patient_id`, `age`, `sex`, `scp_codes`, `strat_fold`, `filename_lr` |
| `scp_statements.csv` | Maps raw SCP diagnosis codes to the five readable superclasses |
| `records100/` | The waveform files actually used (`.dat` + `.hea` pairs) |
| `records500/` | Not downloaded — unnecessary for this project's resolution needs |

PTB-XL ships with an official 10-fold stratified split (`strat_fold`),
pre-balanced by diagnostic prevalence. This project uses folds 1–8 for
training, fold 9 for validation, and fold 10 for testing.

---

## 3. Repository Structure

```
ecg-classifier/
├── README.md
├── DOCUMENTATION.md                # this file
├── download_data.sh                # one-time PTB-XL download
├── requirements.txt
├── .gitignore                      # excludes data/, venv/, models/, results/
│
├── data/
│   ├── physionet.org/files/ptb-xl/1.0.3/   # raw downloaded data
│   │   ├── ptbxl_database.csv
│   │   ├── scp_statements.csv
│   │   └── records100/
│   └── processed/                  # output of preprocessing.py
│       ├── train.npz               # X=(17398, 1000, 12), y=(17398, 5)
│       ├── val.npz                 # X=(2177, 1000, 12),  y=(2177, 5)
│       └── test.npz                # X=(2158, 1000, 12),  y=(2158, 5)
│
├── src/
│   ├── explore_data.py
│   ├── verify_patient_split.py
│   ├── preprocessing.py
│   ├── train_baseline.py
│   ├── error_analysis.py
│   ├── threshold_tuning.py
│   ├── age_only_baseline.py
│   ├── train_cnn.py
│   ├── cnn_inference_utils.py
│   ├── tune_thresholds_cnn.py
│   ├── calibration_check_cnn.py
│   ├── cnn_error_analysis.py
│   ├── cnn_subgroup_rates.py
│   ├── generate_xgb_errors.py
│   └── cross_model_error_comparison.py
│
├── models/                         # trained model checkpoints (not versioned)
└── results/                        # generated metrics, plots, CSVs (not versioned)
```

---

## 4. Environment

**Core dependencies** (`requirements.txt`):

- `wfdb>=4.1.0` — reading PhysioNet `.dat`/`.hea` waveform files
- `numpy`, `pandas`, `scipy` — data handling and signal filtering
- `scikit-learn` — metrics, logistic regression, calibration curves
- `xgboost` — baseline classifiers
- `matplotlib`, `seaborn` — plotting
- `neurokit2>=0.2.9` — ECG peak detection and HRV feature extraction
- `torch>=2.0.0` — CNN (requires a CUDA build for GPU training)
- `tqdm`, `jupyter`

**GPU setup note:** PyTorch installed via plain `pip install torch` often
resolves to a CPU-only wheel. If `torch.cuda.is_available()` returns
`False` on a machine with an NVIDIA GPU, this is almost always the cause.
Fix by checking the driver's supported CUDA version via `nvidia-smi`, then
reinstalling with the matching wheel, e.g.:
```bash
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu121
```
(`cu121` for CUDA 12.1, `cu124` for 12.4, `cu118` for 11.8, etc.)

---

## 5. Methodology

### 5.1 Data Exploration — `explore_data.py`

**Purpose:** Establish a factual baseline understanding of the dataset —
class balance, patient distribution, and fold structure — before any
modeling decisions are made.

**What it does:** Loads `ptbxl_database.csv`, maps raw SCP diagnostic
codes to the five superclasses via `scp_statements.csv`, computes summary
statistics, and saves sample waveform plots for visual inspection.

**Findings:**
- 21,799 total records from 18,869 unique patients (most patients
  contribute a single ECG; the maximum is 10 per patient).
- Label prevalence (multi-label, so counts sum to more than 21,799):
  **NORM: 9,514 · MI: 5,469 · STTC: 5,235 · CD: 4,898 · HYP: 2,649**.
- 10 folds, each containing approximately 2,175–2,200 records, already
  stratified by the dataset creators.

**Interpretation:** HYP is the rarest class by a meaningful margin
(2,649 records vs. 9,514 for NORM), foreshadowing the class-imbalance
issues that surface repeatedly in later steps.

---

### 5.2 Patient-Level Leakage Verification — `verify_patient_split.py`

**Purpose:** Guarantee that no patient's records appear in more than one
of {train, validation, test}. Since some patients contribute multiple
ECGs, splitting naively by record (row) rather than by patient would let
a model implicitly memorize patient-specific characteristics in training
and then "recognize" the same patient at test time — inflating reported
performance in a way that would not generalize to genuinely unseen
patients.

**What it does:** Groups all records by `patient_id`, confirms every
patient's records fall within a single fold, then confirms no
`patient_id` spans more than one of {train = folds 1–8, val = fold 9,
test = fold 10}.

**Result:** **PASS** — zero patients are split across folds. The
dataset's official `strat_fold` assignment is safe to use directly.

---

### 5.3 Preprocessing — `preprocessing.py`

**Purpose:** Convert raw waveform files into a clean, filtered,
normalized, model-ready tensor format, and materialize the train/val/test
split decided in Section 5.2.

**What it does:**
1. **Bandpass filtering:** 0.5Hz–40Hz Butterworth filter, removing
   baseline wander (very low frequency drift) and high-frequency noise
   while preserving the diagnostically relevant waveform morphology.
2. **Normalization:** Per-record, per-lead z-score normalization
   (mean 0, standard deviation 1), so amplitude differences between
   patients/devices don't dominate over shape differences.
3. **Label encoding:** Multi-hot encoding — each record becomes a
   5-element float vector (e.g., `[1, 0, 0, 1, 0]` for a record positive
   for both NORM... *(note: NORM would not co-occur with disease labels
   in practice; illustrative example)* — actual encoding reflects
   whichever combination of the five superclasses is present.
4. **Splitting:** Folds 1–8 → train, fold 9 → validation, fold 10 → test.
5. **Persistence:** Saves compressed `train.npz`, `val.npz`, `test.npz`
   to `data/processed/`.

**Output shapes:**

| Split | X shape | y shape |
|---|---|---|
| Train | (17,398, 1000, 12) | (17,398, 5) |
| Val | (2,177, 1000, 12) | (2,177, 5) |
| Test | (2,158, 1000, 12) | (2,158, 5) |

(1000 = 10 seconds × 100Hz; 12 = number of leads; 5 = number of
diagnostic classes.)

A small number of records with no diagnostic superclass assigned at all
were dropped, since they carry no usable multi-label training signal for
this task.

---

### 5.4 Baseline Model — `train_baseline.py`

**Purpose:** Establish a fast, interpretable, hand-crafted-feature
reference point before investing in a more complex deep-learning
pipeline. A baseline serves three functions here: (1) it sanity-checks
the entire pipeline — data, labels, and evaluation — on something simple
before adding CNN complexity; (2) it gives an honest point of comparison,
so that if a later CNN only marginally outperforms it, that would signal
the "easy" signal was already captured; (3) it is fast to train and easy
to debug.

**Feature engineering** (63 features per record):
- **Rhythm features from Lead II** (via `neurokit2`), chosen because Lead
  II is the standard lead most peak-detection algorithms — including
  `neurokit2`'s defaults — are tuned against: heart rate, heart rate
  variability (SDNN — standard deviation of RR intervals), and detected
  beat count. Records where peak detection fails are filled with the
  column median rather than dropped, so one noisy record doesn't break
  the pipeline.
- **Per-lead statistics across all 12 leads:** mean, standard deviation,
  minimum, maximum, and peak-to-peak range (5 statistics × 12 leads = 60
  features).

**Model:** Five independent binary XGBoost classifiers, one per
diagnostic class (`n_estimators=300`, `max_depth=4`, `learning_rate=0.05`,
early stopping on validation AUC).

**Results (test set):**

| Class | AUROC | AUPRC | Test Prevalence |
|---|---|---|---|
| NORM | 0.849 | 0.800 | 44.6% |
| MI | 0.791 | 0.575 | 25.5% |
| STTC | 0.794 | 0.542 | 24.1% |
| CD | 0.820 | 0.703 | 23.0% |
| HYP | 0.721 | 0.286 | 12.1% |

**Interpretation:** AUROC is threshold-independent and treats all
classes somewhat comparably, but is known to be an optimistic metric
under class imbalance. AUPRC, which reflects the actual precision/recall
tradeoff a user would experience at realistic operating points, tells a
more sobering story for the imbalanced classes: HYP's AUPRC of 0.286
(against 12.1% prevalence) versus NORM's 0.800 (against 44.6% prevalence)
indicates that even though HYP's AUROC (0.721) looks only moderately
worse than NORM's (0.849), its practical precision/recall behavior is
substantially weaker. HYP is established here, at the very first modeling
step, as the hardest class — a pattern that persists through every
subsequent stage of this project.

Saved to: `results/baseline_results.txt`

---

### 5.5 Baseline Error Analysis — `error_analysis.py`

**Purpose:** Move beyond aggregate metrics to understand *how* and *for
whom* the baseline model fails — which is necessary both for debugging
and for surfacing any fairness concerns before further investment in the
pipeline.

**What it does:** Re-trains the baseline, then for each class computes: a
confusion matrix at the default threshold of 0.5; the five most confident
false positives and false negatives (with associated patient age/sex);
recall broken down by sex; and recall broken down by age group
(`<40`, `40–60`, `60–80`, `80+`).

**Findings:**
1. **HYP is functionally non-operational at threshold 0.5:** recall of
   only 2.3%. At a 50% decision threshold, the model almost never
   predicts positive for a class with only 12% prevalence — this is a
   symptom of class imbalance interacting with an unadjusted default
   threshold, not necessarily an absence of learned signal (this
   hypothesis is directly tested and confirmed in Section 5.6).
2. **Recall rises steeply with patient age for every disease class**
   (e.g., MI: 0% in patients under 40 → 42% in patients 80+; CD: 10% →
   56%). NORM shows the mirror-image pattern (89% under 40 → 32% at
   80+). This raised an important methodological question at this stage
   of the project: **is the model tracking patient age as a proxy,
   rather than learning genuine disease-specific waveform shape?** This
   question motivated the age-only sanity check in Section 5.7.
3. **No meaningful sex-based disparity** was found — recall differed by
   only a few percentage points between male and female patients across
   every class.
4. **MI recall of 30.7%** at the default threshold was flagged explicitly
   as clinically dangerous if this model were ever used for screening
   purposes — nearly 7 in 10 true MI cases would be missed.

Saved to: `results/error_analysis.txt`, `results/confusion_matrices.png`

---

### 5.6 Baseline Threshold Tuning — `threshold_tuning.py`

**Purpose:** Test the hypothesis raised in Section 5.5 — that the poor
recall at threshold 0.5 was a threshold-calibration artifact rather than
a genuine absence of model signal — and establish better operating
points for each class.

**Method:** For each class, two threshold-selection methods are computed
using the **validation set only** (never the test set, to avoid
overfitting the threshold choice to test data): (1) **F1-optimal** —
sweep candidate thresholds from 0.01 to 0.99 and select the one
maximizing F1 on validation; (2) **Youden's J statistic** — maximize
(sensitivity + specificity − 1) along the ROC curve. Chosen thresholds
are then evaluated on the held-out test set.

**Results (F1-optimal thresholds, evaluated on test):**

| Class | Default (t=0.5) F1 | F1-Optimal Threshold | F1-Optimal F1 | Recall Improvement |
|---|---|---|---|---|
| NORM | 0.753 | 0.44 | 0.753 | 79% → 84% |
| MI | 0.422 | 0.22 | 0.566 | 31% → 77% |
| STTC | 0.377 | 0.26 | 0.549 | 27% → 70% |
| CD | 0.595 | 0.42 | 0.606 | 47% → 51% |
| HYP | 0.044 | 0.15 | 0.334 | 2% → 50% |

**Interpretation:** This directly confirms the hypothesis from Section
5.5. HYP's F1 rises from 0.044 to 0.334, and its recall from 2% to 50%,
purely by moving the decision threshold — the model did have learned
signal for HYP all along; the default 0.5 threshold was simply badly
miscalibrated for a class with 12% prevalence. The same pattern, though
less extreme, holds for MI. This comes at a cost of lower precision (more
false positives), which is judged an acceptable tradeoff for this kind of
diagnostic screening context, where a missed disease case is typically
costlier than a false alarm that a human clinician subsequently reviews
and dismisses.

Saved to: `results/threshold_tuning.txt`

---

### 5.7 Age-Only Sanity Check — `age_only_baseline.py`

**Purpose:** Directly test the concern raised in Section 5.5 — that the
model might be substantially proxying patient age rather than learning
genuine ECG waveform signal. This is a standard rigor check in clinical
machine learning: if a trivial demographic-only model performs close to
the full model, the full model's apparent skill may be illusory.

**Method:** Train a plain logistic regression using **only age and sex**
(2 features total, no waveform-derived information whatsoever) and
compare its AUROC against the full 63-feature baseline.

**Results:**

| Class | Age-Only AUROC | Full Baseline AUROC | Gap |
|---|---|---|---|
| NORM | 0.758 | 0.849 | +0.091 |
| MI | 0.631 | 0.791 | +0.160 |
| STTC | 0.689 | 0.794 | +0.105 |
| CD | 0.601 | 0.820 | +0.219 |
| HYP | 0.584 | 0.721 | +0.137 |

**Interpretation:** This check **clears the concern raised in Section
5.5.** Every class shows a substantial, age-independent performance gap
(+0.09 to +0.22 AUROC) between the age-only model and the full model —
the waveform features are contributing real, non-demographic diagnostic
signal in every case. The age-recall pattern observed in Section 5.5 is
therefore best explained as reflecting a genuine underlying fact (older
patients truly do have higher rates of cardiac disease) rather than the
model taking a demographic shortcut. CD shows the largest gap (+0.219),
consistent with conduction disturbances having particularly distinctive,
learnable waveform signatures. This result also provided direct
motivation for the CNN: if a fixed 63-feature hand-crafted set could
already add +0.22 AUROC over demographics alone, a model with access to
full raw-waveform shape — especially for shape-dependent classes like MI
and STTC — was expected to add meaningfully more.

Saved to: `results/age_only_sanity_check.txt`

---

### 5.8 CNN Architecture and Training — `train_cnn.py`

**Purpose:** Test whether a model with direct access to raw waveform
shape — rather than a fixed, hand-engineered feature summary — can
extract additional diagnostic signal beyond what the baseline captures.

**Architecture:**
- **Input:** `(batch, 12, 1000)` — 12 leads as channels, 1000 timesteps
  (10 seconds at 100Hz)
- **4 convolutional blocks**, each `Conv1d → BatchNorm1d → ReLU →
  MaxPool1d(2)`:
  - Channel progression: 12 → 32 → 64 → 128 → 128
  - Temporal dimension progression: 1000 → 500 → 250 → 125 → 62
- **Global average pooling**, flattening to a 128-dimensional vector
- **Classifier head:** `Linear(128, 64) → ReLU → Dropout(0.3) →
  Linear(64, 5)`
- **Loss:** `BCEWithLogitsLoss` — the correct choice for multi-label
  classification, applying sigmoid internally with better numerical
  stability than a separate sigmoid-then-BCE-loss combination.
- **Optimizer:** Adam, learning rate `1e-3`
- **Early stopping:** patience of 8 epochs, monitored on validation mean
  AUROC across all five classes
- **Checkpointing:** best model saved to `models/cnn_best.pt`

**Engineering note — GPU acceleration:** Initial CUDA detection failed
(`torch.cuda.is_available()` returned `False`) due to a CPU-only PyTorch
wheel having been installed by default. Resolved by reinstalling PyTorch
with the CUDA-matched wheel corresponding to the machine's driver version
(see Section 4). Once resolved, training correctly used the GPU
(`Using device: cuda` confirmed in later runs).

**Results (test set):**

Best validation mean AUROC during training: **0.912**

| Class | AUROC | AUPRC | Test Prevalence |
|---|---|---|---|
| NORM | 0.942 | 0.921 | 44.6% |
| MI | 0.918 | 0.816 | 25.5% |
| STTC | 0.936 | 0.833 | 24.1% |
| CD | 0.922 | 0.840 | 23.0% |
| HYP | 0.828 | 0.460 | 12.1% |

Saved to: `results/cnn_results.txt`

---

### 5.9 CNN vs. Baseline Comparison

| Class | XGBoost AUROC | CNN AUROC | Δ AUROC |
|---|---|---|---|
| NORM | 0.849 | 0.942 | +0.093 |
| MI | 0.791 | 0.918 | +0.127 |
| STTC | 0.794 | 0.936 | +0.142 |
| CD | 0.820 | 0.922 | +0.102 |
| HYP | 0.721 | 0.828 | +0.107 |

**Interpretation:** The CNN outperforms the hand-crafted-feature baseline
on **every** class, by a substantial and consistent margin (+0.09 to
+0.14 AUROC) — this is a real capability gain, not measurement noise.
STTC shows the largest improvement, consistent with the expectation from
Section 5.7: ST/T-wave morphology is subtle and difficult to hand-engineer
into fixed statistical features but is directly learnable from raw
waveform shape. **HYP remains the weakest class for both models** —
its baseline AUROC (0.721) and CNN AUROC (0.828) are both the lowest
among the five classes, and its AUPRC (0.460 for the CNN, against 12.1%
prevalence) signals a real precision limitation at practical operating
thresholds, echoing the AUPRC-based caveat first identified for the
baseline in Section 5.4.

---

### 5.10 CNN Inference Utilities — `cnn_inference_utils.py`

**Purpose:** Provide a single, shared, reusable interface for loading the
trained CNN checkpoint and running batched inference, so that every
downstream analysis script (threshold tuning, calibration, error
analysis, subgroup analysis) evaluates the model identically rather than
each reimplementing — and potentially subtly diverging in — its own
inference logic.

**What it does:** `get_device()` selects CUDA or CPU; `load_split()`
loads a processed `.npz` split; `load_model()` instantiates the CNN
architecture and loads trained weights from checkpoint; `get_predictions()`
runs batched inference and returns sigmoid probabilities.

**Engineering notes — bugs identified and fixed during integration:**

1. **Model constructor argument mismatch.** The model class `ECGCNN` is
   defined as `__init__(self, n_leads=12, n_classes=5)`. An initial
   version of the loading code incorrectly called it with a `num_classes`
   keyword, raising a `TypeError`. Fixed by matching the call to the
   actual signature (`n_classes=...`).
2. **Channels-last vs. channels-first tensor layout mismatch.** The
   processed `.npz` data is stored as `(batch, sequence_length,
   n_leads)` — e.g., `(256, 1000, 12)` — but PyTorch's `Conv1d` expects
   `(batch, n_leads, sequence_length)` — e.g., `(256, 12, 1000)`. This
   raised a `RuntimeError` referencing a channel-count mismatch. Fixed by
   inserting a conditional `permute(0, 2, 1)` before the forward pass
   whenever the second dimension does not already equal the expected
   number of leads (12).

---

### 5.11 CNN Threshold Tuning — `tune_thresholds_cnn.py`

**Purpose:** Determine F1-optimal decision thresholds for the CNN
independently from the baseline's thresholds (Section 5.6), since the
two models produce differently-distributed probability outputs and
reusing one model's thresholds for another would be miscalibrated.

**Method:** Identical discipline to Section 5.6 — F1-optimal threshold
search performed on the validation set via `sklearn.precision_recall_curve`,
with final precision/recall/F1/confusion-matrix reporting on the held-out
test set.

**Results:**

| Class | Baseline Threshold (§5.6) | CNN Threshold | Test Precision | Test Recall | Test F1 |
|---|---|---|---|---|---|
| NORM | 0.44 | 0.322 | 0.795 | 0.922 | 0.854 |
| MI | 0.22 | 0.420 | 0.690 | 0.776 | 0.731 |
| STTC | 0.26 | 0.510 | 0.760 | 0.741 | 0.750 |
| CD | 0.42 | 0.494 | 0.764 | 0.724 | 0.743 |
| HYP | 0.15 | 0.275 | 0.438 | 0.489 | 0.462 |

**Interpretation:** The CNN's optimal thresholds diverge meaningfully
from the baseline's — most notably MI (0.22 → 0.42) and STTC (0.26 →
0.51) — confirming that threshold values are not transferable between
models and must be tuned per-model. NORM, MI, STTC, and CD all reach a
healthy F1 range (0.73–0.85). **HYP remains the clear outlier**:
precision of 0.438 means roughly half of the model's HYP predictions are
false alarms, and recall of 0.489 means roughly half of true HYP cases
are still missed, even at the F1-optimal operating point. Because this
weakness persists across the entire precision-recall curve rather than
being a single bad threshold choice, it is not resolvable by further
threshold adjustment — it reflects the underlying separability limit
discussed further in Section 5.12.

Saved to: `cnn_thresholds.json`

---

### 5.12 CNN Calibration Check — `calibration_check_cnn.py`

**Purpose:** Determine whether the CNN's output probabilities are
*trustworthy* as confidence estimates — i.e., whether a predicted
probability of 0.7 actually corresponds to a roughly 70% true positive
rate — independent of the model's raw discriminative performance (AUROC/
AUPRC/F1). This distinguishes two different kinds of model weakness that
are easily conflated: being *wrong* versus being *overconfident while
wrong*.

**Method:** Per-class reliability diagrams (predicted probability vs.
observed frequency, via `sklearn.calibration_curve`, 10 bins) and
Expected Calibration Error (ECE) — a scalar summary of the average gap
between predicted confidence and actual accuracy across bins.

**Results:**

| Class | Expected Calibration Error (ECE) |
|---|---|
| NORM | 0.0303 |
| MI | 0.0339 |
| STTC | 0.0268 |
| CD | 0.0329 |
| HYP | 0.0183 |

**Interpretation:** All five classes are well-calibrated (ECE in the
~0.02–0.03 range is small). Notably, **HYP has the lowest ECE of all five
classes** — meaning it is not merely acceptably calibrated, but the
*best*-calibrated class, despite being simultaneously the weakest by F1
(Section 5.11). This is an important and non-obvious distinction: HYP's
poor F1 is **not** a symptom of the model being overconfident or poorly
calibrated about that class. Rather, the model is correctly and honestly
representing high uncertainty on HYP cases — a well-calibrated-but-weak
signal, which is a more trustworthy and more tractable failure mode
(addressable via more data, targeted loss reweighting, or architectural
changes) than a badly-calibrated one would be. One caveat: HYP has the
fewest true-positive test cases of the five classes, so its calibration
bins — particularly at the high-confidence end — are more sparsely
populated and the ECE estimate is correspondingly noisier than for the
higher-prevalence classes.

Saved to: `cnn_calibration_reliability_diagrams.png`

---

### 5.13 CNN Error Analysis — `cnn_error_analysis.py`

**Purpose:** Extend the baseline's error-analysis discipline (Section
5.5) to the CNN, and additionally persist every individual error case
(tagged by `ecg_id`) to enable the cross-model comparison in Section
5.15.

**Method:** Computes confusion matrices per class at the CNN's tuned
thresholds (Section 5.11), extracts every false positive and false
negative, and joins each against PTB-XL patient metadata (age, sex) via
`ecg_id`.

**Confusion matrices (test set, CNN, tuned thresholds):**

| Class | TP | FP | FN | TN |
|---|---|---|---|---|
| NORM | 888 | 229 | 75 | 966 |
| MI | 427 | 192 | 123 | 1,416 |
| STTC | 386 | 122 | 135 | 1,515 |
| CD | 359 | 111 | 137 | 1,551 |
| HYP | 128 | 164 | 134 | 1,732 |

**Interpretation:** These counts are consistent with the precision/recall
figures already reported in Section 5.11. HYP again stands out: its false
positive count (164) exceeds its true positive count (128), and its
false negative count (134) is nearly as large as its true positives —
concretely, the model is wrong more often than right on positive HYP
predictions, and misses roughly as many true HYP cases as it catches.

Raw per-subgroup (age/sex) error counts were also produced at this stage,
but are **not reported as standalone findings**, since raw counts
conflate genuine subgroup performance differences with subgroup sample
size differences. This limitation directly motivated Section 5.14.

Saved to: `cnn_test_errors.csv`

---

### 5.14 CNN Subgroup Rate Analysis — `cnn_subgroup_rates.py`

**Purpose:** Correct the limitation identified in Section 5.13 by
computing subgroup performance as **rates** (recall, precision) rather
than raw counts — since a subgroup with more total cases will
mechanically accumulate more raw errors even under identical underlying
model performance.

**Method:** For each class, computes recall and precision within four age
brackets (`<40`, `40–54`, `55–69`, `70+`) and by sex, with the number of
true-positive cases in each subgroup (`n_positive_cases`) reported
alongside every rate so that estimates built on small samples can be
identified and appropriately discounted.

**Results — Recall/Precision by Age Group:**

| Class | Age Group | n (positive cases) | Recall | Precision |
|---|---|---|---|---|
| MI | <40 | 7 | 0.571 | 0.800 |
| MI | 40–54 | 76 | 0.711 | 0.711 |
| MI | 55–69 | 172 | 0.744 | 0.631 |
| MI | 70+ | 280 | 0.811 | 0.723 |
| NORM | <40 | 256 | 0.965 | 0.888 |
| NORM | 40–54 | 283 | 0.940 | 0.872 |
| NORM | 55–69 | 297 | 0.889 | 0.754 |
| NORM | 70+ | 127 | 0.874 | 0.603 |
| STTC | <40 | 23 | 0.609 | 0.667 |
| STTC | 40–54 | 50 | 0.740 | 0.771 |
| STTC | 55–69 | 183 | 0.776 | 0.789 |
| STTC | 70+ | 246 | 0.732 | 0.735 |
| CD | <40 | 39 | 0.513 | 0.800 |
| CD | 40–54 | 60 | 0.633 | 0.704 |
| CD | 55–69 | 151 | 0.669 | 0.759 |
| CD | 70+ | 230 | 0.800 | 0.767 |
| HYP | <40 | 11 | 0.273 | 0.429 |
| HYP | 40–54 | 29 | 0.345 | 0.345 |
| HYP | 55–69 | 97 | 0.495 | 0.471 |
| HYP | 70+ | 120 | 0.525 | 0.434 |

**Results — Recall/Precision by Sex:**

| Class | Sex Code | n (positive cases) | Recall | Precision |
|---|---|---|---|---|
| NORM | 0 | 516 | 0.942 | 0.821 |
| NORM | 1 | 447 | 0.899 | 0.766 |
| MI | 0 | 297 | 0.801 | 0.726 |
| MI | 1 | 253 | 0.747 | 0.649 |
| STTC | 0 | 229 | 0.755 | 0.786 |
| STTC | 1 | 292 | 0.729 | 0.740 |
| CD | 0 | 266 | 0.737 | 0.775 |
| CD | 1 | 230 | 0.709 | 0.751 |
| HYP | 0 | 135 | 0.467 | 0.434 |
| HYP | 1 | 127 | 0.512 | 0.442 |

*Note: sex codes follow the PTB-XL metadata convention (0/1); mapping to
male/female was not independently re-verified in this analysis and
should be confirmed against PTB-XL's documentation before external
communication of sex-specific findings.*

**Interpretation:**

- **A corrected hypothesis:** raw false-negative counts for MI (Section
  5.13's data, prior to normalization) appeared to climb steeply with
  age, initially suggesting a possible age-related recall deficit. The
  rate-normalized data shows the **opposite** — MI recall actually
  *improves* with age (0.571 → 0.711 → 0.744 → 0.811). The raw-count
  pattern was an artifact of there being far more true MI cases in older
  patients (n=280 for 70+ vs. n=7 for <40), not a genuine model
  weakness. This is a useful illustration of why rate normalization is
  necessary before drawing subgroup conclusions.
- **The single strongest, best-powered finding: NORM precision degrades
  substantially with age.** Precision falls monotonically from 0.888
  (<40) to 0.603 (70+), a large effect built on solid sample sizes
  (127–297 cases per group). In practical terms, roughly 40% of the
  model's "NORM" predictions in patients 70+ are incorrect, versus
  roughly 11% in patients under 40. A plausible explanation is that older
  patients' ECGs more frequently contain borderline or incidental
  abnormalities that shift the waveform away from a clean "normal" shape
  without clearly satisfying any of the four positive diagnostic labels.
  **This is the headline subgroup finding of the project** and should be
  disclosed alongside any deployment or downstream use of NORM
  predictions.
- **CD and HYP recall also rise with age**, following the same
  prevalence-driven pattern established for MI — expected, not a novel
  concern on its own, though the smallest age brackets (n=7–39 positive
  cases for MI/CD/HYP under 40) are too small to treat as statistically
  robust standalone findings.
- **Sex-based differences are present but modest** — roughly 3–7 points
  of recall favor sex code 0 over sex code 1 on NORM/MI/STTC/CD (the
  pattern reverses slightly for HYP). Sample sizes (230–516 per group)
  are large enough that this is likely a real effect rather than noise,
  but the magnitude is small relative to the age/NORM finding above and
  is best treated as a limitations-section note rather than a headline
  result.

Saved to: `cnn_subgroup_rates_by_age.csv`, `cnn_subgroup_rates_by_sex.csv`

---

### 5.15 Cross-Model Error Comparison — `generate_xgb_errors.py` and `cross_model_error_comparison.py`

**Purpose:** Determine whether the CNN and the XGBoost baseline fail on
the *same* test cases or on *different* ones. This directly answers a
practical question left open by every prior comparison: is an ensemble
of the two models likely to be worthwhile? If both models are wrong on
the same hard cases, an ensemble has little room to help; if their errors
are largely independent, an ensemble could combine complementary
strengths.

**Method:**
1. `generate_xgb_errors.py` retrains the baseline models (which are not
   persisted to disk by `train_baseline.py`), applies the previously
   tuned thresholds (Section 5.6), and records every false positive and
   false negative with its `ecg_id`, in the same format as the CNN's
   error log (Section 5.13).
2. `cross_model_error_comparison.py` loads both error logs and computes,
   per class, the size of the intersection and each model's unique error
   set.

**Results:**

| Class | CNN Errors | XGBoost Errors | Both Wrong | % of CNN Errors Also Missed by XGBoost |
|---|---|---|---|---|
| NORM | 304 | 530 | 196 | 64% |
| MI | 315 | 647 | 189 | 60% |
| STTC | 257 | 603 | 147 | 57% |
| CD | 248 | 328 | 163 | 66% |
| HYP | 298 | 519 | 180 | 60% |

**Interpretation:** Because the XGBoost baseline has substantially more
total errors than the CNN in every class (1.5–2.5×), a simple
"overlap as a percentage of the combined error set" metric is misleading
— it mechanically understates agreement whenever one model is simply
weaker overall. The more diagnostic figure is **what fraction of the
CNN's own errors are cases the baseline also gets wrong**: this ranges
from 57% to 66% across all five classes. In other words, roughly
three-fifths of the CNN's mistakes are on cases that are independently
hard for a structurally different model, rather than idiosyncratic CNN
failures. Correspondingly, most of the baseline's *additional*, unique
errors are cases the CNN already handles correctly — meaning a naive
probability-averaging ensemble would risk diluting the CNN's already-
correct predictions on those cases in exchange for only a modest gain on
the smaller pool where the CNN is wrong but the baseline happens to be
right.

**Decision: an ensemble was not pursued.** Given the CNN's clear,
consistent superiority across every class (Section 5.9) and the
substantial, non-complementary overlap in error patterns, the expected
benefit of ensembling did not justify the added complexity (cross-model
probability calibration, a fresh round of threshold re-tuning). This
conclusion was reached empirically, by directly testing the overlap,
rather than assumed.

Saved to: `xgb_test_errors.csv`, `cross_model_error_overlap.csv`

---

## 6. Consolidated Results

| Class | Baseline AUROC | CNN AUROC | Baseline Threshold | CNN Threshold | CNN Test F1 | CNN ECE |
|---|---|---|---|---|---|---|
| NORM | 0.849 | 0.942 | 0.44 | 0.322 | 0.854 | 0.0303 |
| MI | 0.791 | 0.918 | 0.22 | 0.420 | 0.731 | 0.0339 |
| STTC | 0.794 | 0.936 | 0.26 | 0.510 | 0.750 | 0.0268 |
| CD | 0.820 | 0.922 | 0.42 | 0.494 | 0.743 | 0.0329 |
| HYP | 0.721 | 0.828 | 0.15 | 0.275 | 0.462 | 0.0183 |

---

## 7. Key Findings

1. **The CNN outperforms the hand-crafted baseline on every class**, by
   a consistent +0.09 to +0.14 AUROC — a genuine capability gain
   attributable to access to full waveform morphology rather than a
   fixed feature summary.
2. **HYP is a persistent, well-characterized limitation, not a bug.** It
   is the weakest class for both models, remains the weakest even after
   independent per-model threshold tuning, and — critically — is
   simultaneously the *best*-calibrated class, indicating the model is
   appropriately uncertain rather than confidently wrong. This is
   consistent with HYP's low prevalence (12.1%) and hypertrophy's known
   clinical ambiguity even among expert human readers.
3. **NORM prediction reliability degrades substantially with patient
   age** — precision falls from 0.888 in patients under 40 to 0.603 in
   patients 70 and older. This is the most actionable, best-powered
   subgroup finding in the project and warrants explicit disclosure in
   any downstream use.
4. **The waveform-derived signal is not a demographic proxy.** The
   age-only sanity check (Section 5.7) confirmed the baseline model
   relies on genuine ECG morphology, not merely age, with a +0.09 to
   +0.22 AUROC advantage over an age-only model in every class.
5. **Ensembling the two models is unlikely to help**, based on directly
   measured cross-model error overlap (57–66% of CNN errors are also
   baseline errors) rather than assumption.
6. **Sex-based performance differences exist but are modest** (roughly
   3–7 points of recall) relative to the age-related NORM finding, and
   are documented as a limitation rather than a primary result.

---

## 8. Limitations

- **HYP (Hypertrophy) performance is weak** (test F1 ≈ 0.46) in both
  modeling approaches, reflecting genuine class difficulty and low
  prevalence rather than a fixable pipeline defect. Any downstream
  consumer of this model's HYP predictions should treat them with
  reduced confidence relative to the other four classes.
- **NORM precision degrades substantially in patients aged 70 and
  older** — a "NORM" prediction is not equally reliable across the
  patient age range and should not be treated as such.
- **Age-stratified results for MI, CD, and HYP in the youngest bracket
  (under 40) rest on small sample sizes** (7–39 positive test cases) and
  should be treated as directional rather than conclusive.
- **This is a research and educational pipeline, not a validated
  clinical tool.** Decision thresholds are F1-optimal for this
  dataset's specific class balance and are not calibrated for any
  particular clinical decision-making context, patient population, or
  regulatory standard.
- **Trained and evaluated exclusively on the 100Hz version of PTB-XL.**
  Performance on the 500Hz version, on other ECG datasets, or under
  different lead configurations has not been evaluated and should not
  be assumed.
- **Sex-code-to-sex-label mapping was not independently re-verified**
  during subgroup analysis; any external communication of sex-based
  findings should first confirm this mapping against PTB-XL's official
  documentation.

---

## 9. Key Concepts

- **Multi-label classification:** A single ECG record may exhibit
  multiple diagnostic conditions simultaneously. Modeled as five
  independent binary decisions (sigmoid outputs / separate binary
  classifiers), not as a single softmax over mutually exclusive classes.
- **Patient-level split:** Because PTB-XL contains multiple ECGs from
  some patients, a naive row-level train/test split can leak
  patient-specific characteristics across the split boundary, inflating
  apparent performance. All splitting in this project respects patient
  boundaries.
- **AUROC vs. AUPRC:** AUROC is threshold-independent but can be
  misleadingly optimistic under class imbalance. AUPRC more accurately
  reflects real-world precision/recall tradeoffs when positive cases are
  rare (e.g., HYP at 12.1% prevalence), and is weighted more heavily in
  this project's interpretation of imbalanced classes.
- **Threshold tuning:** The default decision threshold of 0.5 is
  essentially never correct for imbalanced medical classification tasks.
  Thresholds are always tuned on validation data and reported on a
  held-out test set to avoid overfitting the threshold choice itself.
- **Calibration vs. discrimination:** A model can have weak
  discriminative performance (low F1/AUPRC) while still being
  well-calibrated (its probability outputs are honest), or vice versa.
  These are evaluated as separate, complementary properties in this
  project (Sections 5.11 vs. 5.12).
- **Age-only sanity check:** A standard rigor check in clinical machine
  learning — if a trivial demographic-only model performs close to the
  full model, the full model's features may be substantially proxying
  demographics rather than learning condition-specific signal.
- **BCEWithLogitsLoss:** The appropriate PyTorch loss function for
  multi-label classification; applies sigmoid internally with better
  numerical stability than a separate sigmoid activation followed by
  binary cross-entropy loss.

---

## 10. Reproducibility

See `README.md` for environment setup and the exact script execution
order required to reproduce every result in this document from raw data.