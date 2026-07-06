#!/usr/bin/env python3

"""
================================================================================
Author:         yxshag
Created:        28-06-2026
GitHub:         https://github.com/yxshag

Project:        heart-byte
File:           train_cnn.py
Description:    [Brief 1-2 sentence overview of what this file does.]
================================================================================
"""

"""
train_cnn.py
-------------
1D CNN trained directly on the raw (filtered, normalized) waveform,
instead of hand-crafted summary features. The goal: capture localized
waveform shape (e.g. ST-segment elevation, T-wave inversion) that the
baseline's mean/std/min/max-style features structurally can't see.

Architecture: a small stack of 1D conv blocks (conv -> batchnorm -> relu
-> pooling) that progressively shrink the time dimension while building
up channel depth, followed by global average pooling and a small
fully-connected head with one sigmoid output per class (multi-label).

Training uses the SAME train/val/test split already verified leak-free.
Early stopping is based on validation AUROC (averaged across classes),
not training loss, to avoid overfitting.

Usage:
    python src/train_cnn.py
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

from train_baseline import SUPERCLASSES, PROCESSED_DIR, RESULTS_DIR

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

BATCH_SIZE = 64       #load only 64 files at one time so as to not overload the system
MAX_EPOCHS = 60
PATIENCE = 8          # stop if val AUROC doesn't improve for this many epochs
LEARNING_RATE = 0.001


class ECGDataset(Dataset):
    """
    This class inherits from the Dataset class from torch.
    This is required to convert the data into a tensor as pytorch doesnt understand numpy.
    """
    def __init__(self, X, y):
        #Transpose is required to convert it into proper tensor
        self.X = torch.tensor(X, dtype=torch.float32).permute(0, 2, 1)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

#Defining the neural network, it directly inherits from nn.Module
class ECGCNN(nn.Module):
    def __init__(self, n_leads=12, n_classes=5):
        super().__init__()

        def conv_block(in_ch, out_ch, kernel_size=7, pool=2):
            return nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2),#makes sure the data length doesnt change
                nn.BatchNorm1d(out_ch), #Normalize the output channels so we can ignore the changes present due to change in machine
                nn.ReLU(),
                nn.MaxPool1d(pool),#Pools every 2 timesteps into 1 timestep and replace it with max value
            )

        #timesteps goes from:- 1000->500->250->125->62
        #Number of feautures goes from :- 12->32->64->128->128
        self.features = nn.Sequential(
            conv_block(n_leads, 32),
            conv_block(32, 64),     
            conv_block(64, 128),    
            conv_block(128, 128),   
        )

        #Once we get the 128 feature, 62 timestep signal, we need to convert it into purely 128 feature signal, that can be done by taking the average
        #value in each of the features.
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        #Our data went from 12,1000 to 128,1
        #The classifier uses these 128 specific features to make a prediction
        
        self.classifier = nn.Sequential(
            nn.Linear(128, 64), #Fully-connected layer 1
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, n_classes),  #Fully-connected layer 2
            # No sigmoid here -- BCEWithLogitsLoss applies it internally,
            # which is more numerically stable than doing it manually.
        )

    #Forward function

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x).squeeze(-1)
        return self.classifier(x)


def load_split(split_name):
    data = np.load(PROCESSED_DIR / f"{split_name}.npz")
    return data["X"], data["y"]


def run_epoch(model, loader, criterion, optimizer=None):
    """
    Run a single epoch of training or evaluation over the given data loader.

    If `optimizer` is provided, the model is put into training mode and
    weights are updated via backpropagation for each batch. If `optimizer`
    is None, the model is run in evaluation mode with gradients disabled
    (validation/test pass only, no weight updates).

    Assumes a multi-label classification setup: `model` outputs raw logits
    of shape (batch_size, num_classes), `criterion` is a loss function
    suitable for multi-label targets (e.g. BCEWithLogitsLoss), and
    `y_batch` contains multi-hot (0/1) labels of the same shape.

    Parameters
    ----------
    model : torch.nn.Module
        The model to train or evaluate. Must output raw logits (pre-sigmoid).
    loader : torch.utils.data.DataLoader
        Yields (X_batch, y_batch) pairs. y_batch must be multi-label
        (multi-hot encoded), shape (batch_size, num_classes).
    criterion : callable
        Loss function taking (logits, y_batch) and returning a scalar loss.
        Expected to internally apply sigmoid (e.g. BCEWithLogitsLoss).
    optimizer : torch.optim.Optimizer, optional
        Optimizer used to update model weights. If None (default), the
        function runs in evaluation-only mode with gradients disabled.

    Returns
    -------
    avg_loss : float
        Average loss per sample across the epoch.
    mean_auroc : float
        Mean AUROC across all classes, computed independently per class
        and averaged. Used as the early-stopping signal.
    all_probs : np.ndarray, shape (n_samples, num_classes)
        Predicted probabilities (post-sigmoid) for every sample in the
        loader, in iteration order.
    all_labels : np.ndarray, shape (n_samples, num_classes)
        Ground-truth multi-hot labels for every sample, in the same order
        as `all_probs`.

    Notes
    -----
    - Requires a global `DEVICE` (torch.device) to be defined, used to move
      batches to the correct device.
    - `roc_auc_score` will raise a ValueError for any class where all
      labels in the epoch are the same value (all 0s or all 1s) — this can
      happen with small batches or highly imbalanced classes.
    """
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    all_probs, all_labels = [], []

    with torch.set_grad_enabled(is_train):
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)

            logits = model(X_batch)
            loss = criterion(logits, y_batch)

            if is_train:
                #Clear previous gradients, compute newer gradients based on current batch, update weights using those gradients
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            #Computing total loss
            total_loss += loss.item() * X_batch.size(0)
            all_probs.append(torch.sigmoid(logits).detach().cpu().numpy())
            all_labels.append(y_batch.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)

    # Mean AUROC across the 5 classes -- our early-stopping signal
    per_class_auroc = [
        roc_auc_score(all_labels[:, i], all_probs[:, i])
        for i in range(all_labels.shape[1])
    ]
    mean_auroc = np.mean(per_class_auroc)

    return avg_loss, mean_auroc, all_probs, all_labels


def main():
    #Create the Dataset in batches for train, val and testing
    X_train, y_train = load_split("train")
    X_val, y_val = load_split("val")
    X_test, y_test = load_split("test")

    train_loader = DataLoader(ECGDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(ECGDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(ECGDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

    #Create the model
    
    model = ECGCNN(n_leads=X_train.shape[2], n_classes=len(SUPERCLASSES)).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_auroc = -1
    epochs_without_improvement = 0
    best_model_path = RESULTS_DIR.parent / "models" / "cnn_best.pt"
    best_model_path.parent.mkdir(exist_ok=True)

    print(f"Training on {len(X_train)} records, validating on {len(X_val)}...")

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss, train_auroc, _, _ = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_auroc, _, _ = run_epoch(model, val_loader, criterion)

        print(
            f"Epoch {epoch:3d} | train loss {train_loss:.4f} auroc {train_auroc:.3f} "
            f"| val loss {val_loss:.4f} auroc {val_auroc:.3f}"
        )

        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            epochs_without_improvement = 0
            #Update the latest model
            torch.save(model.state_dict(), best_model_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print(f"\nNo val improvement for {PATIENCE} epochs, stopping early.")
                break

    #Load up the best model so far
    model.load_state_dict(torch.load(best_model_path))
    test_loss, test_mean_auroc, test_probs, test_labels = run_epoch(model, test_loader, criterion)

    report_lines = ["CNN results (raw waveform, 1D conv net)"]
    report_lines.append("=" * 55)
    report_lines.append(f"Best val mean AUROC during training: {best_val_auroc:.3f}\n")

    for i, cls_name in enumerate(SUPERCLASSES):
        auroc = roc_auc_score(test_labels[:, i], test_probs[:, i])
        auprc = average_precision_score(test_labels[:, i], test_probs[:, i])
        prevalence = test_labels[:, i].mean()
        report_lines.append(
            f"{cls_name:6s} | AUROC: {auroc:.3f} | AUPRC: {auprc:.3f} "
            f"| test prevalence: {prevalence:.1%}"
        )

    report_text = "\n".join(report_lines)
    print("\n" + report_text)

    out_path = RESULTS_DIR / "cnn_results.txt"
    with open(out_path, "w") as f:
        f.write(report_text)
    print(f"\nSaved results to {out_path}")
    print(f"Best model checkpoint saved to {best_model_path}")


if __name__ == "__main__":
    main()
