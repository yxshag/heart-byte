#!/usr/bin/env python3

"""
=======================================================================================================
Author:         yxshag
Created:        03-07-2026
GitHub:         https://github.com/yxshag

Project:        heart-byte
File:           cnn_inference_utils.py
Description:    Shared utilities for running inference with the trained 1D CNN and loading
                val/test splits for the ecg-classifier(heart-byte) project.
=======================================================================================================
"""



import numpy as np
import torch

# ---------------------Paths-DO NOT CHANGE-----------------------------------
CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
DATA_DIR = "data/processed"
CHECKPOINT_PATH = "models/cnn_best.pt"

try:
    from train_cnn import ECGCNN as ModelClass
except ImportError:
    raise ImportError(
        "Could not import a class named ECGCNN, either fix the name of the class"
        "or directly copy paste the code of the ECGCNN class into this file."
    )
# -----------------------------------------------------------------------------


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_split(split_name):
    path = f"{DATA_DIR}/{split_name}.npz"
    data = np.load(path)
    X, y = data["X"], data["y"]
    return X, y


def load_model(checkpoint_path=CHECKPOINT_PATH, device=None):
    device = device or get_device()
    model = ModelClass(n_classes=len(CLASSES))
    checkpoint = torch.load(checkpoint_path, map_location=device)
    #Handle the format of the checkpoint, some checkpoints save the checkpoint as {"model_state_dict":checkpoint}
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def get_predictions(model, X, device, batch_size=256):
    probs = []
    for i in range(0, len(X), batch_size):
        batch = torch.from_numpy(X[i:i + batch_size]).float().to(device)
        if batch.shape[1] != 12:          # (batch, seq_len, n_leads) -> (batch, n_leads, seq_len)
            batch = batch.permute(0, 2, 1)
        logits = model(batch)
        batch_probs = torch.sigmoid(logits).cpu().numpy() #converting the logits to probability
        probs.append(batch_probs)
    return np.concatenate(probs, axis=0)
