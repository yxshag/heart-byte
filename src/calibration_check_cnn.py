#!/usr/bin/env python3

"""
=======================================================================================================
Author:         yxshag
Created:        03-07-2026
GitHub:         https://github.com/yxshag

Project:        heart-byte
File:           calibration_check_cnn.py
Description:    Calibration check for the CNN's per-class probability outputs.
                Draws out the reliability diagram.
=======================================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

from cnn_inference_utils import CLASSES, load_split, load_model, get_predictions, get_device


def expected_calibration_error(y_true, y_prob, n_bins=10):
    """
    Calculates the gap between the calculated probabilities and actual accuracy
    
    Returns:
        ece:
            (float) value that gives the expected calibration error
    """
    #forms 10 bins by default, 0.0,0.1,0.2,0.3.....
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        #sort the data into bins
        if i < n_bins - 1:
            mask = (y_prob >= lo) & (y_prob < hi)
        else:
            mask = (y_prob >= lo) & (y_prob <= hi)
        if mask.sum() == 0:
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return ece


def main():
    device = get_device()
    print(f"Using device: {device}")

    model = load_model(device=device)
    X_test, y_test = load_split("test")
    test_probs = get_predictions(model, X_test, device)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    print("\n" + "=" * 55)
    print("Expected Calibration Error (ECE) per class")
    print("=" * 55)

    for i, cls in enumerate(CLASSES):
        y_true = y_test[:, i]
        y_prob = test_probs[:, i]
        #calibration curve basically gives us the value of fraction of positives and mean predicted value
        #Its the same thing that we have in ece function , they are analogous to bin_acc, bin_conf.
        frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="uniform")
        ece = expected_calibration_error(y_true, y_prob, n_bins=10)
        print(f"{cls:6s} | ECE: {ece:.4f}")

        ax = axes[i]
        ax.plot(mean_pred, frac_pos, marker="o", label=f"{cls} (ECE={ece:.3f})")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Observed frequency")
        ax.set_title(cls)
        ax.legend(fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    axes[-1].axis("off")  # 6th subplot unused (only 5 classes)
    fig.suptitle("CNN Calibration: Reliability Diagrams by Class", fontsize=14)
    fig.tight_layout()
    fig.savefig("./results/cnn_calibration_reliability_diagrams.png", dpi=150)
    print("\nSaved reliability diagrams to results/cnn_calibration_reliability_diagrams.png")


if __name__ == "__main__":
    main()
