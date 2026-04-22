"""
utils.py — Fungsi-Fungsi Pendukung
===================================
Berisi:
  - set_seed()              : reproducibility
  - get_criterion()         : CrossEntropyLoss (opsional weighted)
  - plot_training_curves()  : kurva loss & accuracy per epoch
  - plot_confusion_matrix() : heatmap confusion matrix
  - print_report()          : cetak classification report
"""

import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from datareader import LABEL_LIST, NUM_CLASSES, PLOT_DIR


# ──────────────────────────────────────────────
# REPRODUCIBILITY
# ──────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    """Set seed PyTorch, NumPy, dan random agar hasil reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ──────────────────────────────────────────────
# LOSS FUNCTION
# ──────────────────────────────────────────────

def get_criterion(label_counts: dict | None = None) -> nn.CrossEntropyLoss:
    """
    Buat CrossEntropyLoss.

    Jika label_counts diberikan, hitung class weight untuk menangani
    ketidakseimbangan kelas (class imbalance).

    Args:
        label_counts: Dict {nama_label: jumlah_sampel}. None = tanpa weighting.

    Returns:
        nn.CrossEntropyLoss
    """
    if label_counts is None:
        return nn.CrossEntropyLoss()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total  = sum(label_counts.values())
    weights = torch.tensor(
        [total / (NUM_CLASSES * label_counts.get(lbl, 1)) for lbl in LABEL_LIST],
        dtype=torch.float,
    ).to(device)
    return nn.CrossEntropyLoss(weight=weights)


# ──────────────────────────────────────────────
# VISUALISASI — Training Curves
# ──────────────────────────────────────────────

def plot_training_curves(history: dict, save: bool = True) -> None:
    """
    Plot loss dan accuracy (train vs val) per epoch.

    Args:
        history : dict dengan key 'train_loss', 'val_loss',
                  'train_acc', 'val_acc'.
        save    : Simpan gambar ke plots/ jika True.
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    # Simpan tabel metrik per-epoch agar mudah dianalisis ulang.
    metrics_table = pd.DataFrame({
        "epoch": list(epochs),
        "train_loss": history.get("train_loss", []),
        "val_loss": history.get("val_loss", []),
        "train_accuracy": history.get("train_acc", []),
        "val_accuracy": history.get("val_acc", []),
        "train_precision": history.get("train_precision", []),
        "val_precision": history.get("val_precision", []),
        "train_recall": history.get("train_recall", []),
        "val_recall": history.get("val_recall", []),
        "train_f1": history.get("train_f1", []),
        "val_f1": history.get("val_f1", []),
        "train_lr": history.get("train_lr", []),
    })
    if save:
        csv_path = os.path.join(PLOT_DIR, "training_metrics.csv")
        metrics_table.to_csv(csv_path, index=False)
        print(f"Metrics disimpan: {csv_path}")

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle("Training Metrics — DistilBERT", fontsize=14, fontweight="bold")

    # Loss
    axes[0, 0].plot(epochs, history.get("train_loss", []), marker="o", label="Train Loss")
    axes[0, 0].plot(epochs, history.get("val_loss", []), marker="s", label="Val Loss")
    axes[0, 0].set_xlabel("Epoch"); axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("Loss"); axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

    # Accuracy
    axes[0, 1].plot(epochs, history.get("train_acc", []), marker="o", label="Train Acc")
    axes[0, 1].plot(epochs, history.get("val_acc", []), marker="s", label="Val Acc")
    axes[0, 1].set_xlabel("Epoch"); axes[0, 1].set_ylabel("Accuracy")
    axes[0, 1].set_title("Accuracy"); axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)

    # Precision
    axes[1, 0].plot(epochs, history.get("train_precision", []), marker="o", label="Train Precision")
    axes[1, 0].plot(epochs, history.get("val_precision", []), marker="s", label="Val Precision")
    axes[1, 0].set_xlabel("Epoch"); axes[1, 0].set_ylabel("Precision")
    axes[1, 0].set_title("Precision"); axes[1, 0].legend(); axes[1, 0].grid(alpha=0.3)

    # Recall
    axes[1, 1].plot(epochs, history.get("train_recall", []), marker="o", label="Train Recall")
    axes[1, 1].plot(epochs, history.get("val_recall", []), marker="s", label="Val Recall")
    axes[1, 1].set_xlabel("Epoch"); axes[1, 1].set_ylabel("Recall")
    axes[1, 1].set_title("Recall"); axes[1, 1].legend(); axes[1, 1].grid(alpha=0.3)

    # F1
    axes[2, 0].plot(epochs, history.get("train_f1", []), marker="o", label="Train F1")
    axes[2, 0].plot(epochs, history.get("val_f1", []), marker="s", label="Val F1")
    axes[2, 0].set_xlabel("Epoch"); axes[2, 0].set_ylabel("F1")
    axes[2, 0].set_title("F1 Score"); axes[2, 0].legend(); axes[2, 0].grid(alpha=0.3)

    # Learning Rate
    axes[2, 1].plot(epochs, history.get("train_lr", []), marker="o", color="tab:orange", label="LR")
    axes[2, 1].set_xlabel("Epoch"); axes[2, 1].set_ylabel("Learning Rate")
    axes[2, 1].set_title("Learning Rate"); axes[2, 1].legend(); axes[2, 1].grid(alpha=0.3)

    plt.tight_layout()
    if save:
        path = os.path.join(PLOT_DIR, "training_curves.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Plot disimpan: {path}")
    plt.show()


# ──────────────────────────────────────────────
# VISUALISASI — Confusion Matrix
# ──────────────────────────────────────────────

def plot_confusion_matrix(y_true: list, y_pred: list, save: bool = True) -> None:
    """
    Tampilkan confusion matrix sebagai heatmap.

    Args:
        y_true : List label aktual (integer).
        y_pred : List label prediksi (integer).
        save   : Simpan gambar ke plots/ jika True.
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(9, 7))

    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=LABEL_LIST, yticklabels=LABEL_LIST, ax=ax,
    )
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix — DistilBERT", fontsize=13, fontweight="bold")
    plt.xticks(rotation=30, ha="right"); plt.yticks(rotation=0)
    plt.tight_layout()

    if save:
        path = os.path.join(PLOT_DIR, "confusion_matrix.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Plot disimpan: {path}")
    plt.show()


# ──────────────────────────────────────────────
# LAPORAN KLASIFIKASI
# ──────────────────────────────────────────────

def print_report(y_true: list, y_pred: list) -> None:
    """Cetak precision, recall, F1 per kelas."""
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=LABEL_LIST, digits=4))
