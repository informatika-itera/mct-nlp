"""NGBoost training, evaluation, dan visualisasi untuk IMDB sentiment."""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)
from ngboost import NGBClassifier
from ngboost.distns import Bernoulli
from sklearn.tree import DecisionTreeRegressor

from config import (
    RANDOM_SEED, SAMPLE_SIZE, TEST_SIZE, TFIDF_MAX_FEATURES,
    NGB_N_ESTIMATORS, NGB_LEARNING_RATE, NGB_MINIBATCH_FRAC,
    NGB_VERBOSE_EVAL, MODEL_DIR, PLOT_DIR, LABEL_COL,
)


# ── 1. Pipeline builder ────────────────────────────────────────────────

def build_pipeline(df: pd.DataFrame):
    """
    Sample data, encode label, vectorize teks, split train/test.
    Returns (X_train, X_test, y_train, y_test, vectorizer, label_encoder).
    """
    if SAMPLE_SIZE and len(df) > SAMPLE_SIZE:
        df = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_SEED).reset_index(drop=True)
        print(f"[INFO] Sampled {SAMPLE_SIZE} rows untuk training.")

    le = LabelEncoder()
    y = le.fit_transform(df[LABEL_COL])  # negative=0, positive=1

    tfidf = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES)
    X = tfidf.fit_transform(df["cleaned_text"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y,
    )
    print(f"[INFO] Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
    print(f"[INFO] Label mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    return X_train, X_test, y_train, y_test, tfidf, le


# ── 2. Training ────────────────────────────────────────────────────────

def train_ngboost(X_train, y_train):
    """Latih NGBClassifier dan kembalikan model."""
    base_learner = DecisionTreeRegressor(
        max_depth=4,
        random_state=RANDOM_SEED,
    )
    model = NGBClassifier(
        Dist=Bernoulli,
        Base=base_learner,
        n_estimators=NGB_N_ESTIMATORS,
        learning_rate=NGB_LEARNING_RATE,
        minibatch_frac=NGB_MINIBATCH_FRAC,
        random_state=RANDOM_SEED,
        verbose_eval=NGB_VERBOSE_EVAL,
        verbose=True,
    )
    print("\n[INFO] Training NGBoost ...")
    model.fit(X_train, y_train)
    print("[INFO] Training selesai.\n")
    return model


# ── 3. Evaluation ──────────────────────────────────────────────────────

def evaluate(model, X_test, y_test, label_encoder):
    """Hitung metrik dan print classification report."""
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="binary")
    rec = recall_score(y_test, y_pred, average="binary")
    f1 = f1_score(y_test, y_pred, average="binary")

    print("=" * 55)
    print("  PERFORMANCE METRICS")
    print("=" * 55)
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print("=" * 55)
    print()
    print(classification_report(
        y_test, y_pred,
        target_names=label_encoder.classes_,
    ))

    metrics = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}
    return y_pred, metrics


# ── 4. Visualisasi ─────────────────────────────────────────────────────

def plot_label_distribution(df: pd.DataFrame, save_path: str | None = None):
    """Bar chart distribusi sentiment."""
    save_path = save_path or os.path.join(PLOT_DIR, "distribusi_sentiment.png")
    counts = df[LABEL_COL].value_counts()

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#4CAF50", "#F44336"]
    counts.plot.bar(ax=ax, color=colors, edgecolor="black")
    ax.set_title("Distribusi Sentiment", fontsize=14, fontweight="bold")
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Jumlah")
    ax.set_xticklabels(counts.index, rotation=0)
    for i, v in enumerate(counts.values):
        ax.text(i, v + 100, str(v), ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[PLOT] Distribusi sentiment → {save_path}")


def plot_confusion_matrix(y_test, y_pred, label_encoder, save_path: str | None = None):
    """Heatmap confusion matrix."""
    save_path = save_path or os.path.join(PLOT_DIR, "confusion_matrix.png")
    cm = confusion_matrix(y_test, y_pred)
    labels = label_encoder.classes_

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels, ax=ax,
    )
    ax.set_title("Confusion Matrix — NGBoost", fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[PLOT] Confusion matrix → {save_path}")


def plot_training_loss(model, save_path: str | None = None):
    """Plot training log-loss dari NGBoost."""
    save_path = save_path or os.path.join(PLOT_DIR, "training_loss.png")

    losses = None
    if hasattr(model, "evals_result") and model.evals_result:
        train_res = model.evals_result.get("train", {})
        for key in train_res:
            losses = train_res[key]
            break

    if not losses:
        print("[WARN] Tidak ada data training loss, skip plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(losses, linewidth=1.5, color="#1976D2")
    ax.set_title("NGBoost Training Loss", fontsize=14, fontweight="bold")
    ax.set_xlabel("Iterasi")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[PLOT] Training loss → {save_path}")


def plot_metrics_bar(metrics: dict, save_path: str | None = None):
    """Bar chart metrik evaluasi."""
    save_path = save_path or os.path.join(PLOT_DIR, "metrics_bar.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    names = list(metrics.keys())
    values = list(metrics.values())
    colors = ["#2196F3", "#FF9800", "#4CAF50", "#9C27B0"]
    bars = ax.bar(names, values, color=colors, edgecolor="black")
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005,
                f"{v:.4f}", ha="center", fontweight="bold", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_title("Performance Metrics — NGBoost", fontsize=14, fontweight="bold")
    ax.set_ylabel("Score")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[PLOT] Metrics bar chart → {save_path}")


# ── 5. Save / Load ────────────────────────────────────────────────────

def save_model(model, vectorizer, label_encoder, filename="ngboost_imdb"):
    """Simpan model, vectorizer, dan label encoder."""
    path = os.path.join(MODEL_DIR, f"{filename}.joblib")
    joblib.dump(
        {"model": model, "vectorizer": vectorizer, "label_encoder": label_encoder},
        path,
    )
    print(f"[INFO] Model saved → {path}")
    return path


def load_model(filename="ngboost_imdb"):
    """Load model artifacts."""
    path = os.path.join(MODEL_DIR, f"{filename}.joblib")
    artifacts = joblib.load(path)
    print(f"[INFO] Model loaded ← {path}")
    return artifacts["model"], artifacts["vectorizer"], artifacts["label_encoder"]
