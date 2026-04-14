"""
train.py — Training Loop, Evaluasi, dan Visualisasi
=====================================================
Menyediakan fungsi modular untuk:
- Melatih model per epoch
- Early stopping
- Evaluasi & laporan klasifikasi
- Visualisasi: training curves, confusion matrix,
  attention heatmap, komparasi model
- Inferensi teks tunggal
"""

import os
import time
import copy

import torch
import torch.nn as nn
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
)

from config import (
    DEVICE, LABEL_LIST, NUM_CLASSES,
    LSTM_LR, LSTM_PATIENCE, BERT_LR, BERT_PATIENCE,
    PLOT_DIR, MODEL_DIR,
)

matplotlib.use("Agg")   # backend tanpa GUI agar aman di notebook & terminal


# ──────────────────────────────────────────────
# ⚙️  HELPER: seed
# ──────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    """Set seed untuk reproducibility (PyTorch, NumPy, random)."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ──────────────────────────────────────────────
# ⚙️  HELPER: weighted criterion (class imbalance)
# ──────────────────────────────────────────────

def get_criterion(label_counts: dict | None = None) -> nn.CrossEntropyLoss:
    """
    Membuat CrossEntropyLoss dengan class weight opsional.

    Args:
        label_counts: Dict {label_name: count}. Jika None, tanpa weighting.

    Returns:
        nn.CrossEntropyLoss
    """
    if label_counts is None:
        return nn.CrossEntropyLoss()

    total = sum(label_counts.values())
    weights = torch.tensor(
        [total / (NUM_CLASSES * label_counts.get(lbl, 1)) for lbl in LABEL_LIST],
        dtype=torch.float,
    ).to(DEVICE)
    return nn.CrossEntropyLoss(weight=weights)


# ──────────────────────────────────────────────
# 🔄 TRAINING: satu epoch
# ──────────────────────────────────────────────

def train_one_epoch_lstm(
    model: nn.Module,
    dataloader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device = DEVICE,
) -> tuple[float, float]:
    """
    Melatih BiLSTM / BiLSTMAttention selama satu epoch.

    Returns:
        (avg_loss, accuracy)
    """
    model.train()
    total_loss = 0.0
    correct    = 0
    total      = 0

    for batch in dataloader:
        x, labels, lengths = batch
        x, labels, lengths = x.to(device), labels.to(device), lengths.to(device)

        optimizer.zero_grad()

        if hasattr(model, "attention"):
            logits, _ = model(x, lengths)
        else:
            logits = model(x, lengths)

        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds   = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

    return total_loss / total, correct / total


def train_one_epoch_bert(
    model: nn.Module,
    dataloader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device = DEVICE,
) -> tuple[float, float]:
    """
    Melatih DistilBERT selama satu epoch.

    Returns:
        (avg_loss, accuracy)
    """
    model.train()
    total_loss = 0.0
    correct    = 0
    total      = 0

    for batch in dataloader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss   = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds   = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

    return total_loss / total, correct / total


# ──────────────────────────────────────────────
# 📊 EVALUASI
# ──────────────────────────────────────────────

def evaluate_lstm(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    device: torch.device = DEVICE,
) -> tuple[float, float, list, list]:
    """
    Evaluasi BiLSTM / BiLSTMAttention pada dataloader.

    Returns:
        (avg_loss, accuracy, all_preds, all_labels)
    """
    model.eval()
    total_loss = 0.0
    correct    = 0
    total      = 0
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            x, labels, lengths = batch
            x, labels, lengths = x.to(device), labels.to(device), lengths.to(device)

            if hasattr(model, "attention"):
                logits, _ = model(x, lengths)
            else:
                logits = model(x, lengths)

            loss = criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)
            preds   = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return total_loss / total, correct / total, all_preds, all_labels


def evaluate_bert(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    device: torch.device = DEVICE,
) -> tuple[float, float, list, list]:
    """
    Evaluasi DistilBERT pada dataloader.

    Returns:
        (avg_loss, accuracy, all_preds, all_labels)
    """
    model.eval()
    total_loss = 0.0
    correct    = 0
    total      = 0
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["label"].to(device)

            logits = model(input_ids, attention_mask)
            loss   = criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)
            preds   = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return total_loss / total, correct / total, all_preds, all_labels


# ──────────────────────────────────────────────
# 🏋️  TRAINING LOOP LENGKAP (dengan early stopping)
# ──────────────────────────────────────────────

def train_model(
    model: nn.Module,
    train_loader,
    val_loader,
    model_type: str,          # "lstm" | "lstm_att" | "bert"
    save_path: str,
    epochs: int,
    lr: float,
    patience: int,
    device: torch.device = DEVICE,
    label_counts: dict | None = None,
) -> dict:
    """
    Melatih model dengan early stopping dan menyimpan model terbaik.

    Args:
        model:        Model PyTorch.
        train_loader: DataLoader training.
        val_loader:   DataLoader validasi.
        model_type:   "lstm", "lstm_att", atau "bert".
        save_path:    Path untuk menyimpan model terbaik.
        epochs:       Jumlah epoch maksimum.
        lr:           Learning rate.
        patience:     Early stopping patience.
        device:       CPU / GPU.
        label_counts: Dict {label: count} untuk weighted loss.

    Returns:
        Dict history berisi loss & accuracy per epoch.
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = get_criterion(label_counts)

    use_bert   = (model_type == "bert")
    train_fn   = train_one_epoch_bert if use_bert else train_one_epoch_lstm
    eval_fn    = evaluate_bert        if use_bert else evaluate_lstm

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss":   [], "val_acc":   [],
    }

    best_val_loss    = float("inf")
    best_model_state = None
    patience_counter = 0
    start_time       = time.time()

    print(f"\n{'='*60}")
    print(f"  Training: {model_type.upper()}  |  Device: {device}")
    print(f"  Epochs: {epochs}  |  LR: {lr}  |  Patience: {patience}")
    print(f"{'='*60}")

    for epoch in range(1, epochs + 1):
        t0           = time.time()
        train_loss, train_acc = train_fn(model, train_loader, optimizer, criterion, device)
        val_loss,   val_acc, _, _ = eval_fn(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - t0
        print(
            f"  Epoch {epoch:>3}/{epochs} | "
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f} | "
            f"{elapsed:.1f}s"
        )

        # ── Early stopping ──
        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            torch.save(best_model_state, save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n  ⏹ Early stopping di epoch {epoch}")
                break

    total_time = time.time() - start_time
    history["total_time"] = total_time
    print(f"\n  ✅ Selesai dalam {total_time/60:.1f} menit  |  Best Val Loss: {best_val_loss:.4f}")
    print(f"  💾 Model disimpan: {save_path}")

    # Muat kembali model terbaik
    model.load_state_dict(best_model_state)
    return history


# ──────────────────────────────────────────────
# 📈 VISUALISASI
# ──────────────────────────────────────────────

def plot_training_curves(history: dict, model_name: str, save: bool = True) -> None:
    """Menampilkan kurva loss dan accuracy training vs validasi."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Training Curves — {model_name}", fontsize=14, fontweight="bold")

    # Loss
    axes[0].plot(epochs, history["train_loss"], label="Train Loss", marker="o")
    axes[0].plot(epochs, history["val_loss"],   label="Val Loss",   marker="s")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, history["train_acc"], label="Train Acc", marker="o")
    axes[1].plot(epochs, history["val_acc"],   label="Val Acc",   marker="s")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy"); axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    if save:
        path = os.path.join(PLOT_DIR, f"training_curves_{model_name.lower().replace(' ', '_')}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"💾 Plot disimpan: {path}")
    plt.show()


def plot_confusion_matrix(
    y_true: list,
    y_pred: list,
    model_name: str,
    save: bool = True,
) -> None:
    """Menampilkan confusion matrix sebagai heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(9, 7))

    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=LABEL_LIST, yticklabels=LABEL_LIST,
        ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Actual",    fontsize=11)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13, fontweight="bold")
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save:
        path = os.path.join(PLOT_DIR, f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"💾 Plot disimpan: {path}")
    plt.show()


def print_classification_report(
    y_true: list,
    y_pred: list,
    model_name: str,
) -> dict:
    """
    Mencetak classification report dan mengembalikan dict metrik.

    Returns:
        Dict {"accuracy": ..., "f1_macro": ..., "f1_weighted": ...}
    """
    print(f"\n{'='*60}")
    print(f"  Classification Report — {model_name}")
    print(f"{'='*60}")
    print(classification_report(y_true, y_pred, target_names=LABEL_LIST, zero_division=0))

    acc        = accuracy_score(y_true, y_pred)
    f1_macro   = f1_score(y_true, y_pred, average="macro",    zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    print(f"  Accuracy    : {acc:.4f}")
    print(f"  F1 Macro    : {f1_macro:.4f}")
    print(f"  F1 Weighted : {f1_weighted:.4f}")

    return {"accuracy": acc, "f1_macro": f1_macro, "f1_weighted": f1_weighted}


def plot_attention_heatmap(
    text: str,
    attention_weights: np.ndarray,
    max_words: int = 20,
    model_name: str = "BiLSTM+Attention",
    save: bool = True,
) -> None:
    """
    Menampilkan heatmap attention weights untuk satu teks.

    Kata-kata dengan attention tinggi = lebih penting untuk prediksi.

    Args:
        text:               Teks yang sudah dibersihkan.
        attention_weights:  Array 1-D float, panjang = jumlah token.
        max_words:          Tampilkan maksimum max_words kata.
    """
    words   = text.split()[:max_words]
    weights = attention_weights[:len(words)]
    weights = weights / (weights.sum() + 1e-9)  # normalize

    fig, ax = plt.subplots(figsize=(max(8, len(words) * 0.55), 1.8))

    # Buat heatmap horizontal
    cmap = plt.cm.YlOrRd
    for i, (word, w) in enumerate(zip(words, weights)):
        color = cmap(float(w))
        rect  = mpatches.FancyBboxPatch(
            (i, 0), 0.9, 0.8,
            boxstyle="round,pad=0.05",
            facecolor=color, edgecolor="gray", linewidth=0.5,
        )
        ax.add_patch(rect)
        ax.text(i + 0.45, 0.4, word, ha="center", va="center", fontsize=9)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_array(weights)
    plt.colorbar(sm, ax=ax, orientation="horizontal", pad=0.4, label="Attention Weight")

    ax.set_xlim(0, len(words))
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(f"Attention Heatmap — {model_name}", fontsize=12, fontweight="bold", pad=10)
    plt.tight_layout()

    if save:
        path = os.path.join(PLOT_DIR, "attention_heatmap.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"💾 Plot disimpan: {path}")
    plt.show()


def compare_models(
    results: dict,
    save: bool = True,
) -> None:
    """
    Membuat bar chart perbandingan tiga model.

    Args:
        results: Dict {model_name: {"accuracy": ..., "f1_macro": ..., "f1_weighted": ...}}
    """
    model_names  = list(results.keys())
    accuracy     = [results[m]["accuracy"]    for m in model_names]
    f1_macro     = [results[m]["f1_macro"]    for m in model_names]
    f1_weighted  = [results[m]["f1_weighted"] for m in model_names]

    x     = np.arange(len(model_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width, accuracy,    width, label="Accuracy",    color="#4C72B0")
    bars2 = ax.bar(x,         f1_macro,    width, label="F1 Macro",    color="#DD8452")
    bars3 = ax.bar(x + width, f1_weighted, width, label="F1 Weighted", color="#55A868")

    for bars in (bars1, bars2, bars3):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                f"{h:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3), textcoords="offset points",
                ha="center", va="bottom", fontsize=8,
            )

    ax.set_xlabel("Model", fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Komparasi Model Deep Learning", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if save:
        path = os.path.join(PLOT_DIR, "model_comparison.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"💾 Plot disimpan: {path}")
    plt.show()


# ──────────────────────────────────────────────
# 🔍 INFERENSI TEKS TUNGGAL
# ──────────────────────────────────────────────

def predict_single_lstm(
    model: nn.Module,
    text: str,
    vocab,
    device: torch.device = DEVICE,
    return_attention: bool = False,
) -> dict:
    """
    Prediksi mental health status dari satu teks menggunakan BiLSTM.

    Args:
        model:            BiLSTMClassifier atau BiLSTMAttentionClassifier.
        text:             Teks yang sudah dibersihkan.
        vocab:            Objek Vocabulary.
        device:           Device.
        return_attention: Jika True dan model punya attention, kembalikan weights.

    Returns:
        Dict {"label": ..., "confidence": ..., "probabilities": {...},
              "attention_weights": ... (opsional)}
    """
    from config import MAX_LEN
    from preprocess import clean_text

    model.eval()
    cleaned = clean_text(text)
    indices = vocab.text_to_indices(cleaned, MAX_LEN)
    length  = max(min(len(cleaned.split()), MAX_LEN), 1)

    x       = torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(device)
    lengths = torch.tensor([length], dtype=torch.long).to(device)

    with torch.no_grad():
        if hasattr(model, "attention"):
            logits, attn = model(x, lengths)
        else:
            logits = model(x, lengths)
            attn   = None

    probs      = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    pred_idx   = int(probs.argmax())
    pred_label = LABEL_LIST[pred_idx]

    result = {
        "label":         pred_label,
        "confidence":    float(probs[pred_idx]),
        "probabilities": {LABEL_LIST[i]: float(probs[i]) for i in range(NUM_CLASSES)},
    }
    if return_attention and attn is not None:
        result["attention_weights"] = attn.squeeze(0).cpu().numpy()[:length]
    return result


def predict_single_bert(
    model: nn.Module,
    text: str,
    tokenizer,
    device: torch.device = DEVICE,
) -> dict:
    """
    Prediksi mental health status dari satu teks menggunakan DistilBERT.

    Returns:
        Dict {"label": ..., "confidence": ..., "probabilities": {...}}
    """
    from config import BERT_MAX_LEN

    model.eval()
    encoding = tokenizer(
        text,
        max_length=BERT_MAX_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids      = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        logits = model(input_ids, attention_mask)

    probs      = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    pred_idx   = int(probs.argmax())
    pred_label = LABEL_LIST[pred_idx]

    return {
        "label":         pred_label,
        "confidence":    float(probs[pred_idx]),
        "probabilities": {LABEL_LIST[i]: float(probs[i]) for i in range(NUM_CLASSES)},
    }
