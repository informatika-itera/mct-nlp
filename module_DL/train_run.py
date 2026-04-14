"""
train_run.py — Skrip Terminal: Pipeline Pelatihan Lengkap
=========================================================
Menjalankan seluruh pipeline dari awal hingga akhir:
download → preprocess → build vocab → train BiLSTM →
train BiLSTM+Attention → fine-tune DistilBERT → compare → save

Jalankan dengan:
    python train_run.py

Pastikan virtual environment sudah aktif:
    source .venv/bin/activate
"""

import sys
import os
import time

import torch
from transformers import DistilBertTokenizerFast

from config import (
    DEVICE, SAMPLE_SIZE, LABEL_LIST,
    VOCAB_SIZE, MAX_LEN, LSTM_BATCH_SIZE, LSTM_LR, LSTM_EPOCHS, LSTM_PATIENCE,
    BERT_MAX_LEN, BERT_BATCH_SIZE, BERT_LR, BERT_EPOCHS, BERT_PATIENCE,
    BILSTM_MODEL_PATH, BILSTM_ATT_MODEL_PATH, DISTILBERT_MODEL_DIR,
    VOCAB_PATH, PLOT_DIR,
)
from download_data import download_dataset
from preprocess import load_and_clean, show_cleaning_examples
from dataset import Vocabulary, get_lstm_dataloaders, get_bert_dataloaders
from models import BiLSTMClassifier, BiLSTMAttentionClassifier, DistilBERTClassifier, count_parameters
from train import (
    set_seed, train_model,
    evaluate_lstm, evaluate_bert,
    get_criterion,
    plot_training_curves, plot_confusion_matrix,
    print_classification_report, compare_models,
)

import matplotlib.pyplot as plt
import seaborn as sns


# ──────────────────────────────────────────────
def section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

# ──────────────────────────────────────────────

def main():
    set_seed(42)
    print(f"🖥️  Device: {DEVICE}")
    print(f"📊 Sample size: {SAMPLE_SIZE:,}")
    os.makedirs(PLOT_DIR, exist_ok=True)

    # ── 1. DOWNLOAD ──────────────────────────────
    section("1. Download Dataset")
    csv_path = download_dataset()

    # ── 2. PREPROCESS ────────────────────────────
    section("2. Preprocessing")
    df = load_and_clean(csv_path, sample_size=SAMPLE_SIZE)
    show_cleaning_examples(df, n=3)

    # Label counts untuk weighted loss
    label_counts = df["label"].value_counts().to_dict()

    # Plot distribusi kelas
    plt.figure(figsize=(9, 4))
    df["label"].value_counts().sort_values().plot(kind="barh", color=sns.color_palette("Set2"))
    plt.title("Distribusi Kelas Mental Health Status")
    plt.xlabel("Jumlah Data")
    plt.tight_layout()
    dist_path = os.path.join(PLOT_DIR, "distribusi_kelas.png")
    plt.savefig(dist_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"💾 Plot distribusi disimpan: {dist_path}")

    # ── 3. BUILD VOCABULARY ──────────────────────
    section("3. Membangun Vocabulary")
    vocab = Vocabulary()
    vocab.build_vocab(df["cleaned_text"].tolist(), max_size=VOCAB_SIZE)
    vocab.save(VOCAB_PATH)

    # ── 4. DATALOADER (LSTM) ─────────────────────
    section("4. Membuat DataLoaders (LSTM)")
    train_loader, val_loader, test_loader = get_lstm_dataloaders(
        df, vocab, max_len=MAX_LEN, batch_size=LSTM_BATCH_SIZE
    )

    # ── 5. TRAIN BILSTM ───────────────────────────
    section("5. Training BiLSTM")
    bilstm = BiLSTMClassifier(vocab_size=len(vocab))
    print(f"   Parameter: {count_parameters(bilstm):,}")

    hist_bilstm = train_model(
        model=bilstm,
        train_loader=train_loader, val_loader=val_loader,
        model_type="lstm", save_path=BILSTM_MODEL_PATH,
        epochs=LSTM_EPOCHS, lr=LSTM_LR, patience=LSTM_PATIENCE,
        device=DEVICE, label_counts=label_counts,
    )
    plot_training_curves(hist_bilstm, "BiLSTM")

    # Evaluasi BiLSTM pada test set
    criterion = get_criterion(label_counts)
    _, _, preds_bilstm, labels_bilstm = evaluate_lstm(
        bilstm, test_loader, criterion, DEVICE
    )
    plot_confusion_matrix(labels_bilstm, preds_bilstm, "BiLSTM")
    metrics_bilstm = print_classification_report(labels_bilstm, preds_bilstm, "BiLSTM")
    metrics_bilstm["training_time_min"] = hist_bilstm["total_time"] / 60

    # ── 6. TRAIN BILSTM + ATTENTION ───────────────
    section("6. Training BiLSTM + Attention")
    bilstm_att = BiLSTMAttentionClassifier(vocab_size=len(vocab))
    print(f"   Parameter: {count_parameters(bilstm_att):,}")

    hist_bilstm_att = train_model(
        model=bilstm_att,
        train_loader=train_loader, val_loader=val_loader,
        model_type="lstm_att", save_path=BILSTM_ATT_MODEL_PATH,
        epochs=LSTM_EPOCHS, lr=LSTM_LR, patience=LSTM_PATIENCE,
        device=DEVICE, label_counts=label_counts,
    )
    plot_training_curves(hist_bilstm_att, "BiLSTM+Attention")

    _, _, preds_att, labels_att = evaluate_lstm(
        bilstm_att, test_loader, criterion, DEVICE
    )
    plot_confusion_matrix(labels_att, preds_att, "BiLSTM+Attention")
    metrics_att = print_classification_report(labels_att, preds_att, "BiLSTM+Attention")
    metrics_att["training_time_min"] = hist_bilstm_att["total_time"] / 60

    # ── 7. FINE-TUNE DISTILBERT ───────────────────
    section("7. Fine-tuning DistilBERT")
    print("   Memuat tokenizer DistilBERT...")
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    train_bert, val_bert, test_bert = get_bert_dataloaders(
        df, tokenizer, max_len=BERT_MAX_LEN, batch_size=BERT_BATCH_SIZE
    )

    distilbert = DistilBERTClassifier()
    print(f"   Parameter: {count_parameters(distilbert):,}")

    os.makedirs(DISTILBERT_MODEL_DIR, exist_ok=True)
    bert_save_path = os.path.join(DISTILBERT_MODEL_DIR, "distilbert.pt")

    hist_bert = train_model(
        model=distilbert,
        train_loader=train_bert, val_loader=val_bert,
        model_type="bert", save_path=bert_save_path,
        epochs=BERT_EPOCHS, lr=BERT_LR, patience=BERT_PATIENCE,
        device=DEVICE, label_counts=label_counts,
    )
    plot_training_curves(hist_bert, "DistilBERT")

    _, _, preds_bert, labels_bert = evaluate_bert(
        distilbert, test_bert, get_criterion(label_counts), DEVICE
    )
    plot_confusion_matrix(labels_bert, preds_bert, "DistilBERT")
    metrics_bert = print_classification_report(labels_bert, preds_bert, "DistilBERT")
    metrics_bert["training_time_min"] = hist_bert["total_time"] / 60

    # ── 8. KOMPARASI ──────────────────────────────
    section("8. Komparasi Model")
    all_results = {
        "BiLSTM":           metrics_bilstm,
        "BiLSTM+Attention": metrics_att,
        "DistilBERT":       metrics_bert,
    }
    compare_models(all_results)

    print("\n📊 Ringkasan Akhir:")
    print(f"{'Model':<22} {'Accuracy':>10} {'F1 Macro':>10} {'F1 Weighted':>12} {'Waktu (mnt)':>12}")
    print("-" * 70)
    for name, m in all_results.items():
        print(
            f"{name:<22} {m['accuracy']:>10.4f} {m['f1_macro']:>10.4f} "
            f"{m['f1_weighted']:>12.4f} {m['training_time_min']:>12.1f}"
        )

    print("\n✅ Pipeline selesai! Semua model tersimpan di folder models/")


if __name__ == "__main__":
    main()
