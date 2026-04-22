"""
train.py — Pipeline Training DistilBERT (End-to-End)
======================================================
Jalankan:
    python train.py
    python train.py --lr 3e-5 --optimizer adamw --weight_decay 0.01 --scheduler linear

Alur:
    1. Download dataset
    2. Preprocessing & encode label
    3. Tokenisasi & buat DataLoaders
    4. Inisiasi model DistilBERT
    5. Training loop (dengan early stopping)
    6. Evaluasi pada test set
    7. Simpan model & tampilkan visualisasi
"""

import os
import copy
import time
import argparse

import torch
import torch.nn as nn
import wandb
from sklearn.metrics import precision_recall_fscore_support
from transformers import DistilBertTokenizerFast, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

from datareader import (
    download_dataset, load_and_clean, get_dataloaders,
    MODEL_DIR, BERT_MODEL, BERT_MAX_LEN, BATCH_SIZE,
)
from model import DistilBERTClassifier, count_parameters
from utils import set_seed, get_criterion, plot_training_curves, plot_confusion_matrix, print_report

# ──────────────────────────────────────────────
# HYPERPARAMETER
# ──────────────────────────────────────────────

EPOCHS    = 15
LR        = 1e-4 
PATIENCE  = 7     # early stopping: berhenti jika val_loss tidak membaik
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = os.path.join(MODEL_DIR, "distilbert.pt")
os.makedirs(MODEL_DIR, exist_ok=True)


# ──────────────────────────────────────────────
# TRAINING: satu epoch
# ──────────────────────────────────────────────

def train_one_epoch(model, dataloader, optimizer, criterion, scheduler=None):
    """
    Latih model selama satu epoch, kembalikan (loss, accuracy).

    Langkah per batch:
      1. Forward pass → logits
      2. Hitung loss
      3. Backward pass → hitung gradien
      4. Gradient clipping (mencegah exploding gradients)
      5. Update bobot
      6. Step scheduler per-batch (untuk linear/cosine)
    """
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for batch in dataloader:
        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels         = batch["label"].to(DEVICE)

        optimizer.zero_grad()                                   # reset gradien
        logits = model(input_ids, attention_mask)               # forward pass
        loss   = criterion(logits, labels)                      # hitung loss
        loss.backward()                                         # backward pass
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()                                        # update bobot
        if scheduler is not None:                               # step per-batch
            scheduler.step()

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += labels.size(0)
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

    return total_loss / total, correct / total, all_preds, all_labels


def compute_classification_metrics(y_true, y_pred):
    """Hitung metrik klasifikasi default untuk monitoring training."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# ──────────────────────────────────────────────
# EVALUASI: satu epoch
# ──────────────────────────────────────────────

def evaluate(model, dataloader, criterion):
    """
    Evaluasi model pada dataloader, kembalikan (loss, accuracy, preds, labels).
    Tidak ada backward pass — hanya forward pass dengan torch.no_grad().
    """
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels         = batch["label"].to(DEVICE)

            logits = model(input_ids, attention_mask)
            loss   = criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)
            preds       = logits.argmax(dim=1)
            correct    += (preds == labels).sum().item()
            total      += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return total_loss / total, correct / total, all_preds, all_labels


# ──────────────────────────────────────────────
# TRAINING LOOP LENGKAP
# ──────────────────────────────────────────────

def build_optimizer(model, optimizer_name: str, lr: float, weight_decay: float):
    """
    Buat optimizer berdasarkan nama.

    Pilihan:
      - 'adamw'  : AdamW  (default, direkomendasikan untuk fine-tuning BERT)
      - 'adam'   : Adam   (tanpa weight decay bawaan)
      - 'sgd'    : SGD dengan momentum 0.9
    """
    params = model.parameters()
    if optimizer_name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif optimizer_name == "muon":
        return torch.optim.Muon(params, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Optimizer '{optimizer_name}' tidak dikenal. Pilih: adamw | adam | sgd")


def build_scheduler(optimizer, scheduler_name: str, num_training_steps: int):
    """
    Buat learning rate scheduler berdasarkan nama.

    Pilihan:
      - 'none'    : tidak pakai scheduler, LR tetap
      - 'linear'  : turun linear dari LR → 0 (warmup 10%)
      - 'cosine'  : turun mengikuti kurva cosine (warmup 10%)
      - 'step'    : turun ×0.1 setiap 1 epoch (StepLR)
    """
    num_warmup = int(0.1 * num_training_steps)   # 10% langkah pertama = warmup

    if scheduler_name == "none":
        return None
    elif scheduler_name == "linear":
        return get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup,
            num_training_steps=num_training_steps,
        )
    elif scheduler_name == "cosine":
        return get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup,
            num_training_steps=num_training_steps,
        )
    elif scheduler_name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
    else:
        raise ValueError(f"Scheduler '{scheduler_name}' tidak dikenal. Pilih: none | linear | cosine | step")


def train_model(model, train_loader, val_loader,
                optimizer_name: str, lr: float,
                weight_decay: float, scheduler_name: str,
                use_wandb: bool = True):
    """
    Training loop dengan early stopping.

    Early stopping:
      - Simpan model terbaik berdasarkan val_loss terendah.
      - Hentikan training jika val_loss tidak membaik selama PATIENCE epoch.

    Args:
        model          : DistilBERTClassifier
        train_loader   : DataLoader training
        val_loader     : DataLoader validasi
        optimizer_name : 'adamw' | 'adam' | 'sgd'
        lr             : learning rate awal
        weight_decay   : regularisasi L2 pada optimizer
        scheduler_name : 'none' | 'linear' | 'cosine' | 'step'

    Returns:
        history: dict berisi list loss & accuracy per epoch
    """
    optimizer = build_optimizer(model, optimizer_name, lr, weight_decay)
    criterion = get_criterion()

    num_training_steps = EPOCHS * len(train_loader)
    scheduler = build_scheduler(optimizer, scheduler_name, num_training_steps)

    history = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": [],
        "train_precision": [], "val_precision": [],
        "train_recall": [], "val_recall": [],
        "train_f1": [], "val_f1": [],
        "train_lr": [],
    }
    best_val_loss    = float("inf")
    best_model_state = None
    patience_counter = 0

    print(f"\n{'='*55}")
    print(f"  Training DistilBERT | Device: {DEVICE}")
    print(f"  Epochs: {EPOCHS} | LR: {lr} | Patience: {PATIENCE}")
    print(f"  Optimizer: {optimizer_name} | Weight Decay: {weight_decay}")
    print(f"  Scheduler: {scheduler_name}")
    print(f"{'='*55}")

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        # linear/cosine: step per batch (sudah di dalam train_one_epoch)
        # step          : step per epoch (dilakukan di sini)
        per_batch_scheduler = scheduler if scheduler_name in ("linear", "cosine") else None
        train_loss, train_acc, train_preds, train_labels = train_one_epoch(
            model, train_loader, optimizer, criterion, per_batch_scheduler
        )
        if scheduler is not None and scheduler_name == "step":
            scheduler.step()   # StepLR: step per epoch
        val_loss, val_acc, val_preds, val_labels = evaluate(model, val_loader, criterion)

        train_metrics = compute_classification_metrics(train_labels, train_preds)
        val_metrics = compute_classification_metrics(val_labels, val_preds)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["train_precision"].append(train_metrics["precision"])
        history["val_precision"].append(val_metrics["precision"])
        history["train_recall"].append(train_metrics["recall"])
        history["val_recall"].append(val_metrics["recall"])
        history["train_f1"].append(train_metrics["f1"])
        history["val_f1"].append(val_metrics["f1"])
        current_lr = optimizer.param_groups[0]["lr"]
        history["train_lr"].append(current_lr)
        if use_wandb:
            wandb.log({
                "epoch": epoch,
                "train/loss": train_loss,
                "train/accuracy": train_acc,
                "train/precision": train_metrics["precision"],
                "train/recall": train_metrics["recall"],
                "train/f1": train_metrics["f1"],
                "val/loss": val_loss,
                "val/accuracy": val_acc,
                "val/precision": val_metrics["precision"],
                "val/recall": val_metrics["recall"],
                "val/f1": val_metrics["f1"],
                "train/lr": current_lr,
            })

        print(
            f"  Epoch {epoch}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f} | "
            f"Train F1: {train_metrics['f1']:.4f} | Val F1: {val_metrics['f1']:.4f} | "
            f"{time.time()-t0:.1f}s"
        )

        # Simpan jika ini model terbaik
        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            torch.save(best_model_state, MODEL_PATH)
            print(f"    => Model terbaik disimpan (val_loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\n  Early stopping di epoch {epoch}")
                break

        # Scheduler per-step (linear/cosine) sudah di-step di train_one_epoch;
        # scheduler per-epoch (step) sudah di-step di atas.

    # Muat kembali bobot model terbaik
    model.load_state_dict(best_model_state)
    print(f"\n  Selesai. Model tersimpan: {MODEL_PATH}")
    return history


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def parse_args():
    """
    Parse argumen command-line.

    Contoh penggunaan:
        python train.py
        python train.py --lr 3e-5 --optimizer adam --weight_decay 0.01 --scheduler cosine
    """
    parser = argparse.ArgumentParser(description="Fine-tuning DistilBERT untuk klasifikasi mental health")

    parser.add_argument(
        "--lr",
        type=float,
        default=5e-5,
        help="Learning rate awal (default: 2e-5)",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        choices=["adamw", "adam", "sgd", "muon"],
        help="Jenis optimizer: adamw | adam | sgd  (default: adamw)",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay / regularisasi L2 (default: 0.01)",
    )
    parser.add_argument(
        "--drop_out",
        type=float,
        default=0.2,
        help="Dropout sebelum classification head (default: 0.2)",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="cosine",
        choices=["none", "linear", "cosine", "step"],
        help="LR scheduler: none | linear | cosine | step  (default: cosine)",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="module-dl-minimal",
        help="Nama project Weights & Biases (default: module-dl-minimal)",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="Entity/team W&B (opsional)",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="Nama run W&B (opsional)",
    )
    parser.add_argument(
        "--wandb_off",
        action="store_true",
        help="Matikan logging ke W&B",
    )
    return parser.parse_args()


def main(
    lr: float           = 2e-5,
    optimizer: str      = "adamw",
    weight_decay: float = 0.01,
    drop_out: float     = 0.2,
    scheduler: str      = "cosine",
    wandb_project: str  = "module-dl-minimal",
    wandb_entity: str | None = None,
    wandb_run_name: str | None = None,
    wandb_off: bool = False,
):
    """
    Pipeline training end-to-end.

    Args:
        lr           : Learning rate awal (contoh: 2e-5, 3e-5).
        optimizer    : 'adamw' | 'adam' | 'sgd'.
        weight_decay : Regularisasi L2; mencegah overfitting (contoh: 0.01).
        drop_out     : Dropout sebelum classification head (contoh: 0.2, 0.1, 0.3).
        scheduler    : 'none' | 'linear' | 'cosine' | 'step'.
    """
    set_seed(42)
    print(f"Device: {DEVICE}\n")

    use_wandb = not wandb_off
    if use_wandb:
        # Pastikan file model/checkpoint lokal tidak ikut tersinkron ke W&B.
        os.environ["WANDB_IGNORE_GLOBS"] = "*.pt,*.pth,models/*,*.ckpt"
        print("Inisialisasi W&B login... jika belum login, buka link/browser yang muncul.")
        wandb.login(relogin=True)
        wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=wandb_run_name,
            save_code=False,
            config={
                "epochs": EPOCHS,
                "lr": lr,
                "optimizer": optimizer,
                "weight_decay": weight_decay,
                "drop_out": drop_out,
                "scheduler": scheduler,
                "batch_size": BATCH_SIZE,
                "max_len": BERT_MAX_LEN,
                "model": BERT_MODEL,
                "device": str(DEVICE),
            },
        )

    # 1. Download dataset
    csv_path = download_dataset()

    # 2. Preprocessing
    texts, labels, le = load_and_clean(csv_path)

    # 3. Tokenisasi & DataLoaders
    print(f"Memuat tokenizer: {BERT_MODEL}")
    tokenizer = DistilBertTokenizerFast.from_pretrained(BERT_MODEL)
    train_loader, val_loader, test_loader = get_dataloaders(texts, labels, tokenizer)

    # 4. Inisiasi model
    model = DistilBERTClassifier(dropout=drop_out).to(DEVICE)
    print(f"Jumlah parameter trainable: {count_parameters(model):,}")

    # 5. Training
    history = train_model(
        model, train_loader, val_loader,
        optimizer_name=optimizer,
        lr=lr,
        weight_decay=weight_decay,
        scheduler_name=scheduler,
        use_wandb=use_wandb,
    )

    # 6. Evaluasi pada test set
    print("\n--- Evaluasi Test Set ---")
    criterion = get_criterion()
    test_loss, test_acc, y_pred, y_true = evaluate(model, test_loader, criterion)
    test_metrics = compute_classification_metrics(y_true, y_pred)
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
    print_report(y_true, y_pred)

    if use_wandb:
        wandb.log({
            "test/loss": test_loss,
            "test/accuracy": test_acc,
            "test/precision": test_metrics["precision"],
            "test/recall": test_metrics["recall"],
            "test/f1": test_metrics["f1"],
        })

    # 7. Visualisasi
    plot_training_curves(history)
    plot_confusion_matrix(y_true, y_pred)

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    main(
        lr=args.lr,
        optimizer=args.optimizer,
        weight_decay=args.weight_decay,
        drop_out=args.drop_out,
        scheduler=args.scheduler,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
        wandb_off=args.wandb_off,
    )
