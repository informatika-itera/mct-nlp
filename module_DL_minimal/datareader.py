"""
datareader.py — Download, Preprocessing, Dataset, dan DataLoader
=================================================================
Alur data untuk DistilBERT:
  1. download_dataset()  — unduh CSV dari Kaggle via kagglehub
  2. load_and_clean()    — baca CSV, bersihkan teks, encode label
  3. BERTDataset         — PyTorch Dataset: tokenisasi per sampel
  4. get_dataloaders()   — bagi data train/val/test, buat DataLoader
"""

import os
import re
import shutil

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import DistilBertTokenizerFast

# ──────────────────────────────────────────────
# KONSTANTA
# ──────────────────────────────────────────────

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
PLOT_DIR   = os.path.join(BASE_DIR, "plots")

CSV_PATH = os.path.join(DATA_DIR, "Combined Data.csv")

TEXT_COL  = "statement"
LABEL_COL = "status"

LABEL_LIST = [
    "Normal", "Depression", "Suicidal",
    "Anxiety", "Stress", "Bipolar", "Personality Disorder",
]
NUM_CLASSES = len(LABEL_LIST)

BERT_MODEL   = "distilbert-base-uncased"
BERT_MAX_LEN = 128
BATCH_SIZE   = 64
TEST_SIZE    = 0.10
VAL_SIZE     = 0.10
SAMPLE_SIZE  = 5_000    # None = pakai semua data
RANDOM_SEED  = 42

# Buat folder jika belum ada
for _dir in (DATA_DIR, MODEL_DIR, PLOT_DIR):
    os.makedirs(_dir, exist_ok=True)


# ──────────────────────────────────────────────
# 1. DOWNLOAD DATASET
# ──────────────────────────────────────────────

def download_dataset() -> str:
    """
    Salin dataset dari folder module_DL/data ke folder data/ lokal.

    Returns:
        Path ke file CSV lokal.
    """
    if os.path.exists(CSV_PATH):
        print(f"Dataset sudah ada: {CSV_PATH}")
        return CSV_PATH

    # Cari CSV dari folder module_DL yang satu level di atas
    source_csv = os.path.join(BASE_DIR, "..", "module_DL", "data", "Combined Data.csv")
    source_csv = os.path.abspath(source_csv)

    if not os.path.exists(source_csv):
        raise FileNotFoundError(
            f"File sumber tidak ditemukan: {source_csv}\n"
            "Pastikan folder module_DL/data berisi 'Combined Data.csv'."
        )

    shutil.copy2(source_csv, CSV_PATH)
    print(f"Dataset disalin dari: {source_csv}")
    print(f"Dataset tersedia di:  {CSV_PATH}")
    return CSV_PATH


# ──────────────────────────────────────────────
# 2. PREPROCESSING
# ──────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Lowercase → hapus URL & HTML → hapus non-alpha → strip spasi."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_and_clean(csv_path: str) -> tuple[list, list, LabelEncoder]:
    """
    Baca CSV, bersihkan teks, encode label.

    Returns:
        texts  : list string teks bersih
        labels : list integer label
        le     : LabelEncoder (digunakan untuk decode prediksi)
    """
    print(f"Membaca: {csv_path}")
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    df = df[[TEXT_COL, LABEL_COL]].copy()
    df.rename(columns={TEXT_COL: "text", LABEL_COL: "label"}, inplace=True)
    df.dropna(inplace=True)

    # Normalisasi nama label
    label_map = {l.lower(): l for l in LABEL_LIST}
    df["label"] = df["label"].astype(str).str.strip().apply(
        lambda x: label_map.get(x.lower(), x)
    )
    df = df[df["label"].isin(LABEL_LIST)].reset_index(drop=True)
    print(f"Total data: {len(df):,} baris")

    # Sampling stratified (opsional, untuk mempercepat demo)
    if SAMPLE_SIZE and SAMPLE_SIZE < len(df):
        df, _ = train_test_split(
            df, train_size=SAMPLE_SIZE,
            stratify=df["label"], random_state=RANDOM_SEED,
        )
        df = df.reset_index(drop=True)
        print(f"Setelah sampling: {len(df):,} baris")

    # Bersihkan teks
    df["text"] = df["text"].apply(clean_text)
    df = df[df["text"].str.len() > 0].reset_index(drop=True)

    # Encode label ke integer
    le = LabelEncoder()
    le.fit(LABEL_LIST)
    labels = le.transform(df["label"]).tolist()
    texts  = df["text"].tolist()

    print(f"Distribusi kelas:\n{df['label'].value_counts().to_string()}\n")
    return texts, labels, le


# ──────────────────────────────────────────────
# 3. PYTORCH DATASET
# ──────────────────────────────────────────────

class BERTDataset(Dataset):
    """
    Dataset PyTorch untuk DistilBERT.
    Tokenisasi dilakukan saat __getitem__ dipanggil (lazy).
    """

    def __init__(self, texts: list, labels: list, tokenizer, max_len: int = BERT_MAX_LEN):
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      encoding["input_ids"].squeeze(0),       # (max_len,)
            "attention_mask": encoding["attention_mask"].squeeze(0),  # (max_len,)
            "label":          torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ──────────────────────────────────────────────
# 4. DATALOADER BUILDER
# ──────────────────────────────────────────────

def get_dataloaders(
    texts: list,
    labels: list,
    tokenizer,
    batch_size: int = BATCH_SIZE,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Split data menjadi train/val/test lalu buat DataLoaders.

    Split stratified: 80% train / 10% val / 10% test.

    Returns:
        train_loader, val_loader, test_loader
    """
    # Pisahkan test set dulu
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        texts, labels,
        test_size=TEST_SIZE,
        stratify=labels,
        random_state=RANDOM_SEED,
    )

    # Pisahkan val dari sisa train
    val_ratio = VAL_SIZE / (1 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=val_ratio,
        stratify=y_trainval,
        random_state=RANDOM_SEED,
    )

    print(f"Split data — Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")

    train_ds = BERTDataset(X_train, y_train, tokenizer)
    val_ds   = BERTDataset(X_val,   y_val,   tokenizer)
    test_ds  = BERTDataset(X_test,  y_test,  tokenizer)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
