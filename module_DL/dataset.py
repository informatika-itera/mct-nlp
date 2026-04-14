"""
dataset.py — Vocabulary, PyTorch Dataset, dan DataLoader Builder
================================================================
Menyediakan pipeline data untuk BiLSTM dan DistilBERT:
- Vocabulary : membangun vocab dari teks, konversi token→index.
- MentalHealthDataset : Dataset PyTorch untuk BiLSTM.
- BERTDataset          : Dataset PyTorch untuk DistilBERT.
- get_dataloaders()    : Membuat train/val/test DataLoaders.
"""

import json
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split

from config import (
    VOCAB_SIZE, MAX_LEN, BERT_MAX_LEN,
    LSTM_BATCH_SIZE, BERT_BATCH_SIZE,
    TEST_SIZE, VAL_SIZE, RANDOM_SEED,
    VOCAB_PATH,
)


# ──────────────────────────────────────────────
# 📖 VOCABULARY (untuk BiLSTM)
# ──────────────────────────────────────────────

class Vocabulary:
    """
    Membangun kamus kata dari corpus dan mengonversi
    teks menjadi urutan indeks integer.

    Token khusus:
    - <PAD>  : padding (index 0)
    - <UNK>  : kata tidak dikenal (index 1)
    """

    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    PAD_IDX   = 0
    UNK_IDX   = 1

    def __init__(self):
        self.word2idx: dict[str, int] = {
            self.PAD_TOKEN: self.PAD_IDX,
            self.UNK_TOKEN: self.UNK_IDX,
        }
        self.idx2word: dict[int, str] = {
            self.PAD_IDX: self.PAD_TOKEN,
            self.UNK_IDX: self.UNK_TOKEN,
        }

    def build_vocab(self, texts: list[str], max_size: int = VOCAB_SIZE) -> None:
        """
        Membangun vocabulary dari daftar teks.

        Args:
            texts:    Daftar string teks yang sudah dibersihkan.
            max_size: Jumlah maksimum kata (tidak termasuk PAD & UNK).
        """
        counter: Counter = Counter()
        for text in texts:
            counter.update(text.split())

        # Ambil max_size kata paling sering
        most_common = counter.most_common(max_size - 2)
        for word, _ in most_common:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx]  = word

        print(f"📖 Vocabulary dibangun: {len(self.word2idx):,} kata")

    def text_to_indices(self, text: str, max_len: int = MAX_LEN) -> list[int]:
        """
        Mengonversi teks menjadi list indeks integer (dengan padding/truncation).

        Args:
            text:    Teks yang sudah dibersihkan.
            max_len: Panjang maksimum urutan.

        Returns:
            List integer dengan panjang tepat max_len.
        """
        tokens = text.split()[:max_len]
        indices = [self.word2idx.get(t, self.UNK_IDX) for t in tokens]
        # Padding
        indices += [self.PAD_IDX] * (max_len - len(indices))
        return indices

    def __len__(self) -> int:
        return len(self.word2idx)

    def save(self, path: str = VOCAB_PATH) -> None:
        """Simpan vocabulary ke file JSON."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.word2idx, f, ensure_ascii=False, indent=2)
        print(f"💾 Vocabulary disimpan: {path}")

    @classmethod
    def load(cls, path: str = VOCAB_PATH) -> "Vocabulary":
        """Muat vocabulary dari file JSON."""
        vocab = cls()
        with open(path, "r", encoding="utf-8") as f:
            vocab.word2idx = json.load(f)
        vocab.idx2word = {v: k for k, v in vocab.word2idx.items()}
        print(f"📂 Vocabulary dimuat: {len(vocab):,} kata")
        return vocab


# ──────────────────────────────────────────────
# 📦 DATASET — BiLSTM
# ──────────────────────────────────────────────

class MentalHealthDataset(Dataset):
    """Dataset PyTorch untuk model BiLSTM."""

    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        vocab: Vocabulary,
        max_len: int = MAX_LEN,
    ):
        self.texts   = texts
        self.labels  = labels
        self.vocab   = vocab
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        indices = self.vocab.text_to_indices(self.texts[idx], self.max_len)
        length  = min(len(self.texts[idx].split()), self.max_len)
        length  = max(length, 1)  # minimal 1

        return (
            torch.tensor(indices, dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.long),
            torch.tensor(length, dtype=torch.long),
        )


# ──────────────────────────────────────────────
# 📦 DATASET — DistilBERT
# ──────────────────────────────────────────────

class BERTDataset(Dataset):
    """Dataset PyTorch untuk model DistilBERT."""

    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        tokenizer,
        max_len: int = BERT_MAX_LEN,
    ):
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label":          torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ──────────────────────────────────────────────
# 🔄 DATALOADER BUILDER
# ──────────────────────────────────────────────

def get_lstm_dataloaders(
    df: pd.DataFrame,
    vocab: Vocabulary,
    max_len: int = MAX_LEN,
    batch_size: int = LSTM_BATCH_SIZE,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Membuat train/val/test DataLoaders untuk BiLSTM.

    Split: 80% train / 10% val / 10% test (stratified).

    Args:
        df:         DataFrame dengan kolom 'cleaned_text' dan 'label_encoded'.
        vocab:      Objek Vocabulary yang sudah dibangun.
        max_len:    Panjang maksimum urutan.
        batch_size: Ukuran batch.

    Returns:
        Tuple (train_loader, val_loader, test_loader)
    """
    texts  = df["cleaned_text"].tolist()
    labels = df["label_encoded"].tolist()

    # Split train vs temp (val+test)
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        texts, labels,
        test_size=(TEST_SIZE + VAL_SIZE),
        stratify=labels,
        random_state=RANDOM_SEED,
    )

    # Split val vs test dari temp
    relative_val = VAL_SIZE / (TEST_SIZE + VAL_SIZE)
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels,
        test_size=(1 - relative_val),
        stratify=temp_labels,
        random_state=RANDOM_SEED,
    )

    train_ds = MentalHealthDataset(train_texts, train_labels, vocab, max_len)
    val_ds   = MentalHealthDataset(val_texts,   val_labels,   vocab, max_len)
    test_ds  = MentalHealthDataset(test_texts,  test_labels,  vocab, max_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    print(f"📦 LSTM DataLoaders — Train: {len(train_ds):,} | "
          f"Val: {len(val_ds):,} | Test: {len(test_ds):,}")
    return train_loader, val_loader, test_loader


def get_bert_dataloaders(
    df: pd.DataFrame,
    tokenizer,
    max_len: int = BERT_MAX_LEN,
    batch_size: int = BERT_BATCH_SIZE,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Membuat train/val/test DataLoaders untuk DistilBERT.

    Args:
        df:         DataFrame dengan kolom 'text' dan 'label_encoded'.
        tokenizer:  HuggingFace tokenizer.
        max_len:    Panjang maksimum token.
        batch_size: Ukuran batch.

    Returns:
        Tuple (train_loader, val_loader, test_loader)
    """
    texts  = df["text"].tolist()
    labels = df["label_encoded"].tolist()

    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        texts, labels,
        test_size=(TEST_SIZE + VAL_SIZE),
        stratify=labels,
        random_state=RANDOM_SEED,
    )

    relative_val = VAL_SIZE / (TEST_SIZE + VAL_SIZE)
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels,
        test_size=(1 - relative_val),
        stratify=temp_labels,
        random_state=RANDOM_SEED,
    )

    train_ds = BERTDataset(train_texts, train_labels, tokenizer, max_len)
    val_ds   = BERTDataset(val_texts,   val_labels,   tokenizer, max_len)
    test_ds  = BERTDataset(test_texts,  test_labels,  tokenizer, max_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    print(f"📦 BERT DataLoaders — Train: {len(train_ds):,} | "
          f"Val: {len(val_ds):,} | Test: {len(test_ds):,}")
    return train_loader, val_loader, test_loader
