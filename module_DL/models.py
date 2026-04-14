"""
models.py — Definisi Arsitektur Deep Learning
==============================================
Tiga model untuk klasifikasi mental health status (7 kelas):

1. BiLSTMClassifier       — BiLSTM dua lapis dengan concat last hidden
2. BiLSTMAttentionClassifier — BiLSTM + mekanisme attention Bahdanau
3. DistilBERTClassifier   — DistilBERT fine-tuning dengan classification head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT,
    NUM_CLASSES, BERT_MODEL,
)


# ──────────────────────────────────────────────
# MODEL 1: BiLSTM
# ──────────────────────────────────────────────

class BiLSTMClassifier(nn.Module):
    """
    BiLSTM dua lapis untuk klasifikasi teks.

    Arsitektur:
        Input (token indices)
          → Embedding layer
          → Dropout
          → BiLSTM (2 layers)
          → Ambil hidden state terakhir (concat forward + backward)
          → Dropout
          → Fully Connected (hidden_dim*2 → num_classes)
    """

    def __init__(
        self,
        vocab_size: int   = VOCAB_SIZE,
        embed_dim: int    = EMBED_DIM,
        hidden_dim: int   = HIDDEN_DIM,
        num_layers: int   = NUM_LAYERS,
        num_classes: int  = NUM_CLASSES,
        dropout: float    = DROPOUT,
        pad_idx: int      = 0,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size, embed_dim, padding_idx=pad_idx
        )
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:       (batch, seq_len) — token indices
            lengths: (batch,) — panjang asli tiap urutan (sebelum padding)

        Returns:
            logits: (batch, num_classes)
        """
        embedded = self.dropout(self.embedding(x))   # (B, L, E)

        # Pack padded sequence agar LSTM tidak memproses padding
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (hidden, _) = self.lstm(packed)
        # hidden: (num_layers*2, B, H)

        # Ambil hidden state lapisan terakhir (forward + backward)
        last_hidden = torch.cat(
            (hidden[-2], hidden[-1]), dim=1
        )  # (B, H*2)

        out = self.dropout(last_hidden)
        logits = self.fc(out)
        return logits


# ──────────────────────────────────────────────
# MODEL 2: BiLSTM + Attention
# ──────────────────────────────────────────────

class BiLSTMAttentionClassifier(nn.Module):
    """
    BiLSTM dua lapis dengan mekanisme Attention (Bahdanau-style).

    Arsitektur:
        Input
          → Embedding + Dropout
          → BiLSTM (2 layers)  → semua hidden states H
          → Attention: score = tanh(W·H) → softmax → weighted sum c
          → Dropout
          → FC (hidden_dim*2 → num_classes)

    forward() mengembalikan (logits, attention_weights) sehingga
    attention_weights dapat divisualisasikan sebagai heatmap kata.
    """

    def __init__(
        self,
        vocab_size: int   = VOCAB_SIZE,
        embed_dim: int    = EMBED_DIM,
        hidden_dim: int   = HIDDEN_DIM,
        num_layers: int   = NUM_LAYERS,
        num_classes: int  = NUM_CLASSES,
        dropout: float    = DROPOUT,
        pad_idx: int      = 0,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size, embed_dim, padding_idx=pad_idx
        )
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.dropout   = nn.Dropout(dropout)
        self.fc        = nn.Linear(hidden_dim * 2, num_classes)

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x:       (batch, seq_len)
            lengths: (batch,)

        Returns:
            logits:             (batch, num_classes)
            attention_weights:  (batch, seq_len)  — untuk visualisasi
        """
        embedded = self.dropout(self.embedding(x))   # (B, L, E)

        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        output, _ = self.lstm(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(
            output, batch_first=True
        )  # (B, L, H*2)

        # ── Attention ──
        # score = tanh(W·h_t)  →  softmax
        scores  = torch.tanh(self.attention(output))  # (B, L, 1)
        scores  = scores.squeeze(-1)                  # (B, L)

        # Mask padding positions dengan nilai sangat negatif
        max_len = output.size(1)
        mask = torch.arange(max_len, device=x.device).unsqueeze(0) >= lengths.unsqueeze(1)
        scores = scores.masked_fill(mask, float("-inf"))

        attention_weights = F.softmax(scores, dim=1)  # (B, L)

        # Weighted sum
        context = (attention_weights.unsqueeze(-1) * output).sum(dim=1)  # (B, H*2)

        out    = self.dropout(context)
        logits = self.fc(out)
        return logits, attention_weights


# ──────────────────────────────────────────────
# MODEL 3: DistilBERT Fine-tuning
# ──────────────────────────────────────────────

class DistilBERTClassifier(nn.Module):
    """
    DistilBERT fine-tuning untuk klasifikasi teks.

    Arsitektur:
        Input (input_ids, attention_mask)
          → DistilBertModel (pre-trained)
          → Ambil [CLS] token representation
          → Dropout
          → FC (768 → num_classes)
    """

    def __init__(
        self,
        bert_model: str  = BERT_MODEL,
        num_classes: int = NUM_CLASSES,
        dropout: float   = DROPOUT,
    ):
        super().__init__()
        from transformers import DistilBertModel

        self.bert    = DistilBertModel.from_pretrained(bert_model)
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            input_ids:      (batch, seq_len)
            attention_mask: (batch, seq_len)

        Returns:
            logits: (batch, num_classes)
        """
        outputs      = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output   = outputs.last_hidden_state[:, 0, :]   # [CLS] token
        out          = self.dropout(cls_output)
        logits       = self.fc(out)
        return logits


# ──────────────────────────────────────────────
# HELPER: Hitung jumlah parameter
# ──────────────────────────────────────────────

def count_parameters(model: nn.Module) -> int:
    """Mengembalikan jumlah parameter yang dapat dilatih."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
