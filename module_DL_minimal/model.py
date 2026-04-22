"""
model.py — Definisi Model DistilBERT untuk Klasifikasi Teks
============================================================
Arsitektur:
    Input (input_ids, attention_mask)
      → DistilBertModel (pre-trained, 6 encoder layers)
      → Ambil output [CLS] token  →  shape: (batch, 768)
      → Dropout
      → Linear(768 → num_classes)
      → Logits  (batch, num_classes)
"""

import torch
import torch.nn as nn
from transformers import DistilBertModel

from datareader import BERT_MODEL, NUM_CLASSES


class DistilBERTClassifier(nn.Module):
    """
    DistilBERT fine-tuning dengan satu classification head.

    Parameter utama yang di-fine-tune:
    - Semua bobot DistilBERT (transformer layers)
    - Bobot Linear layer klasifikasi

    Args:
        bert_model  : Nama model HuggingFace (default: distilbert-base-uncased)
        num_classes : Jumlah kelas output
        dropout     : Dropout sebelum linear layer
    """

    def __init__(
        self,
        bert_model: str  = BERT_MODEL,
        num_classes: int = NUM_CLASSES,
        dropout: float   = 0.2,
    ):
        super().__init__()
        self.bert    = DistilBertModel.from_pretrained(bert_model)
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(self.bert.config.hidden_size, num_classes)
        # hidden_size = 768 untuk distilbert-base-uncased

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            input_ids:      (batch, seq_len)  — token IDs dari tokenizer
            attention_mask: (batch, seq_len)  — 1 = token asli, 0 = padding

        Returns:
            logits: (batch, num_classes)  — belum di-softmax
        """
        # outputs.last_hidden_state: (batch, seq_len, 768)
        outputs    = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # [CLS] token berada di posisi index 0 setiap urutan
        cls_output = outputs.last_hidden_state[:, 0, :]   # (batch, 768)

        out    = self.dropout(cls_output)
        logits = self.fc(out)                              # (batch, num_classes)
        return logits


def count_parameters(model: nn.Module) -> int:
    """Hitung jumlah parameter yang dapat dilatih."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
