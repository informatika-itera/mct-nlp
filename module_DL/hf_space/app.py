"""
app.py — Hugging Face Space: Mental Health Sentiment Analyzer
=============================================================
Gradio app dengan tiga model pilihan:
- BiLSTM
- BiLSTM + Attention (dengan visualisasi attention)
- DistilBERT

Upload model ke HF Hub terlebih dahulu:
    bilstm.pt, bilstm_attention.pt, distilbert/distilbert.pt, vocab.json

Jalankan lokal:
    python app.py
"""

import os
import re
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ──────────────────────────────────────────────
# Konfigurasi
# ──────────────────────────────────────────────
LABEL_LIST = [
    "Normal",
    "Depression",
    "Suicidal",
    "Anxiety",
    "Stress",
    "Bipolar",
    "Personality Disorder",
]
NUM_CLASSES  = len(LABEL_LIST)
VOCAB_SIZE   = 20_000
EMBED_DIM    = 128
HIDDEN_DIM   = 256
NUM_LAYERS   = 2
DROPOUT      = 0.3
MAX_LEN      = 128
BERT_MAX_LEN = 128
BERT_MODEL   = "distilbert-base-uncased"

DEVICE = torch.device("cpu")   # HF free tier: CPU only

# ──────────────────────────────────────────────
# Text cleaning (duplikasi agar app mandiri)
# ──────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Membersihkan teks: lowercase → hapus URL/HTML/non-alpha → strip."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ──────────────────────────────────────────────
# Vocabulary (duplikasi agar app mandiri)
# ──────────────────────────────────────────────

class Vocabulary:
    PAD_IDX = 0
    UNK_IDX = 1

    def __init__(self):
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}

    def load(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            self.word2idx = json.load(f)

    def text_to_indices(self, text: str, max_len: int = MAX_LEN) -> list:
        tokens  = text.split()[:max_len]
        indices = [self.word2idx.get(t, self.UNK_IDX) for t in tokens]
        indices += [self.PAD_IDX] * (max_len - len(indices))
        return indices

    def __len__(self):
        return len(self.word2idx)


# ──────────────────────────────────────────────
# Model definitions (duplikasi agar app mandiri)
# ──────────────────────────────────────────────

class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM,
                 hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS,
                 num_classes=NUM_CLASSES, dropout=DROPOUT, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, bidirectional=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x, lengths):
        embedded = self.dropout(self.embedding(x))
        packed   = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed)
        last_hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return self.fc(self.dropout(last_hidden))


class BiLSTMAttentionClassifier(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM,
                 hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS,
                 num_classes=NUM_CLASSES, dropout=DROPOUT, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, bidirectional=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.dropout   = nn.Dropout(dropout)
        self.fc        = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x, lengths):
        embedded = self.dropout(self.embedding(x))
        packed   = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        output, _ = self.lstm(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

        scores = torch.tanh(self.attention(output)).squeeze(-1)
        max_len = output.size(1)
        mask   = torch.arange(max_len).unsqueeze(0) >= lengths.unsqueeze(1)
        scores = scores.masked_fill(mask, float("-inf"))
        attn   = F.softmax(scores, dim=1)

        context = (attn.unsqueeze(-1) * output).sum(dim=1)
        return self.fc(self.dropout(context)), attn


class DistilBERTClassifier(nn.Module):
    def __init__(self, bert_model=BERT_MODEL, num_classes=NUM_CLASSES, dropout=DROPOUT):
        super().__init__()
        from transformers import DistilBertModel
        self.bert    = DistilBertModel.from_pretrained(bert_model)
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        return self.fc(self.dropout(cls))


# ──────────────────────────────────────────────
# Load models & vocab
# ──────────────────────────────────────────────

def _load_models():
    """Muat semua model dari file yang tersedia."""
    models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    vocab_path  = os.path.join(models_dir, "vocab.json")

    vocab = Vocabulary()
    if os.path.exists(vocab_path):
        vocab.load(vocab_path)
    else:
        print(f"⚠️ vocab.json tidak ditemukan di {vocab_path}")

    loaded = {}

    # BiLSTM
    bilstm_path = os.path.join(models_dir, "bilstm.pt")
    if os.path.exists(bilstm_path):
        m = BiLSTMClassifier(vocab_size=len(vocab))
        m.load_state_dict(torch.load(bilstm_path, map_location=DEVICE))
        m.eval()
        loaded["BiLSTM"] = m
        print("✅ BiLSTM dimuat")

    # BiLSTM+Attention
    att_path = os.path.join(models_dir, "bilstm_attention.pt")
    if os.path.exists(att_path):
        m = BiLSTMAttentionClassifier(vocab_size=len(vocab))
        m.load_state_dict(torch.load(att_path, map_location=DEVICE))
        m.eval()
        loaded["BiLSTM+Attention"] = m
        print("✅ BiLSTM+Attention dimuat")

    # DistilBERT
    bert_path = os.path.join(models_dir, "distilbert", "distilbert.pt")
    if os.path.exists(bert_path):
        try:
            m = DistilBERTClassifier()
            m.load_state_dict(torch.load(bert_path, map_location=DEVICE))
            m.eval()
            loaded["DistilBERT"] = m
            print("✅ DistilBERT dimuat")
        except Exception as e:
            print(f"⚠️ Gagal muat DistilBERT: {e}")

    return loaded, vocab


MODELS, VOCAB = _load_models()

# Tokenizer untuk BERT (lazy load)
_TOKENIZER = None

def get_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        from transformers import DistilBertTokenizerFast
        _TOKENIZER = DistilBertTokenizerFast.from_pretrained(BERT_MODEL)
    return _TOKENIZER


# ──────────────────────────────────────────────
# Attention heatmap sebagai gambar
# ──────────────────────────────────────────────

def make_attention_figure(text: str, attention_weights: np.ndarray) -> plt.Figure:
    """Membuat figure matplotlib dari attention heatmap."""
    words   = text.split()[:20]
    weights = attention_weights[:len(words)]
    weights = weights / (weights.sum() + 1e-9)

    n   = len(words)
    fig = plt.figure(figsize=(max(8, n * 0.7), 2.2))
    ax  = fig.add_subplot(111)
    cmap = plt.cm.YlOrRd

    for i, (word, w) in enumerate(zip(words, weights)):
        color = cmap(float(w))
        rect  = mpatches.FancyBboxPatch(
            (i, 0.1), 0.88, 0.8,
            boxstyle="round,pad=0.05",
            facecolor=color, edgecolor="gray", linewidth=0.5,
        )
        ax.add_patch(rect)
        ax.text(i + 0.44, 0.5, word, ha="center", va="center",
                fontsize=9, wrap=True)

    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_array(weights)
    plt.colorbar(sm, ax=ax, orientation="horizontal", pad=0.35, label="Attention Weight")

    ax.set_xlim(0, n)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Attention Heatmap — kata-kata yang diperhatikan model",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    return fig


# ──────────────────────────────────────────────
# Fungsi prediksi utama
# ──────────────────────────────────────────────

def predict(text: str, model_name: str):
    """
    Prediksi mental health status dari teks input.

    Returns:
        label, confidence_dict, attention_figure_or_none
    """
    if not text or not text.strip():
        return "—", {}, None

    if model_name not in MODELS:
        return f"Model '{model_name}' belum dimuat.", {}, None

    model   = MODELS[model_name]
    cleaned = clean_text(text)

    if model_name in ("BiLSTM", "BiLSTM+Attention"):
        indices = VOCAB.text_to_indices(cleaned, MAX_LEN)
        length  = max(min(len(cleaned.split()), MAX_LEN), 1)
        x       = torch.tensor(indices, dtype=torch.long).unsqueeze(0)
        lengths = torch.tensor([length], dtype=torch.long)

        with torch.no_grad():
            if model_name == "BiLSTM+Attention":
                logits, attn = model(x, lengths)
                attn_np = attn.squeeze(0).numpy()[:length]
            else:
                logits  = model(x, lengths)
                attn_np = None

        probs      = torch.softmax(logits, dim=1).squeeze(0).numpy()
        pred_idx   = int(probs.argmax())
        pred_label = LABEL_LIST[pred_idx]

        conf_dict = {LABEL_LIST[i]: round(float(probs[i]), 4)
                     for i in range(NUM_CLASSES)}

        attn_fig = None
        if attn_np is not None and len(cleaned.split()) > 0:
            attn_fig = make_attention_figure(cleaned, attn_np)

        return pred_label, conf_dict, attn_fig

    else:  # DistilBERT
        tokenizer = get_tokenizer()
        enc = tokenizer(
            text, max_length=BERT_MAX_LEN,
            padding="max_length", truncation=True, return_tensors="pt",
        )
        with torch.no_grad():
            logits = model(enc["input_ids"], enc["attention_mask"])

        probs      = torch.softmax(logits, dim=1).squeeze(0).numpy()
        pred_idx   = int(probs.argmax())
        pred_label = LABEL_LIST[pred_idx]
        conf_dict  = {LABEL_LIST[i]: round(float(probs[i]), 4)
                      for i in range(NUM_CLASSES)}

        return pred_label, conf_dict, None


# ──────────────────────────────────────────────
# Gradio Interface
# ──────────────────────────────────────────────

CONTOH_TEKS = [
    "I have been feeling hopeless and worthless for weeks. I don't see any reason to continue.",
    "Everything makes me nervous. I can't stop worrying about what might happen next.",
    "Today was a wonderful day! I feel grateful and at peace with my life.",
    "My mood swings are uncontrollable. One moment I'm euphoric, the next I'm in complete despair.",
    "I hate myself. I wish I wasn't here anymore. Nobody would miss me.",
    "Work pressure is overwhelming and I can't seem to relax or sleep properly.",
]

available_models = list(MODELS.keys()) if MODELS else ["BiLSTM", "BiLSTM+Attention", "DistilBERT"]

with gr.Blocks(theme=gr.themes.Soft(), title="Mental Health Sentiment Analyzer") as demo:
    gr.Markdown("""
    # 🧠 Mental Health Sentiment Analyzer
    Klasifikasi status kesehatan mental dari teks menggunakan tiga model deep learning.

    **7 Kelas:** Normal · Depression · Suicidal · Anxiety · Stress · Bipolar · Personality Disorder

    > ⚠️ **Disclaimer:** Ini adalah alat demonstrasi akademis dan **bukan alat diagnosis medis**.
    """)

    with gr.Row():
        with gr.Column(scale=2):
            model_choice = gr.Dropdown(
                choices=available_models,
                value=available_models[-1] if available_models else "DistilBERT",
                label="Pilih Model",
                info="DistilBERT umumnya paling akurat; BiLSTM+Attention menampilkan heatmap.",
            )
            text_input = gr.Textbox(
                lines=4,
                placeholder="Masukkan teks di sini...",
                label="Teks Input",
            )
            submit_btn   = gr.Button("🔍 Analisis", variant="primary")
            example_comp = gr.Examples(
                examples=CONTOH_TEKS,
                inputs=text_input,
                label="Contoh Teks",
            )

        with gr.Column(scale=3):
            label_output  = gr.Label(label="Prediksi Status")
            probs_output  = gr.Label(label="Confidence per Kelas", num_top_classes=7)
            attn_output   = gr.Plot(label="Attention Heatmap (BiLSTM+Attention saja)")

    submit_btn.click(
        fn=predict,
        inputs=[text_input, model_choice],
        outputs=[label_output, probs_output, attn_output],
    )

    gr.Markdown("""
    ---
    **Model:**
    - 🔵 **BiLSTM** — Bidirectional LSTM, embedding train from scratch
    - 🟡 **BiLSTM+Attention** — BiLSTM + mekanisme attention (dengan visualisasi)
    - 🟢 **DistilBERT** — Fine-tuned DistilBERT (akurasi tertinggi)

    _Workshop PBA — ITERA 2026_
    """)


if __name__ == "__main__":
    demo.launch()
