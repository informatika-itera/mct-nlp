"""
config.py — Konfigurasi & Konstanta untuk Workshop Deep Learning
================================================================
Berisi path, hyperparameter model, dan label mapping untuk
klasifikasi mental health status (7 kelas).
"""

import os
import torch

# ──────────────────────────────────────────────
# 📁 PATH
# ──────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
PLOT_DIR  = os.path.join(BASE_DIR, "plots")

RAW_CSV = os.path.join(DATA_DIR, "Combined Data.csv")

# Buat folder kalau belum ada
os.makedirs(DATA_DIR,  exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR,  exist_ok=True)

# ──────────────────────────────────────────────
# 📋 DATASET
# ──────────────────────────────────────────────
KAGGLE_DATASET   = "suchintikasarkar/sentiment-analysis-for-mental-health"
EXPECTED_FILENAME = "Combined Data.csv"

TEXT_COL  = "statement"
LABEL_COL = "status"

LABEL_LIST = [
    "Normal",
    "Depression",
    "Suicidal",
    "Anxiety",
    "Stress",
    "Bipolar",
    "Personality Disorder",
]
NUM_CLASSES = len(LABEL_LIST)

# ──────────────────────────────────────────────
# ⚙️  UMUM
# ──────────────────────────────────────────────
RANDOM_SEED = 42
SAMPLE_SIZE = 15_000      # None = pakai semua data
TEST_SIZE   = 0.10        # 80 / 10 / 10  train / val / test
VAL_SIZE    = 0.10

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ──────────────────────────────────────────────
# 🧠 HYPERPARAMETER — BiLSTM & BiLSTM+Attention
# ──────────────────────────────────────────────
VOCAB_SIZE  = 20_000
EMBED_DIM   = 128
HIDDEN_DIM  = 256
NUM_LAYERS  = 2
DROPOUT     = 0.3
MAX_LEN     = 128

LSTM_EPOCHS     = 10
LSTM_BATCH_SIZE = 64
LSTM_LR         = 1e-3
LSTM_PATIENCE   = 3       # early stopping patience

# ──────────────────────────────────────────────
# 🤗 HYPERPARAMETER — DistilBERT
# ──────────────────────────────────────────────
BERT_MODEL      = "distilbert-base-uncased"
BERT_MAX_LEN    = 128
BERT_EPOCHS     = 3
BERT_BATCH_SIZE = 16
BERT_LR         = 2e-5
BERT_PATIENCE   = 2

# ──────────────────────────────────────────────
# 💾 NAMA FILE MODEL
# ──────────────────────────────────────────────
BILSTM_MODEL_PATH     = os.path.join(MODEL_DIR, "bilstm.pt")
BILSTM_ATT_MODEL_PATH = os.path.join(MODEL_DIR, "bilstm_attention.pt")
DISTILBERT_MODEL_DIR  = os.path.join(MODEL_DIR, "distilbert")
VOCAB_PATH            = os.path.join(MODEL_DIR, "vocab.json")
LABEL_ENCODER_PATH    = os.path.join(MODEL_DIR, "label_encoder.json")
