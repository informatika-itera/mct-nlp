"""
config.py — Konfigurasi & Konstanta untuk Workshop NLP Sesi 1 (IMDB)
=====================================================================
Berisi path, kamus kontraksi bahasa Inggris, dan daftar stopwords.
Dataset: IMDB 50K Movie Reviews — Binary Sentiment Classification.
"""

import os

# ──────────────────────────────────────────────
# 📁 PATH
# ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

RAW_CSV = os.path.join(DATA_DIR, "IMDB Dataset.csv")

# Buat folder kalau belum ada
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ──────────────────────────────────────────────
# 🔤 CONTRACTION MAPPING (Bahasa Inggris)
# ──────────────────────────────────────────────
# Review film sering menggunakan kontraksi informal.
# Kita ekspansi supaya TF-IDF lebih konsisten.
CONTRACTION_MAP = {
    "won't": "will not",
    "can't": "cannot",
    "i'm": "i am",
    "ain't": "is not",
    "let's": "let us",
    "isn't": "is not",
    "it's": "it is",
    "i've": "i have",
    "doesn't": "does not",
    "didn't": "did not",
    "don't": "do not",
    "wasn't": "was not",
    "weren't": "were not",
    "couldn't": "could not",
    "shouldn't": "should not",
    "wouldn't": "would not",
    "haven't": "have not",
    "hasn't": "has not",
    "hadn't": "had not",
    "they're": "they are",
    "we're": "we are",
    "you're": "you are",
    "he's": "he is",
    "she's": "she is",
    "that's": "that is",
    "what's": "what is",
    "who's": "who is",
    "there's": "there is",
    "here's": "here is",
    "where's": "where is",
    "i'll": "i will",
    "you'll": "you will",
    "he'll": "he will",
    "she'll": "she will",
    "we'll": "we will",
    "they'll": "they will",
    "i'd": "i would",
    "you'd": "you would",
    "he'd": "he would",
    "she'd": "she would",
    "we'd": "we would",
    "they'd": "they would",
    "you've": "you have",
    "we've": "we have",
    "they've": "they have",
}

# ──────────────────────────────────────────────
# 🛑 KOLOM DATASET
# ──────────────────────────────────────────────
# Nama kolom teks & label di CSV IMDB.
TEXT_COL = "review"
LABEL_COL = "sentiment"

# ──────────────────────────────────────────────
# 🎯 PYCARET SETTINGS
# ──────────────────────────────────────────────
SESSION_ID = 42        # Random seed untuk reprodusibilitas
TRAIN_SIZE = 0.8       # 80% train, 20% test
N_TOP_MODELS = 5       # Jumlah top model dari compare_models
