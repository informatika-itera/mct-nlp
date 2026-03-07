"""
config.py — Konfigurasi & Konstanta untuk Workshop NLP Sesi 1
=============================================================
Berisi path, mapping leetspeak, kamus slang gamer Indonesia,
dan daftar stopwords dasar.
"""

import os

# ──────────────────────────────────────────────
# 📁 PATH
# ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

RAW_CSV = os.path.join(DATA_DIR, "indonesian_chat.csv")

# Buat folder kalau belum ada
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ──────────────────────────────────────────────
# 🔤 LEETSPEAK MAPPING
# ──────────────────────────────────────────────
# Gamer Indonesia sering mengganti huruf dengan angka
# untuk menghindari filter kata kasar.
LEETSPEAK_MAP = {
    "0": "o",
    "1": "i",
    "2": "z",
    "3": "e",
    "4": "a",
    "5": "s",
    "6": "g",
    "7": "t",
    "8": "b",
    "9": "g",
    "@": "a",
}

# ──────────────────────────────────────────────
# 💬 KAMUS SLANG GAMER INDONESIA
# ──────────────────────────────────────────────
# Singkatan & slang yang umum di chat game Indonesia.
# Digunakan untuk normalisasi teks sebelum TF-IDF.
SLANG_DICT = {
    # --- Kata kasar / toxic ---
    "anj": "anjing",
    "anjg": "anjing",
    "anjr": "anjing",
    "anjir": "anjing",
    "anjer": "anjing",
    "ajg": "anjing",
    "gblk": "goblok",
    "gblg": "goblok",
    "goblog": "goblok",
    "bgo": "bego",
    "bngst": "bangsat",
    "bgst": "bangsat",
    "kntl": "kontol",
    "mmk": "memek",
    "jnck": "jancok",
    "jncok": "jancok",
    "jncuk": "jancok",
    "tll": "tolol",
    "tlol": "tolol",
    "bdsm": "bodoh",
    "bdh": "bodoh",

    # --- Slang umum ---
    "gw": "gue",
    "gua": "gue",
    "lu": "lo",
    "elu": "lo",
    "lo": "lo",
    "loe": "lo",
    "ga": "tidak",
    "gak": "tidak",
    "nggak": "tidak",
    "ngga": "tidak",
    "g": "tidak",
    "tdk": "tidak",
    "gk": "tidak",
    "kyk": "kayak",
    "kek": "kayak",
    "emg": "emang",
    "emng": "emang",
    "bgt": "banget",
    "bngt": "banget",
    "bgtt": "banget",
    "udh": "sudah",
    "udah": "sudah",
    "sdh": "sudah",
    "dah": "sudah",
    "blm": "belum",
    "blom": "belum",
    "yg": "yang",
    "dgn": "dengan",
    "dg": "dengan",
    "sm": "sama",
    "sma": "sama",
    "tp": "tapi",
    "tpi": "tapi",
    "org": "orang",
    "ornag": "orang",
    "krn": "karena",
    "krna": "karena",
    "jgn": "jangan",
    "jng": "jangan",
    "bkn": "bukan",
    "gpp": "tidak apa-apa",
    "otw": "on the way",
    "btw": "by the way",
    "cmn": "cuman",
    "lg": "lagi",
    "lgi": "lagi",
    "aja": "saja",
    "aj": "saja",
    "bs": "bisa",
    "bsa": "bisa",
    "dr": "dari",
    "dri": "dari",
    "utk": "untuk",
    "trs": "terus",
    "trus": "terus",
    "trus": "terus",
    "msh": "masih",
    "masi": "masih",
    "jd": "jadi",
    "jdi": "jadi",
    "skrg": "sekarang",
    "skrng": "sekarang",

    # --- Gaming terms ---
    "noob": "pemula",
    "newbie": "pemula",
    "pro": "profesional",
    "gg": "good game",
    "wp": "well played",
    "afk": "away from keyboard",
    "ez": "easy",
    "lag": "lag",
    "dc": "disconnect",
    "bcs": "karena",
}

# ──────────────────────────────────────────────
# 🛑 KOLOM DATASET
# ──────────────────────────────────────────────
# Nama kolom teks & label di CSV.
# Sesuaikan jika nama kolom berbeda.
TEXT_COL = "chat"
LABEL_COL = "label"

# ──────────────────────────────────────────────
# 🎯 PYCARET SETTINGS
# ──────────────────────────────────────────────
SESSION_ID = 42        # Random seed untuk reprodusibilitas
TRAIN_SIZE = 0.8       # 80% train, 20% test
N_TOP_MODELS = 5       # Jumlah top model dari compare_models
