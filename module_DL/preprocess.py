"""
preprocess.py — Pipeline Pembersihan Teks
==========================================
Membersihkan teks statement mental health:
lowercase → hapus URL → hapus HTML → hapus non-alpha → strip.
"""

import re
import json

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from config import TEXT_COL, LABEL_COL, LABEL_LIST, RANDOM_SEED, LABEL_ENCODER_PATH


# ──────────────────────────────────────────────
# 🧹 FUNGSI PEMBERSIHAN
# ──────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Membersihkan satu string teks.

    Langkah:
    1. Lowercase
    2. Hapus URL / link
    3. Hapus tag HTML
    4. Hapus karakter non-alfanumerik (kecuali spasi)
    5. Hapus spasi berlebih

    Args:
        text: Teks mentah.

    Returns:
        Teks yang sudah bersih.
    """
    if not isinstance(text, str):
        return ""

    # 1) Lowercase
    text = text.lower()

    # 2) Hapus URL
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # 3) Hapus tag HTML
    text = re.sub(r"<[^>]+>", " ", text)

    # 4) Hapus karakter selain huruf dan spasi
    text = re.sub(r"[^a-z\s]", " ", text)

    # 5) Hapus spasi berlebih
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ──────────────────────────────────────────────
# 📥 LOAD & CLEAN
# ──────────────────────────────────────────────

def load_and_clean(
    csv_path: str,
    sample_size: int | None = None,
) -> pd.DataFrame:
    """
    Membaca CSV, membersihkan teks, dan encode label.

    Args:
        csv_path:    Path ke file CSV dataset.
        sample_size: Jika tidak None, ambil sampel stratified
                     sebanyak sample_size baris.

    Returns:
        DataFrame dengan kolom:
        - 'text'          : teks asli
        - 'cleaned_text'  : teks yang sudah bersih
        - 'label'         : label string asli
        - 'label_encoded' : label numerik (0–6)
    """
    print(f"📂 Membaca dataset: {csv_path}")
    df = pd.read_csv(csv_path)

    # Normalisasi nama kolom
    df.columns = df.columns.str.strip()

    # Ambil hanya kolom yang diperlukan dan hapus baris kosong
    df = df[[TEXT_COL, LABEL_COL]].copy()
    df.rename(columns={TEXT_COL: "text", LABEL_COL: "label"}, inplace=True)
    df.dropna(subset=["text", "label"], inplace=True)
    df["text"]  = df["text"].astype(str)
    df["label"] = df["label"].astype(str).str.strip()

    # Normalisasi nilai label agar cocok dengan LABEL_LIST
    label_map = {l.lower(): l for l in LABEL_LIST}
    df["label"] = df["label"].apply(
        lambda x: label_map.get(x.lower(), x)
    )

    # Buang baris label yang tidak dikenal
    df = df[df["label"].isin(LABEL_LIST)].reset_index(drop=True)

    print(f"📊 Total data setelah filter: {len(df):,} baris")
    print(f"   Distribusi kelas:\n{df['label'].value_counts().to_string()}\n")

    # ── Sampling stratified ──
    if sample_size and sample_size < len(df):
        df, _ = train_test_split(
            df,
            train_size=sample_size,
            stratify=df["label"],
            random_state=RANDOM_SEED,
        )
        df = df.reset_index(drop=True)
        print(f"🔀 Setelah sampling: {len(df):,} baris")

    # ── Pembersihan teks ──
    print("🧹 Membersihkan teks...")
    df["cleaned_text"] = df["text"].apply(clean_text)

    # Hapus baris dengan teks kosong setelah dibersihkan
    df = df[df["cleaned_text"].str.len() > 0].reset_index(drop=True)

    # ── Encode label ──
    le = LabelEncoder()
    le.fit(LABEL_LIST)
    df["label_encoded"] = le.transform(df["label"])

    # Simpan mapping label encoder
    label_mapping = {label: int(idx) for idx, label in enumerate(le.classes_)}
    with open(LABEL_ENCODER_PATH, "w") as f:
        json.dump(label_mapping, f, indent=2)

    print(f"✅ Selesai. {len(df):,} baris siap digunakan.")
    return df


# ──────────────────────────────────────────────
# 🔍 TAMPILKAN CONTOH PEMBERSIHAN
# ──────────────────────────────────────────────

def show_cleaning_examples(df: pd.DataFrame, n: int = 5) -> None:
    """Menampilkan contoh teks sebelum dan sesudah dibersihkan."""
    print(f"{'='*70}")
    print("CONTOH PEMBERSIHAN TEKS")
    print(f"{'='*70}")
    for i, row in df.sample(n=n, random_state=RANDOM_SEED).iterrows():
        print(f"\n[{row['label']}]")
        print(f"  Asli    : {row['text'][:120]}")
        print(f"  Bersih  : {row['cleaned_text'][:120]}")
    print(f"\n{'='*70}")
