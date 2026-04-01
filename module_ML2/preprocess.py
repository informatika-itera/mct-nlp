"""
preprocess.py — Custom Text Cleaning untuk Review Film IMDB
============================================================
Pipeline: lowercase → hapus tag HTML → ekspansi kontraksi →
hapus URL → hapus karakter non-alfabet → strip whitespace.
"""

import re

import pandas as pd

from config import CONTRACTION_MAP, TEXT_COL, LABEL_COL, RAW_CSV


# ══════════════════════════════════════════════
# 🔧 FUNGSI PEMBANTU
# ══════════════════════════════════════════════


def remove_html_tags(text: str) -> str:
    """
    Hapus tag HTML dari teks review.

    Dataset IMDB mengandung tag HTML seperti <br />, <p>, dll.
    karena review diambil langsung dari halaman web.

    Contoh:
        "Great movie!<br /><br />Loved it" → "Great movie! Loved it"
    """
    return re.sub(r"<[^>]+>", " ", text)


def expand_contractions(text: str) -> str:
    """
    Ekspansi kontraksi bahasa Inggris ke bentuk lengkap.

    Review film sering menggunakan bahasa informal
    dengan banyak kontraksi.

    Contoh:
        "I can't believe it's so good" → "I cannot believe it is so good"
        "They won't disappoint you"   → "They will not disappoint you"
    """
    words = text.split()
    expanded = [CONTRACTION_MAP.get(w, w) for w in words]
    return " ".join(expanded)


def clean_text(text: str) -> str:
    """
    Pipeline pembersihan teks lengkap untuk review film IMDB.

    Urutan:
    1. Lowercase
    2. Hapus tag HTML (<br />, <p>, dll.)
    3. Hapus URL (http/https/www)
    4. Ekspansi kontraksi (can't → cannot)
    5. Hapus karakter non-alfabet (kecuali spasi)
    6. Hapus whitespace berlebih

    Args:
        text: Teks mentah review film

    Returns:
        Teks yang sudah dinormalisasi
    """
    if not isinstance(text, str):
        return ""

    # 1. Lowercase
    text = text.lower()

    # 2. Hapus tag HTML
    text = remove_html_tags(text)

    # 3. Hapus URL
    text = re.sub(r"https?://\S+|www\.\S+", "", text)

    # 4. Ekspansi kontraksi
    text = expand_contractions(text)

    # 5. Hapus karakter non-alfabet (kecuali spasi)
    text = re.sub(r"[^a-z\s]", "", text)

    # 6. Hapus whitespace berlebih
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ══════════════════════════════════════════════
# 📂 FUNGSI UTAMA
# ══════════════════════════════════════════════


def load_and_clean(csv_path: str = None) -> pd.DataFrame:
    """
    Baca CSV dataset dan jalankan pipeline pembersihan teks.

    Args:
        csv_path: Path ke file CSV. Default: RAW_CSV dari config.

    Returns:
        DataFrame dengan kolom teks yang sudah dibersihkan.
        Kolom baru 'cleaned_text' ditambahkan, kolom asli tetap ada.
    """
    if csv_path is None:
        csv_path = RAW_CSV

    print(f"📂 Membaca dataset: {csv_path}")
    df = pd.read_csv(csv_path)

    print(f"   Jumlah baris: {len(df):,}")
    print(f"   Kolom: {list(df.columns)}")

    # Hapus baris kosong
    df = df.dropna(subset=[TEXT_COL, LABEL_COL]).reset_index(drop=True)

    # Bersihkan teks
    print("🧹 Membersihkan teks...")
    df["cleaned_text"] = df[TEXT_COL].apply(clean_text)

    # Hapus baris yang setelah dibersihkan jadi kosong
    df = df[df["cleaned_text"].str.len() > 0].reset_index(drop=True)

    print(f"✅ Selesai! Jumlah baris bersih: {len(df):,}")
    return df


def show_cleaning_examples(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """
    Tampilkan contoh before vs after pembersihan teks.
    Berguna untuk presentasi di notebook.

    Args:
        df: DataFrame yang sudah melalui load_and_clean
        n: Jumlah contoh

    Returns:
        DataFrame kecil dengan kolom 'original' dan 'cleaned'
    """
    sample = df.sample(n=min(n, len(df)), random_state=42)
    return sample[[TEXT_COL, "cleaned_text", LABEL_COL]].rename(
        columns={TEXT_COL: "original", "cleaned_text": "cleaned", LABEL_COL: "label"}
    )


# ──────────────────────────────────────────────
# Jika dijalankan langsung: python preprocess.py
# ──────────────────────────────────────────────
if __name__ == "__main__":
    df = load_and_clean()
    print("\n📋 Contoh hasil pembersihan:")
    print(show_cleaning_examples(df).to_string(index=False))
