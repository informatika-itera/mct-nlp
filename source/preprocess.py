"""
preprocess.py — Custom Text Cleaning untuk Chat Gamer Indonesia
================================================================
Pipeline: lowercase → hapus URL/mention → normalisasi leetspeak →
ekspansi slang → hapus karakter non-alfabet → strip whitespace.
"""

import re

import pandas as pd

from config import LEETSPEAK_MAP, SLANG_DICT, TEXT_COL, LABEL_COL, RAW_CSV


# ══════════════════════════════════════════════
# 🔧 FUNGSI PEMBANTU
# ══════════════════════════════════════════════


def normalize_leetspeak(text: str) -> str:
    """
    Konversi angka/simbol leetspeak ke huruf biasa.

    Contoh:
        "4nj1n9"  → "anjing"
        "g0bl0k"  → "goblok"
        "t0l0l"   → "tolol"

    Hanya mengganti angka/simbol yang BERDEKATAN dengan huruf,
    agar angka murni (misal tahun, skor) tidak ikut berubah.
    """
    result = []
    for i, char in enumerate(text):
        if char in LEETSPEAK_MAP:
            # Cek apakah ada huruf di sekitar (sebelum atau sesudah)
            prev_is_alpha = (i > 0 and text[i - 1].isalpha())
            next_is_alpha = (i < len(text) - 1 and text[i + 1].isalpha())

            if prev_is_alpha or next_is_alpha:
                result.append(LEETSPEAK_MAP[char])
            else:
                result.append(char)
        else:
            result.append(char)
    return "".join(result)


def expand_slang(text: str) -> str:
    """
    Ekspansi singkatan & slang gamer ke bentuk lengkap.

    Menggunakan word boundary agar tidak mengganti substring
    di dalam kata lain.

    Contoh:
        "gw gblk bgt" → "gue goblok banget"
    """
    words = text.split()
    expanded = [SLANG_DICT.get(w, w) for w in words]
    return " ".join(expanded)


def clean_text(text: str) -> str:
    """
    Pipeline pembersihan teks lengkap.

    Urutan:
    1. Lowercase
    2. Hapus URL (http/https/www)
    3. Hapus mention (@user)
    4. Normalisasi leetspeak (angka → huruf)
    5. Ekspansi slang gamer
    6. Hapus karakter non-alfabet (kecuali spasi)
    7. Hapus whitespace berlebih

    Args:
        text: Teks mentah dari chat

    Returns:
        Teks yang sudah dinormalisasi
    """
    if not isinstance(text, str):
        return ""

    # 1. Lowercase
    text = text.lower()

    # 2. Hapus URL
    text = re.sub(r"https?://\S+|www\.\S+", "", text)

    # 3. Hapus mention
    text = re.sub(r"@\w+", "", text)

    # 4. Normalisasi leetspeak
    text = normalize_leetspeak(text)

    # 5. Ekspansi slang
    text = expand_slang(text)

    # 6. Hapus karakter non-alfabet (kecuali spasi)
    text = re.sub(r"[^a-z\s]", "", text)

    # 7. Hapus whitespace berlebih
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
