"""Text preprocessing pipeline untuk IMDB reviews."""

import re
import pandas as pd
from config import TEXT_COL, LABEL_COL, CONTRACTION_MAP


# ── helper functions ────────────────────────────────────────────────────

def lowercase(text: str) -> str:
    return text.lower()


def remove_html_tags(text: str) -> str:
    return re.sub(r"<[^>]+>", " ", text)


def remove_urls(text: str) -> str:
    return re.sub(r"https?://\S+|www\.\S+", " ", text)


def expand_contractions(text: str) -> str:
    for contraction, expanded in CONTRACTION_MAP.items():
        text = text.replace(contraction, expanded)
    return text


def remove_non_alpha(text: str) -> str:
    return re.sub(r"[^a-z\s]", " ", text)


def strip_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def clean_text(text: str) -> str:
    """Full cleaning pipeline."""
    text = lowercase(text)
    text = remove_html_tags(text)
    text = remove_urls(text)
    text = expand_contractions(text)
    text = remove_non_alpha(text)
    text = strip_whitespace(text)
    return text


# ── main API ────────────────────────────────────────────────────────────

def load_and_clean(csv_path: str) -> pd.DataFrame:
    """Load CSV, bersihkan teks, dan kembalikan DataFrame."""
    df = pd.read_csv(csv_path)
    print(f"[INFO] Loaded {len(df)} rows dari {csv_path}")
    df["cleaned_text"] = df[TEXT_COL].astype(str).apply(clean_text)
    print("[INFO] Text cleaning selesai.")
    return df


def show_cleaning_examples(df: pd.DataFrame, n: int = 5) -> None:
    """Tampilkan contoh sebelum & sesudah cleaning."""
    sample = df.head(n)
    print("\n── Contoh Cleaning ──")
    for i, row in sample.iterrows():
        print(f"\n[{i}] BEFORE: {row[TEXT_COL][:120]}...")
        print(f"    AFTER : {row['cleaned_text'][:120]}...")
    print()


if __name__ == "__main__":
    from download_data import download_dataset
    path = download_dataset()
    df = load_and_clean(path)
    show_cleaning_examples(df)
