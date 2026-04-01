"""Download IMDB 50K movie-review dataset dari Kaggle."""

import os
import shutil
from config import DATA_DIR, KAGGLE_DATASET, CSV_FILENAME


def download_dataset() -> str:
    """Download dataset dan kembalikan path ke CSV."""
    csv_path = os.path.join(DATA_DIR, CSV_FILENAME)
    if os.path.isfile(csv_path):
        print(f"[INFO] Dataset sudah ada: {csv_path}")
        return csv_path

    # ── primary: kagglehub ──────────────────────────────────────────
    try:
        import kagglehub
        downloaded = kagglehub.dataset_download(KAGGLE_DATASET)
        print(f"[INFO] Dataset downloaded via kagglehub: {downloaded}")

        # cari file CSV di folder hasil download
        for root, _, files in os.walk(downloaded):
            for f in files:
                if f == CSV_FILENAME:
                    src = os.path.join(root, f)
                    shutil.copy2(src, csv_path)
                    print(f"[INFO] Copied to {csv_path}")
                    return csv_path
    except Exception as e:
        print(f"[WARN] kagglehub gagal: {e}")

    # ── fallback: opendatasets ──────────────────────────────────────
    try:
        import opendatasets as od
        url = f"https://www.kaggle.com/datasets/{KAGGLE_DATASET}"
        od.download(url, data_dir=DATA_DIR)
        for root, _, files in os.walk(DATA_DIR):
            for f in files:
                if f == CSV_FILENAME:
                    src = os.path.join(root, f)
                    if src != csv_path:
                        shutil.copy2(src, csv_path)
                    return csv_path
    except Exception as e:
        print(f"[WARN] opendatasets gagal: {e}")

    raise FileNotFoundError(
        f"Gagal download dataset. Letakkan '{CSV_FILENAME}' secara manual di {DATA_DIR}"
    )


if __name__ == "__main__":
    path = download_dataset()
    print(f"Dataset path: {path}")
