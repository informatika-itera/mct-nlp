"""
download_data.py — Auto-Download Dataset dari Kaggle
=====================================================
Mendownload dataset "Indonesian Chat Dataset" secara otomatis
menggunakan kagglehub (tidak perlu API key untuk public dataset).
"""

import os
import shutil
import glob

from config import DATA_DIR, RAW_CSV


# ──────────────────────────────────────────────
# Konstanta Kaggle
# ──────────────────────────────────────────────
KAGGLE_DATASET = "jprestiliano/indonesian-chat-dataset"
EXPECTED_FILENAME = "indonesian_chat.csv"


def download_dataset() -> str:
    """
    Download dataset dari Kaggle ke folder data/.

    Alur:
    1. Cek apakah file CSV sudah ada → skip jika sudah.
    2. Download via kagglehub (tanpa API key untuk public dataset).
    3. Copy file CSV ke data/ lokal.

    Returns:
        str: Path absolut ke file CSV yang siap dipakai.
    """

    # ── Sudah ada? Skip ──
    if os.path.exists(RAW_CSV):
        print(f"✅ Dataset sudah ada: {RAW_CSV}")
        return RAW_CSV

    # ── Download via kagglehub ──
    print("📥 Mendownload dataset dari Kaggle...")
    try:
        import kagglehub

        download_path = kagglehub.dataset_download(KAGGLE_DATASET)
        print(f"📦 Dataset didownload ke: {download_path}")

        # Cari file CSV di folder download
        csv_files = glob.glob(
            os.path.join(download_path, "**", EXPECTED_FILENAME),
            recursive=True,
        )

        if not csv_files:
            # Cari CSV manapun yang ada
            csv_files = glob.glob(
                os.path.join(download_path, "**", "*.csv"),
                recursive=True,
            )

        if csv_files:
            src = csv_files[0]
            shutil.copy2(src, RAW_CSV)
            print(f"✅ Dataset disalin ke: {RAW_CSV}")
            return RAW_CSV
        else:
            raise FileNotFoundError(
                f"Tidak ditemukan file CSV di {download_path}"
            )

    except Exception as e:
        print(f"⚠️  kagglehub gagal: {e}")
        print("🔄 Mencoba download via opendatasets...")
        return _fallback_opendatasets()


def _fallback_opendatasets() -> str:
    """Fallback: download via opendatasets jika kagglehub gagal."""
    try:
        import opendatasets as od

        kaggle_url = (
            "https://www.kaggle.com/datasets/"
            "jprestiliano/indonesian-chat-dataset"
        )
        od.download(kaggle_url, data_dir=DATA_DIR)

        # opendatasets membuat subfolder
        downloaded = glob.glob(
            os.path.join(DATA_DIR, "**", "*.csv"), recursive=True
        )

        if downloaded:
            src = downloaded[0]
            if src != RAW_CSV:
                shutil.copy2(src, RAW_CSV)
            print(f"✅ Dataset tersedia di: {RAW_CSV}")
            return RAW_CSV
        else:
            raise FileNotFoundError("File CSV tidak ditemukan setelah download.")

    except Exception as e2:
        raise RuntimeError(
            f"❌ Gagal mendownload dataset.\n"
            f"   Error: {e2}\n"
            f"   Silakan download manual dari:\n"
            f"   https://www.kaggle.com/datasets/jprestiliano/"
            f"indonesian-chat-dataset\n"
            f"   Lalu simpan sebagai: {RAW_CSV}"
        ) from e2


# ──────────────────────────────────────────────
# Jika dijalankan langsung: python download_data.py
# ──────────────────────────────────────────────
if __name__ == "__main__":
    path = download_dataset()
    print(f"\n🎉 Dataset siap digunakan: {path}")
