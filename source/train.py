"""
train.py — AutoML Training Pipeline via PyCaret
================================================
Setup PyCaret, compare models, tune, evaluate, dan finalize + export.
"""

import os
import warnings

import pandas as pd

from config import (
    LABEL_COL,
    MODEL_DIR,
    SESSION_ID,
    TRAIN_SIZE,
    N_TOP_MODELS,
)

# Suppress warnings agar output notebook lebih bersih
warnings.filterwarnings("ignore")

import logging
try:
    import lightgbm as lgb
    # Menyembunyikan warning LightGBM dari sisi C++ maupun Python
    logging.getLogger("lightgbm").setLevel(logging.ERROR)
    
    # Patch fungsi inisialisasinya untuk selalu menggunakan verbose=-1
    for lgb_model in [lgb.LGBMClassifier, lgb.LGBMRegressor]:
        original_init = lgb_model.__init__
        def new_init(self, *args, **kwargs):
            kwargs['verbose'] = -1
            original_init(self, *args, **kwargs)
        lgb_model.__init__ = new_init
except ImportError:
    pass


# ══════════════════════════════════════════════
# ⚙️ SETUP PYCARET
# ══════════════════════════════════════════════


def setup_pycaret(df: pd.DataFrame):
    """
    Inisialisasi PyCaret Classification environment.

    PyCaret akan otomatis melakukan TF-IDF pada kolom teks
    ketika kita set `text_features = ['cleaned_text']`.

    Args:
        df: DataFrame yang sudah dibersihkan (harus punya kolom
            'cleaned_text' dan kolom label).

    Returns:
        PyCaret setup object (untuk chaining).
    """
    from pycaret.classification import setup

    print("⚙️  Menginisialisasi PyCaret...")
    print(f"   Kolom teks  : cleaned_text")
    print(f"   Kolom label : {LABEL_COL}")
    print(f"   Train size  : {TRAIN_SIZE}")
    print(f"   Random seed : {SESSION_ID}")

    # Siapkan DataFrame hanya dengan kolom yang diperlukan
    df_model = df[["cleaned_text", LABEL_COL]].copy()

    s = setup(
        data=df_model,
        target=LABEL_COL,
        text_features=["cleaned_text"],
        session_id=SESSION_ID,
        train_size=TRAIN_SIZE,
        verbose=True,
        html=True,
        use_gpu=True,
        fold=5,
    )

    print("✅ PyCaret setup selesai!")
    return s


# ══════════════════════════════════════════════
# 🏟️ MODEL ARENA — COMPARE MODELS
# ══════════════════════════════════════════════


def compare_all_models(sort: str = "F1", n_select: int = None):
    """
    Jalankan arena perbandingan semua model klasifikasi.

    PyCaret akan melatih & mengevaluasi banyak algoritma
    (Logistic Regression, SVM, LightGBM, Random Forest, dll.)
    lalu menampilkan ranking berdasarkan metrik.

    Args:
        sort: Metrik untuk sorting ("Accuracy", "F1", "AUC", dll.)
        n_select: Jumlah model terbaik yang di-return.
                  Default: N_TOP_MODELS dari config.

    Returns:
        Tabel perbandingan (DataFrame) dari compare_models.
    """
    from pycaret.classification import compare_models

    if n_select is None:
        n_select = N_TOP_MODELS

    print(f"🏟️  Memulai Model Arena (sort by {sort})...")
    print(f"   Ini mungkin memakan waktu beberapa menit...\n")

    best = compare_models(
        sort=sort, 
        n_select=n_select,
    )

    print(f"\n✅ Selesai! Top {n_select} model telah dipilih.")
    return best


# ══════════════════════════════════════════════
# 🎯 TUNING
# ══════════════════════════════════════════════


def tune_best(model, optimize: str = "F1"):
    """
    Tuning hyperparameter untuk model terbaik.

    Args:
        model: Model terbaik dari compare_models
        optimize: Metrik yang ingin dioptimalkan

    Returns:
        Model yang telah di-tune
    """
    from pycaret.classification import tune_model

    print(f"🎯 Tuning hyperparameter (optimize: {optimize})...")
    tuned = tune_model(model, optimize=optimize)
    print("✅ Tuning selesai!")
    return tuned


# ══════════════════════════════════════════════
# 📊 EVALUASI
# ══════════════════════════════════════════════


def evaluate(model):
    """
    Tampilkan dashboard evaluasi interaktif PyCaret.

    Menampilkan confusion matrix, classification report,
    dan visualisasi lainnya.

    Args:
        model: Model yang ingin dievaluasi
    """
    from pycaret.classification import evaluate_model

    print("📊 Membuka dashboard evaluasi...")
    evaluate_model(model)


def plot_confusion_matrix(model):
    """Plot confusion matrix untuk model."""
    from pycaret.classification import plot_model

    print("📊 Menampilkan Confusion Matrix...")
    plot_model(model, plot="confusion_matrix")


def plot_feature_importance(model):
    """
    Plot feature importance — kata apa yang paling berpengaruh.

    ⚠️ Hanya bekerja untuk model tree-based
    (Random Forest, LightGBM, XGBoost, dll.)
    """
    from pycaret.classification import plot_model

    print("📊 Menampilkan Feature Importance...")
    try:
        plot_model(model, plot="feature")
    except Exception as e:
        print(f"⚠️  Feature importance tidak tersedia untuk model ini: {e}")
        print("   Coba gunakan model tree-based (RF, LightGBM, XGBoost).")


def plot_class_report(model):
    """Plot classification report visual."""
    from pycaret.classification import plot_model

    print("📊 Menampilkan Classification Report...")
    plot_model(model, plot="class_report")


# ══════════════════════════════════════════════
# 💾 FINALIZE & EXPORT
# ══════════════════════════════════════════════


def finalize_and_save(model, filename: str = "nlp_pipeline_final"):
    """
    Finalize model (retrain di seluruh data) dan simpan sebagai .pkl.

    Args:
        model: Model terbaik yang sudah di-tune
        filename: Nama file (tanpa ekstensi)

    Returns:
        Model yang sudah di-finalize
    """
    from pycaret.classification import finalize_model, save_model

    print("💾 Memfinalisasi model (retrain pada seluruh data)...")
    final = finalize_model(model)

    save_path = os.path.join(MODEL_DIR, filename)
    save_model(final, save_path)
    print(f"✅ Model disimpan: {save_path}.pkl")

    return final


# ──────────────────────────────────────────────
# Jika dijalankan langsung: python train.py
# ──────────────────────────────────────────────
if __name__ == "__main__":
    from preprocess import load_and_clean
    from download_data import download_dataset

    # 1. Download
    download_dataset()

    # 2. Preprocess
    df = load_and_clean()

    # 3. Setup
    setup_pycaret(df)

    # 4. Compare
    best_models = compare_all_models()
    best = best_models[0] if isinstance(best_models, list) else best_models

    # 5. Tune
    tuned = tune_best(best)

    # 6. Finalize
    finalize_and_save(tuned)
