import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from config import RAW_CSV, TEXT_COL, LABEL_COL
from download_data import download_dataset
from preprocess import load_and_clean, show_cleaning_examples, clean_text
from train import (
    setup_pycaret,
    compare_all_models,
    tune_best,
    finalize_and_save,
)
from pycaret.classification import plot_model

def main():
    print("🚀 Memulai Pipeline Training dari Terminal...")

    # Langkah 2: Download Dataset
    csv_path = download_dataset()
    print(f"\n📁 File tersimpan di: {csv_path}")

    # Langkah 3: Eksplorasi Data Awal
    df_raw = pd.read_csv(csv_path)
    print(f"📏 Ukuran dataset: {df_raw.shape[0]:,} baris × {df_raw.shape[1]} kolom")
    print(f"📋 Kolom: {list(df_raw.columns)}\n")

    print("📊 Distribusi Label:")
    print("=" * 30)
    label_counts = df_raw[LABEL_COL].value_counts()
    for label, count in label_counts.items():
        pct = count / len(df_raw) * 100
        print(f"   {label:12s}: {count:5,} ({pct:.1f}%)")
    print(f"   {'TOTAL':12s}: {len(df_raw):5,}\n")

    # Visualisasi distribusi label disimpan sebagai file
    plt.figure(figsize=(8, 5))
    sns.barplot(x=label_counts.index, y=label_counts.values, palette='viridis')
    plt.title('Distribusi Label', fontsize=14)
    plt.xlabel('Label', fontsize=12)
    plt.ylabel('Jumlah', fontsize=12)
    plt.xticks(rotation=45)
    
    plots_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    dist_plot_path = os.path.join(plots_dir, 'distribusi_label.png')
    plt.tight_layout()
    plt.savefig(dist_plot_path)
    print(f"✅ Plot distribusi label disimpan sebagai '{dist_plot_path}'\n")

    # Langkah 4: Text Preprocessing
    df = load_and_clean() # load_and_clean without args pulls from config inside

    print("🔄 Contoh Hasil Pembersihan (Before → After):")
    print("=" * 70)
    print(show_cleaning_examples(df, n=10).to_string(index=False))

    # Langkah 5: Setup PyCaret
    setup_pycaret(df)

    # Langkah 6: Model Arena
    best_models = compare_all_models(sort="F1")
    
    if isinstance(best_models, list):
        best = best_models[0]
        print(f"🏆 Model Terbaik: {best}")
    else:
        best = best_models
        print(f"🏆 Model Terbaik: {best}")

    # Langkah 7: Tuning Hyperparameter
    tuned_model = tune_best(best, optimize="F1")

    # Langkah 8: Evaluasi Model (menyimpan plot)
    print("\n📊 Menyimpan Plot Evaluasi Model...")
    
    # Pindah direktori kerja ke plots_dir agar file plotters tersimpan di sana
    original_cwd = os.getcwd()
    os.chdir(plots_dir)
    
    try:
        plot_model(tuned_model, plot='confusion_matrix', save=True)
        print("✅ Confusion Matrix disimpan.")
    except Exception as e:
        print(f"⚠️ Gagal menyimpan Confusion Matrix: {e}")

    try:
        plot_model(tuned_model, plot='feature', save=True)
        print("✅ Feature Importance disimpan.")
    except Exception as e:
        print(f"⚠️ Gagal menyimpan Feature Importance: {e}")

    try:
        plot_model(tuned_model, plot='class_report', save=True)
        print("✅ Classification Report disimpan.")
    except Exception as e:
        print(f"⚠️ Gagal menyimpan Classification Report: {e}")

    # Kembalikan ke direktori semula
    os.chdir(original_cwd)

    # Langkah 9: Finalize & Export Model
    print("\n💾 Finalisasi dan Menyimpan Model...")
    final_model = finalize_and_save(tuned_model)
    print("\n🎉 Pipeline Training Selesai!")

if __name__ == "__main__":
    main()
