"""
train_run.py — NGBoost IMDB Sentiment Classifier
Jalankan: python train_run.py
"""

from download_data import download_dataset
from preprocess import load_and_clean, show_cleaning_examples
from train import (
    build_pipeline, train_ngboost, evaluate,
    plot_label_distribution, plot_confusion_matrix,
    plot_training_loss, plot_metrics_bar, save_model,
)


def main():
    # 1. Download / locate dataset
    print("\n" + "=" * 55)
    print("  STEP 1: Download Dataset")
    print("=" * 55)
    csv_path = download_dataset()

    # 2. Load & clean
    print("\n" + "=" * 55)
    print("  STEP 2: Load & Preprocess")
    print("=" * 55)
    df = load_and_clean(csv_path)
    show_cleaning_examples(df, n=5)

    # 3. Visualisasi distribusi label
    print("\n" + "=" * 55)
    print("  STEP 3: Distribusi Label")
    print("=" * 55)
    print(df["sentiment"].value_counts())
    plot_label_distribution(df)

    # 4. Build pipeline (TF-IDF + split)
    print("\n" + "=" * 55)
    print("  STEP 4: Build TF-IDF Pipeline")
    print("=" * 55)
    X_train, X_test, y_train, y_test, vectorizer, le = build_pipeline(df)

    # 5. Train NGBoost
    print("\n" + "=" * 55)
    print("  STEP 5: Train NGBoost Classifier")
    print("=" * 55)
    model = train_ngboost(X_train, y_train)

    # 6. Evaluate
    print("\n" + "=" * 55)
    print("  STEP 6: Evaluasi Model")
    print("=" * 55)
    y_pred, metrics = evaluate(model, X_test, y_test, le)

    # 7. Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred, le)

    # 8. Plot training loss
    plot_training_loss(model)

    # 9. Plot metrics bar chart
    plot_metrics_bar(metrics)

    # 10. Save model
    print("\n" + "=" * 55)
    print("  STEP 7: Save Model")
    print("=" * 55)
    save_model(model, vectorizer, le)

    print("\n✅ Pipeline selesai!\n")


if __name__ == "__main__":
    main()
