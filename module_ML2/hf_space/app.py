"""
app.py — Gradio App untuk Analisis Sentimen Review Film IMDB
=============================================================
Deploy di Hugging Face Spaces.
Model: PyCaret NLP Classification Pipeline (.pkl)
"""

print("Starting app initialization...")
import re
print("Importing Gradio...")
import gradio as gr
print("Importing Pandas...")
import pandas as pd
print("Importing PyCaret...")
from pycaret.classification import load_model, predict_model

# ══════════════════════════════════════════════
# 📦 LOAD MODEL
# ══════════════════════════════════════════════
print("Loading Model...")
model = load_model("imdb_sentiment_pipeline_final")
print("Model Successfully Loaded!")

# ══════════════════════════════════════════════
# 🔤 PREPROCESSING (sama persis dengan training)
# ══════════════════════════════════════════════

CONTRACTION_MAP = {
    "won't": "will not", "can't": "cannot", "i'm": "i am",
    "ain't": "is not", "let's": "let us", "isn't": "is not",
    "it's": "it is", "i've": "i have", "doesn't": "does not",
    "didn't": "did not", "don't": "do not", "wasn't": "was not",
    "weren't": "were not", "couldn't": "could not",
    "shouldn't": "should not", "wouldn't": "would not",
    "haven't": "have not", "hasn't": "has not", "hadn't": "had not",
    "they're": "they are", "we're": "we are", "you're": "you are",
    "he's": "he is", "she's": "she is", "that's": "that is",
    "what's": "what is", "who's": "who is", "there's": "there is",
    "here's": "here is", "where's": "where is",
    "i'll": "i will", "you'll": "you will", "he'll": "he will",
    "she'll": "she will", "we'll": "we will", "they'll": "they will",
    "i'd": "i would", "you'd": "you would", "he'd": "he would",
    "she'd": "she would", "we'd": "we would", "they'd": "they would",
    "you've": "you have", "we've": "we have", "they've": "they have",
}


def remove_html_tags(text: str) -> str:
    """Hapus tag HTML dari teks review."""
    return re.sub(r"<[^>]+>", " ", text)


def expand_contractions(text: str) -> str:
    """Ekspansi kontraksi bahasa Inggris ke bentuk lengkap."""
    words = text.split()
    expanded = [CONTRACTION_MAP.get(w, w) for w in words]
    return " ".join(expanded)


def clean_text(text: str) -> str:
    """Pipeline pembersihan teks (sama persis dengan training)."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = remove_html_tags(text)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = expand_contractions(text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ══════════════════════════════════════════════
# 🎯 FUNGSI PREDIKSI
# ══════════════════════════════════════════════


def predict_sentiment(text: str) -> dict:
    """
    Prediksi sentimen dari teks review film.

    Returns:
        Dictionary {label: confidence} untuk Gradio Label component.
    """
    if not text or not text.strip():
        return {"Error": 1.0}

    # 1. Bersihkan teks
    cleaned = clean_text(text)

    if not cleaned:
        return {"Teks kosong setelah dibersihkan": 1.0}

    # 2. Buat DataFrame (PyCaret butuh DataFrame sebagai input)
    df_input = pd.DataFrame({"cleaned_text": [cleaned]})

    # 3. Prediksi menggunakan PyCaret
    result = predict_model(model, data=df_input)

    # 4. Ambil hasil
    if "prediction_label" in result.columns:
        label = result["prediction_label"].iloc[0]
    elif "Label" in result.columns:
        label = result["Label"].iloc[0]
    else:
        label = str(result.iloc[0, -1])

    if "prediction_score" in result.columns:
        score = result["prediction_score"].iloc[0]
    elif "Score" in result.columns:
        score = result["Score"].iloc[0]
    else:
        score = 1.0

    return {label: float(score)}


# ══════════════════════════════════════════════
# 🎨 GRADIO INTERFACE
# ══════════════════════════════════════════════

EXAMPLES = [
    ["This movie was absolutely fantastic! The acting was superb and the plot kept me engaged throughout."],
    ["Terrible film. I couldn't even finish watching it. Complete waste of time."],
    ["A decent movie with some good moments, but the ending was disappointing."],
    ["One of the best movies I've ever seen! A true masterpiece of cinema."],
    ["The worst movie of the year. Bad acting, bad script, bad everything."],
    ["An okay film. Nothing special but entertaining enough for a lazy Sunday."],
]

demo = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(
        label="🎬 Masukkan Review Film",
        placeholder="Ketik review film di sini... (contoh: 'This movie was great!')",
        lines=3,
    ),
    outputs=gr.Label(
        label="🎯 Hasil Prediksi Sentimen",
        num_top_classes=2,
    ),
    title="🎬 Analisis Sentimen Review Film IMDB",
    description=(
        "Model NLP untuk mendeteksi apakah sebuah review film bersifat "
        "**positive** atau **negative**.\n\n"
        "Model ini dilatih menggunakan PyCaret dengan TF-IDF + "
        "custom preprocessing pada dataset IMDB 50K Movie Reviews."
    ),
    examples=EXAMPLES,
    cache_examples=False,
    theme=gr.themes.Soft(),
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
