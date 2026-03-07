"""
app.py — Gradio App untuk Deteksi Toksisitas Chat Gamer Indonesia
=================================================================
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
model = load_model("nlp_pipeline_final")
print("Model Successfully Loaded!")

# ══════════════════════════════════════════════
# 🔤 PREPROCESSING (sama persis dengan training)
# ══════════════════════════════════════════════

LEETSPEAK_MAP = {
    "0": "o", "1": "i", "2": "z", "3": "e", "4": "a",
    "5": "s", "6": "g", "7": "t", "8": "b", "9": "g", "@": "a",
}

SLANG_DICT = {
    # Kata kasar / toxic
    "anj": "anjing", "anjg": "anjing", "anjr": "anjing", "anjir": "anjing",
    "anjer": "anjing", "ajg": "anjing", "gblk": "goblok", "gblg": "goblok",
    "goblog": "goblok", "bgo": "bego", "bngst": "bangsat", "bgst": "bangsat",
    "kntl": "kontol", "mmk": "memek", "jnck": "jancok", "jncok": "jancok",
    "jncuk": "jancok", "tll": "tolol", "tlol": "tolol", "bdsm": "bodoh",
    "bdh": "bodoh",
    # Slang umum
    "gw": "gue", "gua": "gue", "lu": "lo", "elu": "lo", "lo": "lo",
    "loe": "lo", "ga": "tidak", "gak": "tidak", "nggak": "tidak",
    "ngga": "tidak", "g": "tidak", "tdk": "tidak", "gk": "tidak",
    "kyk": "kayak", "kek": "kayak", "emg": "emang", "emng": "emang",
    "bgt": "banget", "bngt": "banget", "bgtt": "banget", "udh": "sudah",
    "udah": "sudah", "sdh": "sudah", "dah": "sudah", "blm": "belum",
    "blom": "belum", "yg": "yang", "dgn": "dengan", "dg": "dengan",
    "sm": "sama", "sma": "sama", "tp": "tapi", "tpi": "tapi",
    "org": "orang", "ornag": "orang", "krn": "karena", "krna": "karena",
    "jgn": "jangan", "jng": "jangan", "bkn": "bukan",
    "gpp": "tidak apa-apa", "otw": "on the way", "btw": "by the way",
    "cmn": "cuman", "lg": "lagi", "lgi": "lagi", "aja": "saja",
    "aj": "saja", "bs": "bisa", "bsa": "bisa", "dr": "dari",
    "dri": "dari", "utk": "untuk", "trs": "terus", "trus": "terus",
    "msh": "masih", "masi": "masih", "jd": "jadi", "jdi": "jadi",
    "skrg": "sekarang", "skrng": "sekarang",
    # Gaming terms
    "noob": "pemula", "newbie": "pemula", "pro": "profesional",
    "gg": "good game", "wp": "well played", "afk": "away from keyboard",
    "ez": "easy", "lag": "lag", "dc": "disconnect", "bcs": "karena",
}


def normalize_leetspeak(text: str) -> str:
    """Konversi angka/simbol leetspeak ke huruf biasa."""
    result = []
    for i, char in enumerate(text):
        if char in LEETSPEAK_MAP:
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
    """Ekspansi singkatan & slang gamer ke bentuk lengkap."""
    words = text.split()
    expanded = [SLANG_DICT.get(w, w) for w in words]
    return " ".join(expanded)


def clean_text(text: str) -> str:
    """Pipeline pembersihan teks (sama persis dengan training)."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = normalize_leetspeak(text)
    text = expand_slang(text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ══════════════════════════════════════════════
# 🎯 FUNGSI PREDIKSI
# ══════════════════════════════════════════════


def predict_toxicity(text: str) -> dict:
    """
    Prediksi label toksisitas dari teks chat.

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
        label = str(result.iloc[0, -1]) # Fallback to last column

    if "prediction_score" in result.columns:
        score = result["prediction_score"].iloc[0]
    elif "Score" in result.columns:
        score = result["Score"].iloc[0]
    else:
        score = 1.0 # Fallback jika score tidak ada

    return {label: float(score)}


# ══════════════════════════════════════════════
# 🎨 GRADIO INTERFACE
# ══════════════════════════════════════════════

EXAMPLES = [
    ["gg wp semua, main lagi yuk"],
    ["anjg lo noob bgt tolol"],
    ["nice play bro, carry terus"],
    ["gblk banget sih lu main"],
    ["gw afk dulu ya bentar"],
    ["dasar bego kontol gak bisa main"],
]

demo = gr.Interface(
    fn=predict_toxicity,
    inputs=gr.Textbox(
        label="💬 Masukkan Chat",
        placeholder="Ketik chat game di sini... (contoh: 'gg wp semua')",
        lines=3,
    ),
    outputs=gr.Label(
        label="🎯 Hasil Prediksi",
        num_top_classes=3,
    ),
    title="🎮 Deteksi Toksisitas Chat Gamer Indonesia",
    description=(
        "Model NLP untuk mendeteksi apakah chat dalam game online Indonesia "
        "bersifat **toxic** atau **non-toxic**.\n\n"
        "Model ini dilatih menggunakan PyCaret dengan TF-IDF + "
        "custom preprocessing untuk slang & leetspeak gamer Indonesia."
    ),
    examples=EXAMPLES,
    cache_examples=False,
    theme=gr.themes.Soft(),
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
