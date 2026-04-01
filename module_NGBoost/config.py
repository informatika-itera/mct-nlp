import os

# ── paths ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
PLOT_DIR = os.path.join(BASE_DIR, "plots")

for d in (DATA_DIR, MODEL_DIR, PLOT_DIR):
    os.makedirs(d, exist_ok=True)

# ── dataset ────────────────────────────────────────────────────────────
KAGGLE_DATASET = "lakshmi25npathi/imdb-dataset-of-50k-movie-reviews"
CSV_FILENAME = "IMDB Dataset.csv"
TEXT_COL = "review"
LABEL_COL = "sentiment"

# ── preprocessing ──────────────────────────────────────────────────────
CONTRACTION_MAP = {
    "ain't": "is not", "aren't": "are not", "can't": "cannot",
    "can't've": "cannot have", "could've": "could have",
    "couldn't": "could not", "didn't": "did not", "doesn't": "does not",
    "don't": "do not", "hadn't": "had not", "hasn't": "has not",
    "haven't": "have not", "he'd": "he would", "he'll": "he will",
    "he's": "he is", "i'd": "i would", "i'll": "i will",
    "i'm": "i am", "i've": "i have", "isn't": "is not",
    "it's": "it is", "let's": "let us", "might've": "might have",
    "mustn't": "must not", "she'd": "she would", "she'll": "she will",
    "she's": "she is", "should've": "should have", "shouldn't": "should not",
    "that's": "that is", "there's": "there is", "they'd": "they would",
    "they'll": "they will", "they're": "they are", "they've": "they have",
    "we'd": "we would", "we're": "we are", "we've": "we have",
    "weren't": "were not", "what's": "what is", "won't": "will not",
    "wouldn't": "would not", "you'd": "you would", "you'll": "you will",
    "you're": "you are", "you've": "you have",
}

# ── training ───────────────────────────────────────────────────────────
RANDOM_SEED = 42
SAMPLE_SIZE = 10_000          # None = pakai semua data (lambat)
TEST_SIZE = 0.2
TFIDF_MAX_FEATURES = 5_000

# ngboost hyper-parameters
NGB_N_ESTIMATORS = 200
NGB_LEARNING_RATE = 0.05
NGB_MINIBATCH_FRAC = 0.5     # stochastic mini-batch fraction
NGB_VERBOSE_EVAL = 25        # print log setiap N iterasi
