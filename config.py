"""
Central configuration for the Tweet Sentiment Analysis project.
"""
import os

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "tweets.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "sentiment_model.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")

# ── Dataset ────────────────────────────────────────────────────────────────────
COLUMN_NAMES = ["target", "id", "date", "flag", "user", "tweet"]
LABEL_MAP = {0: "Negative", 4: "Positive"}
BINARY_LABEL_MAP = {0: 0, 4: 1}          # 0 → 0 (Neg), 4 → 1 (Pos)
REVERSE_LABEL_MAP = {0: "Negative", 1: "Positive"}

# ── Training ───────────────────────────────────────────────────────────────────
SAMPLE_SIZE = 100_000        # rows to sample for training (set None for full dataset)
TEST_SIZE = 0.2
RANDOM_STATE = 42
MAX_TFIDF_FEATURES = 50_000
NGRAM_RANGE = (1, 2)         # unigrams + bigrams
