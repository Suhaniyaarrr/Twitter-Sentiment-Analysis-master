"""
Train a TF-IDF + Logistic Regression sentiment model and save it to disk.

Usage:
    python train.py              # train with default sample size from config
    python train.py --samples N  # override sample size
"""
import os
import argparse
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import config
from preprocessing import preprocess_series


def load_dataset(sample_size=None):
    """Read the CSV in chunks and retain a deterministic uniform sample."""
    if sample_size is not None and sample_size < 1:
        raise ValueError("sample_size must be a positive integer or None")
    print(f"Loading dataset from {config.DATASET_PATH} …")
    rng = np.random.default_rng(config.RANDOM_STATE)
    chunks = pd.read_csv(
        config.DATASET_PATH, header=None, names=config.COLUMN_NAMES,
        encoding="latin-1", usecols=["target", "tweet"], chunksize=50_000,
    )
    selected = []
    priorities = np.empty(0)
    total = 0

    for chunk in chunks:
        chunk = chunk[chunk["target"].isin(config.BINARY_LABEL_MAP.keys())]
        if chunk.empty:
            continue
        total += len(chunk)
        if sample_size is None:
            selected.append(chunk)
            continue
        chunk_priorities = rng.random(len(chunk))
        candidates = pd.concat(selected + [chunk], ignore_index=True) if selected else chunk.reset_index(drop=True)
        candidate_priorities = np.concatenate([priorities, chunk_priorities])
        keep = min(sample_size, len(candidates))
        indices = np.argpartition(candidate_priorities, -keep)[-keep:]
        selected = [candidates.iloc[indices].reset_index(drop=True)]
        priorities = candidate_priorities[indices]

    df = pd.concat(selected, ignore_index=True) if selected else pd.DataFrame(columns=["target", "tweet"])
    if sample_size is not None and total > sample_size:
        print(f"Sampled {len(df):,} of {total:,} labeled rows.")
    else:
        print(f"Using {len(df):,} labeled rows.")

    df["label"] = df["target"].map(config.BINARY_LABEL_MAP)
    return df


def train_model(sample_size=None):
    """Full training pipeline. Returns (model, vectorizer, X_test, y_test)."""
    sample_size = config.SAMPLE_SIZE if sample_size is None else sample_size

    # 1. Load & preprocess
    df = load_dataset(sample_size)
    print("Preprocessing tweets …")
    t0 = time.time()
    df["clean"] = preprocess_series(df["tweet"])
    print(f"  Preprocessing done in {time.time() - t0:.1f}s")

    # 2. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean"], df["label"],
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=df["label"],
    )

    # 3. TF-IDF
    print("Fitting TF-IDF vectorizer …")
    vectorizer = TfidfVectorizer(
        max_features=config.MAX_TFIDF_FEATURES,
        ngram_range=config.NGRAM_RANGE,
        sublinear_tf=True,
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # 4. Logistic Regression
    print("Training Logistic Regression …")
    t0 = time.time()
    model = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
    model.fit(X_train_tfidf, y_train)
    print(f"  Training done in {time.time() - t0:.1f}s")

    # 5. Save
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    joblib.dump(model, config.MODEL_PATH)
    joblib.dump(vectorizer, config.VECTORIZER_PATH)
    print(f"Model saved to {config.MODEL_DIR}/")

    return model, vectorizer, X_test_tfidf, y_test


# ── CLI entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train sentiment model")
    parser.add_argument("--samples", type=int, default=None,
                        help="Number of rows to sample (default: config value)")
    args = parser.parse_args()

    from evaluate import print_evaluation
    model, vectorizer, X_test, y_test = train_model(args.samples)
    print_evaluation(model, X_test, y_test)
