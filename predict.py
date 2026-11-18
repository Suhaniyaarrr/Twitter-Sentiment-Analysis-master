"""
Load the saved model and predict sentiment for new text.
"""
import os
import joblib

import config
from preprocessing import preprocess_tweet


def _ensure_model():
    """Fail clearly if the saved model artifacts are missing."""
    missing = [path for path in (config.MODEL_PATH, config.VECTORIZER_PATH)
               if not os.path.isfile(path)]
    if missing:
        paths = ", ".join(missing)
        raise FileNotFoundError(
            f"Required model artifact(s) missing: {paths}. Run `python train.py` to create them."
        )


def load_model():
    """Load and return the saved (model, vectorizer) pair."""
    _ensure_model()
    model = joblib.load(config.MODEL_PATH)
    vectorizer = joblib.load(config.VECTORIZER_PATH)
    return model, vectorizer


def predict_sentiment(text: str, model=None, vectorizer=None):
    """
    Predict sentiment for a single tweet.

    Returns dict:  {"label": "Positive"/"Negative",
                    "confidence": float 0-1,
                    "probabilities": {"Negative": float, "Positive": float}}
    """
    if model is None or vectorizer is None:
        model, vectorizer = load_model()

    clean = preprocess_tweet(text)
    tfidf = vectorizer.transform([clean])
    proba = model.predict_proba(tfidf)[0]
    classes = [int(label) for label in model.classes_]
    if set(classes) != set(config.REVERSE_LABEL_MAP):
        raise ValueError(f"Unexpected model classes: {classes}")
    probability_by_label = dict(zip(classes, proba))
    pred = max(probability_by_label, key=probability_by_label.get)

    return {
        "label": config.REVERSE_LABEL_MAP[pred],
        "confidence": float(probability_by_label[pred]),
        "probabilities": {
            "Negative": float(probability_by_label[0]),
            "Positive": float(probability_by_label[1]),
        },
    }


def predict_batch(texts: list, model=None, vectorizer=None):
    """Predict sentiment for a list of texts. Returns list of result dicts."""
    if model is None or vectorizer is None:
        model, vectorizer = load_model()
    return [predict_sentiment(t, model, vectorizer) for t in texts]
