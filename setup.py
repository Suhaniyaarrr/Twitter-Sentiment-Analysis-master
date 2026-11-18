from pathlib import Path

from setuptools import setup


setup(
    name="twitter-sentiment-analysis",
    version="2.0.0",
    description="Streamlit dashboard and TF-IDF sentiment classifier for tweets.",
    long_description=Path(__file__).with_name("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    py_modules=["app", "config", "evaluate", "main", "predict", "preprocessing", "train"],
    python_requires=">=3.9",
    install_requires=[
        "streamlit>=1.28.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "nltk>=3.8.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "wordcloud>=1.9.0",
        "joblib>=1.3.0",
        "plotly>=5.15.0",
        "textblob>=0.17.1",
    ],
)
