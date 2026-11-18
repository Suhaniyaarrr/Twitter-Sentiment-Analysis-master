# Twitter Sentiment Analysis

A binary tweet sentiment classifier using TF-IDF features and Logistic Regression, with a Streamlit dashboard for exploration, evaluation, and prediction. The repository also retains `main.py`, a small TextBlob-based analysis of keyword matches in the local dataset.

## Requirements and setup

Use Python 3.9 or newer. From the project directory, install dependencies with:

```bash
python -m pip install -r requirements.txt
```

The training data file `tweets.csv` should be in the project directory. The included saved model and vectorizer are in `models/`.

## Run the dashboard

```bash
streamlit run app.py
```

The dashboard loads the saved artifacts; it does not train automatically. If either artifact is missing, create them explicitly with the training command below.

## Deploy on Streamlit Community Cloud

Create a Community Cloud app from this repository and set the entry point to `app.py` at the repository root. Keep `requirements.txt`, `config.py`, the Python modules, and both files under `models/` in the deployed Git branch. The model artifacts are required and should be committed; they are not excluded by `.gitignore`.

The 239 MB `tweets.csv` is intentionally excluded from Git. Without that optional local dataset, prediction remains available; dataset charts, word clouds, exploration, and dataset-based evaluation are unavailable. Enabling those tabs in a deployment requires a separate dataset delivery/storage setup.

## Train or evaluate a model

```bash
python train.py
python train.py --samples 100000
```

Training reads the CSV in chunks and samples up to the configured number of labeled rows. The training pipeline fits TF-IDF on the training split only, then evaluates on the held-out split and saves the model and vectorizer under `models/`.

## Predict from Python

```python
from predict import predict_sentiment

result = predict_sentiment("I love this product!")
print(result["label"], result["confidence"])
```

## Legacy keyword analysis

`python main.py` runs the older TextBlob workflow against `tweets.csv`; it does not fetch tweets from Twitter. It writes analyzed tweet text to `result.csv` and displays charts.

## License

See [License.txt](License.txt).
