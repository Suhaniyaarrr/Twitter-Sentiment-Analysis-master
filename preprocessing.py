"""
Text preprocessing pipeline for tweet sentiment analysis.
"""
import re
from nltk.stem import PorterStemmer

_stemmer = PorterStemmer()

# English stopwords (from NLTK) — embedded to avoid runtime download issues
_NLTK_STOPS = {
    'a', 'about', 'above', 'after', 'again', 'against', 'ain', 'all', 'am',
    'an', 'and', 'any', 'are', 'aren', "aren't", 'as', 'at', 'be', 'because',
    'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'can',
    'couldn', "couldn't", 'd', 'did', 'didn', "didn't", 'do', 'does', 'doesn',
    "doesn't", 'doing', 'don', "don't", 'down', 'during', 'each', 'few', 'for',
    'from', 'further', 'had', 'hadn', "hadn't", 'has', 'hasn', "hasn't", 'have',
    'haven', "haven't", 'having', 'he', "he'd", "he'll", 'her', 'here', 'hers',
    'herself', "he's", 'him', 'himself', 'his', 'how', 'i', "i'd", 'if', "i'll",
    "i'm", 'in', 'into', 'is', 'isn', "isn't", 'it', "it'd", "it'll", "it's",
    'its', 'itself', "i've", 'just', 'll', 'm', 'ma', 'me', 'mightn', "mightn't",
    'more', 'most', 'mustn', "mustn't", 'my', 'myself', 'needn', "needn't", 'no',
    'nor', 'not', 'now', 'o', 'of', 'off', 'on', 'once', 'only', 'or', 'other',
    'our', 'ours', 'ourselves', 'out', 'over', 'own', 're', 's', 'same', 'shan',
    "shan't", 'she', "she'd", "she'll", "she's", 'should', 'shouldn', "shouldn't",
    "should've", 'so', 'some', 'such', 't', 'than', 'that', "that'll", 'the',
    'their', 'theirs', 'them', 'themselves', 'then', 'there', 'these', 'they',
    "they'd", "they'll", "they're", "they've", 'this', 'those', 'through', 'to',
    'too', 'under', 'until', 'up', 've', 'very', 'was', 'wasn', "wasn't", 'we',
    "we'd", "we'll", "we're", 'were', 'weren', "weren't", "we've", 'what', 'when',
    'where', 'which', 'while', 'who', 'whom', 'why', 'will', 'with', 'won',
    "won't", 'wouldn', "wouldn't", 'y', 'you', "you'd", "you'll", 'your',
    "you're", 'yours', 'yourself', 'yourselves', "you've",
}

# Words that carry sentiment — keep them even though they appear in NLTK stops
_KEEP_WORDS = {"not", "no", "nor", "never", "neither", "nobody", "nothing",
               "nowhere", "hardly", "scarcely", "barely", "but", "very"}

_stop_words = _NLTK_STOPS - _KEEP_WORDS


def preprocess_tweet(text: str) -> str:
    """Clean and normalise a single tweet string."""
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)       # URLs
    text = re.sub(r"@\w+", "", text)                    # @mentions
    text = re.sub(r"#", "", text)                       # hashtag symbol (keep word)
    text = re.sub(r"[^a-z\s]", "", text)                # non-alpha characters
    text = re.sub(r"\s+", " ", text).strip()            # collapse whitespace

    tokens = text.split()
    tokens = [_stemmer.stem(w) for w in tokens if w not in _stop_words]
    return " ".join(tokens)


def preprocess_series(series):
    """Apply preprocessing to a pandas Series; returns a new Series."""
    return series.astype(str).apply(preprocess_tweet)
