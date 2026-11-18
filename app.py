"""
Tweet Sentiment Analysis — Streamlit Dashboard
"""
import os
from io import StringIO
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split

import config
from preprocessing import preprocess_series
from predict import load_model, predict_sentiment
from evaluate import evaluate_model
from train import load_dataset

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Tweet Sentiment Analyzer",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Visual theme ───────────────────────────────────────────────────────────────
px.defaults.template = "plotly_dark"
px.defaults.color_discrete_map = {"Positive": "#39d98a", "Negative": "#ff6685"}
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
    :root {
        --ink: #f1f5ff;
        --muted: #9aa9c5;
        --line: rgba(150, 176, 224, .16);
        --panel: rgba(18, 31, 57, .76);
        --blue: #67a8ff;
        --mint: #39d98a;
        --rose: #ff6685;
    }
    html, body, [class*="css"] { font-family: 'DM Sans', 'Segoe UI', sans-serif; }
    .stApp {
        color: var(--ink);
        background:
            radial-gradient(ellipse at 12% 0%, rgba(46, 108, 208, .24), transparent 34%),
            radial-gradient(ellipse at 92% 16%, rgba(85, 65, 177, .17), transparent 30%),
            linear-gradient(145deg, #07111f 0%, #0b1628 52%, #10152b 100%);
    }
    [data-testid="stHeader"] { background: rgba(7, 17, 31, .68); }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(14, 27, 49, .98), rgba(12, 21, 39, .98));
        border-right: 1px solid var(--line);
    }
    [data-testid="stSidebar"] > div:first-child { padding-top: 1.6rem; }
    .block-container { max-width: 1440px; padding-top: 2.2rem; padding-bottom: 4rem; }
    h1, h2, h3, h4 { color: var(--ink); font-family: 'Space Grotesk', 'Segoe UI', sans-serif; letter-spacing: -.025em; }
    p, label, [data-testid="stCaptionContainer"] { color: var(--muted); }
    .main-header {
        font-family: 'Space Grotesk', 'Segoe UI', sans-serif;
        font-size: clamp(2.4rem, 5vw, 4.2rem);
        line-height: 1.04;
        font-weight: 700;
        background: linear-gradient(100deg, #f4f7ff 5%, #8dc4ff 52%, #b9a8ff 96%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: .25rem 0 .65rem;
    }
    .sub-header { color: #aab8d2; font-size: 1.12rem; margin: 0 0 1.4rem; }
    .hero-kicker {
        display: inline-block; color: #a9cbff; background: rgba(78, 139, 230, .13);
        border: 1px solid rgba(112, 166, 245, .24); border-radius: 999px;
        padding: .38rem .8rem; font-size: .78rem; font-weight: 700;
        letter-spacing: .12em; text-transform: uppercase;
    }
    [data-testid="stMetric"] {
        background: linear-gradient(145deg, rgba(24, 41, 71, .92), rgba(16, 29, 52, .82));
        border: 1px solid var(--line); border-radius: 18px; padding: 1.1rem 1.2rem;
        box-shadow: 0 14px 40px rgba(0, 0, 0, .16); transition: transform .2s ease, border-color .2s ease;
    }
    [data-testid="stMetric"]:hover { transform: translateY(-3px); border-color: rgba(103, 168, 255, .5); }
    [data-testid="stMetricLabel"] p { color: #aab8d2; font-weight: 600; }
    [data-testid="stMetricValue"] { color: #f5f8ff; font-family: 'Space Grotesk', sans-serif; }
    [data-testid="stMetricDelta"] { font-weight: 600; }
    [data-testid="stTabs"] [role="tablist"] { gap: .45rem; border-bottom: 1px solid var(--line); }
    [data-testid="stTabs"] button[role="tab"] {
        color: #9eacc6; border-radius: 11px 11px 0 0; padding: .72rem 1rem;
        transition: color .18s ease, background .18s ease;
    }
    [data-testid="stTabs"] button[role="tab"]:hover { color: white; background: rgba(103, 168, 255, .08); }
    [data-testid="stTabs"] button[aria-selected="true"] { color: #9bc8ff; border-bottom: 2px solid #67a8ff; }
    .stButton > button, [data-testid="stDownloadButton"] button {
        border: 1px solid rgba(126, 175, 255, .42); border-radius: 12px;
        color: white; background: linear-gradient(110deg, #2867c8, #644bd0);
        font-weight: 700; box-shadow: 0 8px 24px rgba(51, 100, 201, .24);
        transition: transform .18s ease, box-shadow .18s ease;
    }
    .stButton > button:hover, [data-testid="stDownloadButton"] button:hover {
        transform: translateY(-2px); box-shadow: 0 12px 28px rgba(77, 117, 220, .35); color: white;
    }
    [data-testid="stTextArea"] textarea, [data-testid="stTextInput"] input,
    [data-testid="stSelectbox"] [data-baseweb="select"] > div,
    [data-testid="stSlider"] [role="slider"] { border-radius: 11px; }
    [data-testid="stTextArea"] textarea, [data-testid="stTextInput"] input {
        background: rgba(10, 21, 39, .82); border: 1px solid var(--line); color: var(--ink);
    }
    [data-testid="stAlert"] { background: rgba(37, 61, 96, .45); border: 1px solid var(--line); border-radius: 14px; }
    [data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 14px; overflow: hidden; }
    [data-testid="stCode"] { border: 1px solid var(--line); border-radius: 14px; }
    hr { border-color: var(--line); }
    .positive { color: var(--mint); font-weight: 700; }
    .negative { color: var(--rose); font-weight: 700; }
    @media (max-width: 700px) {
        .block-container { padding: 1.2rem 1rem 3rem; }
        [data-testid="stTabs"] [role="tablist"] { overflow-x: auto; }
        [data-testid="stTabs"] button[role="tab"] { padding: .55rem .7rem; font-size: .84rem; }
    }
</style>
""", unsafe_allow_html=True)


# ── Cached helpers ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model …")
def get_model():
    return load_model()


@st.cache_data(show_spinner="Loading dataset …")
def load_dataset_sample(n_rows):
    df = load_dataset(n_rows)
    df["sentiment"] = df["target"].map(config.LABEL_MAP)
    return df


@st.cache_data(show_spinner="Evaluating model …")
def get_evaluation(_model, _vectorizer, n_rows):
    df = load_dataset_sample(n_rows)
    df["clean"] = preprocess_series(df["tweet"])
    _, X_test, _, y_test = train_test_split(
        df["clean"], df["label"],
        test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE,
        stratify=df["label"],
    )
    X_test_tfidf = _vectorizer.transform(X_test)
    return evaluate_model(_model, X_test_tfidf, y_test)


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ◈ Settings")
    sample_size = st.slider(
        "Dataset sample size", min_value=10_000, max_value=200_000,
        value=config.SAMPLE_SIZE, step=10_000,
        help="Rows used for charts & evaluation (model uses its own training set)",
    )
    st.divider()
    st.markdown("### ▤ Project Files")
    if os.path.isfile(config.DATASET_PATH):
        st.caption(f"Dataset: `tweets.csv` ({os.path.getsize(config.DATASET_PATH)/1e6:.0f} MB)")
    else:
        st.caption("Dataset: `tweets.csv` not found")
    artifacts_exist = all(os.path.isfile(path) for path in (config.MODEL_PATH, config.VECTORIZER_PATH))
    st.caption(f"Model artifacts: {'✓ Available' if artifacts_exist else '! Missing; run `python train.py`'}")
    st.divider()
    st.markdown(
        "Built with **Scikit-learn** · TF-IDF · Logistic Regression",
        help="No TextBlob, no transformers — pure ML pipeline",
    )

# ── Load model & data ─────────────────────────────────────────────────────────
model, vectorizer = get_model()
dataset_available = os.path.isfile(config.DATASET_PATH)
if dataset_available:
    df = load_dataset_sample(sample_size)
    eval_res = get_evaluation(model, vectorizer, sample_size)
else:
    df = pd.DataFrame(columns=["target", "tweet", "label", "sentiment"])
    eval_res = None
    st.info(
        "The optional `tweets.csv` dataset is not included in this deployment. "
        "Tweet prediction remains available; dataset charts and evaluation need the CSV."
    )

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_overview, tab_eval, tab_predict, tab_charts, tab_explore = st.tabs(
    ["Overview", "Evaluation", "Predict", "Charts", "Explore"]
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 1 — Overview
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_overview:
    st.markdown('<span class="hero-kicker">Social pulse · NLP insights</span>', unsafe_allow_html=True)
    st.markdown('<p class="main-header">Tweet Sentiment Analyzer</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Explore the mood behind the message with TF-IDF and Logistic Regression.</p>', unsafe_allow_html=True)
    st.markdown("")

    col1, col2, col3, col4 = st.columns(4)
    if dataset_available:
        total = len(df)
        pos = int((df["label"] == 1).sum())
        neg = int((df["label"] == 0).sum())
        col1.metric("# Total Tweets", f"{total:,}")
        col2.metric("+ Positive", f"{pos:,}", f"{pos/total*100:.1f}%")
        col3.metric("− Negative", f"{neg:,}", f"{neg/total*100:.1f}%")
        col4.metric("◎ Model Accuracy", f"{eval_res['accuracy']:.1%}")
    else:
        st.caption("Dataset statistics and accuracy metrics appear when `tweets.csv` is provided.")

    st.divider()
    st.markdown("### How It Works")
    cols = st.columns(4)
    steps = [
        ("01", "**Preprocess**", "Clean tweets: remove URLs, mentions, stopwords, then stem."),
        ("02", "**Vectorize**", "Convert text to numerical features with TF-IDF (up to 50K features)."),
        ("03", "**Train**", "Fit a Logistic Regression classifier on 80% of the data."),
        ("04", "**Predict**", "Classify new tweets as Positive or Negative with confidence scores."),
    ]
    for col, (icon, title, desc) in zip(cols, steps):
        col.markdown(f"#### {icon} {title}\n{desc}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 2 — Model Evaluation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_eval:
    st.markdown("## ◫ Model Evaluation")
    if eval_res is None:
        st.info("Evaluation needs `tweets.csv`. The saved classifier is still available in the Predict tab.")
    else:
        st.info(
            "Metrics use a deterministic sample and split from the current dataset. "
            "The project does not record which rows trained the saved model, so this is not a guaranteed held-out evaluation."
        )
        res = eval_res

        m1, m2, m3, m4 = st.columns(4)
        for col, (name, val) in zip([m1, m2, m3, m4],
            [("Accuracy", res["accuracy"]), ("Precision", res["precision"]),
             ("Recall", res["recall"]), ("F1 Score", res["f1"])]):
            col.metric(name, f"{val:.4f}")

        st.divider()
        col_cm, col_rpt = st.columns(2)

        with col_cm:
            st.markdown("### Confusion Matrix")
            labels = list(config.REVERSE_LABEL_MAP.values())
            fig_cm = px.imshow(
                res["confusion_matrix"], text_auto=True, color_continuous_scale="Blues",
                x=labels, y=labels,
                labels=dict(x="Predicted", y="Actual", color="Count"),
            )
            fig_cm.update_layout(height=400)
            st.plotly_chart(fig_cm, use_container_width=True)

        with col_rpt:
            st.markdown("### Classification Report")
            st.code(res["classification_report"], language=None)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 3 — Predict Sentiment
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_predict:
    st.markdown("## → Predict Sentiment")

    input_mode = st.radio("Input mode", ["Single tweet", "Batch (one per line)"], horizontal=True)

    if input_mode == "Single tweet":
        tweet_input = st.text_area(
            "Enter a tweet:",
            placeholder="e.g. I love this product! It's amazing 😍",
            height=100,
        )
        if st.button("Analyze →", type="primary") and tweet_input.strip():
            result = predict_sentiment(tweet_input, model, vectorizer)
            label = result["label"]
            conf = result["confidence"]
            probs = result["probabilities"]

            sign = "+" if label == "Positive" else "−"
            color = "positive" if label == "Positive" else "negative"
            st.markdown(f'### <span class="{color}">{sign} {label}</span> ({conf:.1%} confidence)',
                        unsafe_allow_html=True)

            # Confidence gauge
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number", value=conf * 100,
                title={"text": "Confidence", "font": {"color": "#b6c5df"}},
                number={"suffix": "%", "font": {"color": "#f1f5ff", "family": "Space Grotesk"}},
                gauge={
                    "axis": {"range": [50, 100]},
                    "bar": {"color": "#39d98a" if label == "Positive" else "#ff6685"},
                    "bgcolor": "rgba(15, 27, 48, .8)",
                    "bordercolor": "rgba(150, 176, 224, .2)",
                    "steps": [
                        {"range": [50, 70], "color": "#263650"},
                        {"range": [70, 85], "color": "#304664"},
                        {"range": [85, 100], "color": "#3c5676"},
                    ],
                },
            ))
            fig_gauge.update_layout(height=250, paper_bgcolor="rgba(0,0,0,0)", margin={"t": 45, "b": 10})
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Probability breakdown
            fig_bar = px.bar(
                x=list(probs.keys()), y=list(probs.values()),
                color=list(probs.keys()),
                color_discrete_map={"Positive": "#39d98a", "Negative": "#ff6685"},
                labels={"x": "Sentiment", "y": "Probability"},
            )
            fig_bar.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)

    else:
        batch_input = st.text_area(
            "Enter tweets (one per line):",
            placeholder="I love it!\nThis is terrible.\nWhat a great day!",
            height=200,
        )
        if st.button("Analyze batch →", type="primary") and batch_input.strip():
            tweets = [t.strip() for t in batch_input.strip().split("\n") if t.strip()]
            results = []
            for t in tweets:
                r = predict_sentiment(t, model, vectorizer)
                results.append({
                    "Tweet": t, "Sentiment": r["label"],
                    "Confidence": f"{r['confidence']:.1%}",
                    "Positive probability": f"{r['probabilities']['Positive']:.1%}",
                    "Negative probability": f"{r['probabilities']['Negative']:.1%}",
                })
            results_df = pd.DataFrame(results)
            st.success(f"Analyzed {len(results_df)} tweets")
            st.dataframe(results_df, use_container_width=True, hide_index=True)
            csv_buffer = StringIO()
            results_df.to_csv(csv_buffer, index=False)
            st.download_button(
                "↓ Download results as CSV", csv_buffer.getvalue(),
                file_name="sentiment_predictions.csv", mime="text/csv",
            )

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 4 — Sentiment Charts
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_charts:
    if not dataset_available:
        st.info("Charts need the optional `tweets.csv` dataset. Tweet prediction is available in the Predict tab.")
    else:
        st.markdown("## ∿ Sentiment & Language Patterns")
        st.caption("Interactive views of sentiment balance, tweet length, and frequently used terms.")

        counts = df["sentiment"].value_counts()
        col_pie, col_bar = st.columns(2)

        with col_pie:
            fig_pie = px.pie(
                names=counts.index, values=counts.values,
                color=counts.index,
                color_discrete_map={"Positive": "#39d98a", "Negative": "#ff6685"},
                title="Sentiment Split", hole=0.4,
            )
            fig_pie.update_traces(textinfo="percent+label")
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_bar:
            fig_bar = px.bar(
                x=counts.index, y=counts.values, color=counts.index,
                color_discrete_map={"Positive": "#39d98a", "Negative": "#ff6685"},
                title="Tweet Counts by Sentiment",
                labels={"x": "Sentiment", "y": "Count"},
            )
            fig_bar.update_layout(showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("### ↔ Tweet Length Patterns")
        df_len = df.copy()
        df_len["length"] = df_len["tweet"].astype(str).str.len()
        df_len["word_count"] = df_len["tweet"].astype(str).str.split().str.len()
        fig_hist = px.histogram(
            df_len, x="length", color="sentiment", nbins=50,
            color_discrete_map={"Positive": "#39d98a", "Negative": "#ff6685"},
            barmode="overlay", opacity=0.7,
            labels={"length": "Characters", "count": "Tweets"},
            title="Tweet character counts by sentiment",
        )
        fig_hist.update_layout(height=360, xaxis_title="Characters per tweet", yaxis_title="Tweet count")

        band_col, words_col = st.columns([1.15, 0.85])
        with band_col:
            st.plotly_chart(fig_hist, use_container_width=True)
        with words_col:
            fig_words = px.violin(
                df_len, x="sentiment", y="word_count", color="sentiment", box=True,
                points=False, color_discrete_map={"Positive": "#39d98a", "Negative": "#ff6685"},
                labels={"word_count": "Words per tweet", "sentiment": "Sentiment"},
                title="Word count by sentiment",
            )
            fig_words.update_layout(height=360, showlegend=False)
            st.plotly_chart(fig_words, use_container_width=True)

        try:
            df_len["length_band"] = pd.qcut(df_len["length"], q=6, duplicates="drop")
            length_summary = (
                df_len.groupby("length_band", observed=True)
                .agg(tweet_count=("label", "size"), positive_share=("label", "mean"))
                .reset_index()
            )
            length_summary["length_band"] = length_summary["length_band"].astype(str)
            fig_length_sentiment = px.bar(
                length_summary, x="length_band", y="positive_share", color="positive_share",
                color_continuous_scale=["#ff6685", "#7784d8", "#39d98a"],
                hover_data={"tweet_count": True, "positive_share": ":.1%"},
                labels={"length_band": "Tweet character range", "positive_share": "Positive share"},
                title="Positive share across tweet-length bands",
            )
            fig_length_sentiment.update_layout(height=370, yaxis_tickformat=".0%", coloraxis_showscale=False)
            st.plotly_chart(fig_length_sentiment, use_container_width=True)
        except ValueError:
            st.info("There is not enough variation in tweet lengths to group this sample into length bands.")

        st.markdown("### ◌ Frequent Terms")
        term_filter, term_limit = st.columns([1, 1])
        with term_filter:
            term_sentiment = st.selectbox("Sentiment for term chart", ["All", "Positive", "Negative"])
        with term_limit:
            top_n = st.slider("Number of terms", min_value=8, max_value=30, value=15, step=2)
        term_data = df[["tweet", "sentiment"]]
        if term_sentiment != "All":
            term_data = term_data[term_data["sentiment"] == term_sentiment]
        term_data = term_data.head(10_000)
        if not term_data.empty and term_data["tweet"].astype(str).str.strip().any():
            term_vectorizer = CountVectorizer(stop_words="english", ngram_range=(1, 2), max_features=10_000)
            try:
                term_matrix = term_vectorizer.fit_transform(term_data["tweet"].fillna("").astype(str))
                term_counts = term_matrix.sum(axis=0).A1
                term_names = term_vectorizer.get_feature_names_out()
                top_indices = term_counts.argsort()[-top_n:][::-1]
                terms_df = pd.DataFrame({"Term": term_names[top_indices], "Mentions": term_counts[top_indices]})
                fig_terms = px.bar(
                    terms_df.sort_values("Mentions"), x="Mentions", y="Term", orientation="h",
                    color="Mentions", color_continuous_scale=["#355787", "#67a8ff", "#b9a8ff"],
                    title=f"Top {top_n} terms · {term_sentiment} tweets",
                )
                fig_terms.update_layout(height=max(380, top_n * 27), coloraxis_showscale=False, yaxis_title="")
                st.plotly_chart(fig_terms, use_container_width=True)
            except ValueError:
                st.info("No recognizable terms were found for this selection.")
        else:
            st.info("No tweets are available for the selected sentiment.")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 5 — Word Cloud & Dataset Explorer
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_explore:
    if not dataset_available:
        st.info("Word clouds and dataset search need the optional `tweets.csv` dataset.")
    else:
        st.markdown("## ◌ Word Clouds")

        wc_col1, wc_col2 = st.columns(2)
        wc_sample = df.sample(n=min(20_000, len(df)), random_state=config.RANDOM_STATE)

        for col, label, lbl_int, color in [
            (wc_col1, "+ Positive", 1, "YlGn"),
            (wc_col2, "− Negative", 0, "OrRd"),
        ]:
            with col:
                st.markdown(f"### {label}")
                text = " ".join(wc_sample[wc_sample["label"] == lbl_int]["tweet"].astype(str))
                if text.strip():
                    wc = WordCloud(
                        width=800, height=400, background_color="#111d31",
                        colormap=color, max_words=150,
                    ).generate(text)
                    fig_wc, ax = plt.subplots(figsize=(8, 4), facecolor="#111d31")
                    ax.imshow(wc, interpolation="bilinear")
                    ax.axis("off")
                    ax.set_facecolor("#111d31")
                    st.pyplot(fig_wc)
                    plt.close(fig_wc)

        st.divider()
        st.markdown("## ⌕ Dataset Explorer")

        col_filter, col_search = st.columns([1, 2])
        with col_filter:
            sentiment_filter = st.selectbox("Filter by sentiment", ["All", "Positive", "Negative"])
        with col_search:
            search_term = st.text_input("Search tweets", placeholder="Enter keyword …")

        display_df = df.copy()
        if sentiment_filter != "All":
            display_df = display_df[display_df["sentiment"] == sentiment_filter]
        if search_term:
            display_df = display_df[display_df["tweet"].astype(str).str.contains(search_term, case=False, na=False)]

        st.caption(f"Showing {len(display_df):,} tweets")
        st.dataframe(
            display_df[["tweet", "sentiment"]].head(500).reset_index(drop=True),
            use_container_width=True, hide_index=True,
            column_config={
                "tweet": st.column_config.TextColumn("Tweet", width="large"),
                "sentiment": st.column_config.TextColumn("Sentiment", width="small"),
            },
        )
