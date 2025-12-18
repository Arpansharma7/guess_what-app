import streamlit as st
import pandas as pd
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")
# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Sentiment AI",
    page_icon="$",
    layout="wide"
)

# ---------------- CSS (VISIBLE UPGRADE) ----------------
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #0b1220, #0f172a);
}
.app-shell {
    background: rgba(255,255,255,0.06);
    backdrop-filter: blur(12px);
    border-radius: 18px;
    padding: 22px 26px;
    margin-bottom: 18px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.brand {
    font-size: 22px;
    font-weight: 700;
    color: #e5e7eb;
}
.badge {
    font-size: 12px;
    color: #c7d2fe;
    background: rgba(99,102,241,0.18);
    padding: 6px 10px;
    border-radius: 999px;
}
.card {
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(14px);
    border-radius: 18px;
    padding: 26px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.35);
}
.h1 {
    font-size: 36px;
    font-weight: 800;
    color: #f9fafb;
    margin-bottom: 4px;
}
.sub {
    color: #cbd5f5;
    margin-bottom: 22px;
}
.result-pos {
    color: #22c55e;
    font-size: 22px;
    font-weight: 700;
}
.result-neg {
    color: #ef4444;
    font-size: 22px;
    font-weight: 700;
}
.small {
    color: #9ca3af;
}
</style>
""", unsafe_allow_html=True)

# ---------------- APP SHELL ----------------
st.markdown("""
<div class="app-shell">
  <div class="brand">Sentiment AI</div>
  <div class="badge">Made By: Arpan Sharma</div>
</div>
""", unsafe_allow_html=True)

# ---------------- NLP ----------------
def normalized(text):
    text = str(text).lower()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    cleaned = [w for w in tokens if w not in stop_words and w not in string.punctuation]
    return " ".join(cleaned)

# ---------------- TRAIN ONCE (CACHED) ----------------
@st.cache_resource
def train_model():
    df = pd.read_csv("tweets.csv", header=None)
    df.columns = ["id", "topic", "sentiment", "text"]

    df = df[df["sentiment"].isin(["Positive", "Negative"])]
    df["sentiment"] = df["sentiment"].map({"Negative": 0, "Positive": 1})

    df["clean_text"] = df["text"].apply(normalized)

    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2)
    )
    X = vectorizer.fit_transform(df["clean_text"])
    y = df["sentiment"]

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    return model, vectorizer

model, vectorizer = train_model()

# ---------------- MAIN LAYOUT ----------------
left, right = st.columns([1.25, 1])

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="h1">Sentiment Analyzer</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub">Analyze tweets with a fast classical ML pipeline.</div>', unsafe_allow_html=True)

    user_text = st.text_area(
        "Enter a text",
        height=160,
        placeholder="Type or paste a tweet here…"
    )

    analyze = st.button("Analyze Sentiment")
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Result")

    if analyze and user_text.strip():
        cleaned = normalized(user_text)
        X_test = vectorizer.transform([cleaned])
        pred = model.predict(X_test)[0]
        prob = model.predict_proba(X_test)[0]
        confidence = float(prob[pred])

        if pred == 1:
            st.markdown('<div class="result-pos">Positive Sentiment</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-neg">Negative Sentiment</div>', unsafe_allow_html=True)

        st.progress(confidence)
        st.caption(f"Confidence: {confidence:.2f}")

    else:
        st.markdown('<div class="small">Run an analysis to see results here.</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown('<div class="small" style="margin-top:16px;">An end-to-end NLP application</div>', unsafe_allow_html=True)

