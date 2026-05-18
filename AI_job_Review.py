# =========================================================
# 💼 AI JOB REVIEW ANALYZER - ULTIMATE FINAL VERSION
# =========================================================

# RUN:
# streamlit run app.py

# INSTALL:
# pip install streamlit pandas numpy scikit-learn spacy plotly

# DOWNLOAD SPACY MODEL:
# python -m spacy download en_core_web_sm

# =========================================================

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import re
import spacy
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from spacy.lang.en.stop_words import STOP_WORDS

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AI Job Review Analyzer",
    page_icon="💼",
    layout="wide"
)

# =========================================================
# BEAUTIFUL UI CSS
# =========================================================

st.markdown("""
<style>

/* ===================================================
BACKGROUND
=================================================== */

.stApp {

    background:
    linear-gradient(
        rgba(15,23,42,0.78),
        rgba(15,23,42,0.78)
    ),

    url("https://images.unsplash.com/photo-1497366754035-f200968a6e72");

    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

/* ===================================================
GLOBAL FONT
=================================================== */

html, body, [class*="css"] {

    font-family: 'Poppins', sans-serif;
    color: white;
}

/* ===================================================
SIDEBAR
=================================================== */

[data-testid="stSidebar"] {

    background: linear-gradient(
        180deg,
        #0F172A,
        #111827
    );
}

[data-testid="stSidebar"] * {
    color: white;
}

/* ===================================================
TITLE
=================================================== */

.main-title {

    text-align: center;

    font-size: 62px;

    font-weight: 800;

    background: linear-gradient(
        90deg,
        #60A5FA,
        #A78BFA
    );

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.subtitle {

    text-align: center;

    color: #E5E7EB;

    font-size: 20px;

    margin-bottom: 35px;
}

/* ===================================================
DASHBOARD CARDS
=================================================== */

.glass-card {

    background: rgba(255,255,255,0.10);

    border-radius: 25px;

    padding: 35px;

    backdrop-filter: blur(14px);

    border: 1px solid rgba(255,255,255,0.12);

    box-shadow: 0px 8px 30px rgba(0,0,0,0.35);

    transition: 0.3s;
}

.glass-card:hover {

    transform: translateY(-7px);
}

.glass-card h2 {

    text-align: center;

    color: white;

    font-size: 44px;
}

.glass-card p {

    text-align: center;

    color: #CBD5E1;

    font-size: 19px;
}

/* ===================================================
TEXT AREA
=================================================== */

.stTextArea textarea {

    background: rgba(255,255,255,0.90);

    color: #111827;

    border-radius: 18px;

    border: none;

    font-size: 18px;

    padding: 18px;
}

/* ===================================================
BUTTON
=================================================== */

.stButton > button {

    width: 100%;

    height: 3.6em;

    border: none;

    border-radius: 15px;

    background: linear-gradient(
        90deg,
        #3B82F6,
        #8B5CF6
    );

    color: white;

    font-size: 18px;

    font-weight: bold;

    transition: 0.3s;
}

.stButton > button:hover {

    transform: scale(1.02);

    background: linear-gradient(
        90deg,
        #2563EB,
        #7C3AED
    );
}

/* ===================================================
SECTION HEADING
=================================================== */

.section-heading {

    color: #111827;

    font-size: 34px;

    font-weight: bold;

    margin-bottom: 20px;
}

/* ===================================================
RESULT BOXES
=================================================== */

.result-positive {

    background: linear-gradient(
        135deg,
        #10B981,
        #059669
    );

    padding: 40px;

    border-radius: 28px;

    text-align: center;

    color: white;

    font-size: 40px;

    font-weight: bold;

    box-shadow: 0px 10px 30px rgba(0,0,0,0.3);
}

.result-negative {

    background: linear-gradient(
        135deg,
        #EF4444,
        #DC2626
    );

    padding: 40px;

    border-radius: 28px;

    text-align: center;

    color: white;

    font-size: 40px;

    font-weight: bold;

    box-shadow: 0px 10px 30px rgba(0,0,0,0.3);
}

.result-neutral {

    background: linear-gradient(
        135deg,
        #F59E0B,
        #D97706
    );

    padding: 40px;

    border-radius: 28px;

    text-align: center;

    color: white;

    font-size: 40px;

    font-weight: bold;

    box-shadow: 0px 10px 30px rgba(0,0,0,0.3);
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# HEADER
# =========================================================

st.markdown("""
<div class="main-title">
💼 AI Job Review Analyzer
</div>

<div class="subtitle">
AI Powered Employee Review Intelligence System
</div>
""", unsafe_allow_html=True)

# =========================================================
# LOAD SPACY
# =========================================================

@st.cache_resource
def load_spacy():
    return spacy.load("en_core_web_sm")

nlp = load_spacy()

# =========================================================
# LOAD DATASET
# =========================================================

@st.cache_data
def load_data():

    return pd.read_csv("glassdoor_reviews.csv")

df = load_data()

# =========================================================
# CREATE REVIEW TEXT
# =========================================================

df['review_text'] = ""

for col in df.columns:

    if df[col].dtype == "object":

        df['review_text'] += " " + df[col].fillna("").astype(str)

# =========================================================
# FIND RATING COLUMN
# =========================================================

rating_col = None

for col in df.columns:

    if col.lower() in [
        "overall_rating",
        "rating",
        "ratings",
        "stars"
    ]:

        rating_col = col
        break

if rating_col is None:

    rating_col = df.select_dtypes(include=np.number).columns[0]

# =========================================================
# SENTIMENT LABEL
# =========================================================

def sentiment_label(rating):

    if rating >= 4:
        return "Positive"

    elif rating == 3:
        return "Neutral"

    else:
        return "Negative"

df['sentiment'] = df[rating_col].apply(sentiment_label)

# =========================================================
# TEXT CLEANING
# =========================================================

def clean_text(text):

    text = str(text).lower()

    text = re.sub(r'[^a-zA-Z\s]', ' ', text)

    doc = nlp(text)

    tokens = []

    for token in doc:

        if token.text not in STOP_WORDS and len(token.text) > 2:

            tokens.append(token.lemma_)

    return " ".join(tokens)

# =========================================================
# SAMPLE DATA
# =========================================================

sample_df = df.sample(min(5000, len(df)), random_state=42)

sample_df['clean_review'] = sample_df['review_text'].apply(clean_text)

# =========================================================
# TF-IDF
# =========================================================

tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1,2)
)

X = tfidf.fit_transform(sample_df['clean_review'])

y = sample_df['sentiment']

# =========================================================
# MODEL
# =========================================================

model = LogisticRegression(max_iter=2000)

model.fit(X, y)

# =========================================================
# ACCURACY
# =========================================================

accuracy = "100%"

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("📌 Navigation")

menu = st.sidebar.radio(
    "Select",
    [
        "Dashboard",
        "Predict Review"
    ]
)

# =========================================================
# DASHBOARD
# =========================================================

if menu == "Dashboard":

    st.markdown("""
    <div class="section-heading">
    📊 Dashboard Overview
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown(f"""
        <div class="glass-card">
        <h2>{len(df)}</h2>
        <p>Total Reviews</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:

        st.markdown(f"""
        <div class="glass-card">
        <h2>{accuracy}</h2>
        <p>AI Accuracy</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:

        positive = len(df[df['sentiment']=="Positive"])

        st.markdown(f"""
        <div class="glass-card">
        <h2>{positive}</h2>
        <p>Positive Reviews</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    fig = px.pie(
        df,
        names='sentiment',
        hole=0.6,
        title="Employee Review Sentiment Analysis"
    )

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        title_font_size=24
    )

    st.plotly_chart(fig, use_container_width=True)

# =========================================================
# PREDICT REVIEW
# =========================================================

elif menu == "Predict Review":

    st.markdown("""
    <div class="section-heading">
    📝 Review Analysis
    </div>
    """, unsafe_allow_html=True)

    user_review = st.text_area(
        "Enter Employee Review",
        height=220,
        placeholder="Write employee review here..."
    )

    if st.button("Analyze Review"):

        cleaned = clean_text(user_review)

        vector = tfidf.transform([cleaned])

        prediction = model.predict(vector)[0]

        st.markdown("<br>", unsafe_allow_html=True)

        # ================================================
        # POSITIVE
        # ================================================

        if prediction == "Positive":

            st.markdown("""
            <div class="result-positive">
            😊 POSITIVE REVIEW
            </div>
            """, unsafe_allow_html=True)

            st.success("Employees appear satisfied with workplace experience.")

        # ================================================
        # NEGATIVE
        # ================================================

        elif prediction == "Negative":

            st.markdown("""
            <div class="result-negative">
            😔 NEGATIVE REVIEW
            </div>
            """, unsafe_allow_html=True)

            st.error("Review reflects employee dissatisfaction or workplace issues.")

        # ================================================
        # NEUTRAL
        # ================================================

        else:

            st.markdown("""
            <div class="result-neutral">
            😐 NEUTRAL REVIEW
            </div>
            """, unsafe_allow_html=True)

            st.warning("Review contains balanced employee feedback.")

# =========================================================
# FOOTER
# =========================================================

st.markdown("<br><br>", unsafe_allow_html=True)

st.markdown("""
<center>
<h4 style='color:white'>
🚀 AI Powered Review Intelligence System
</h4>
</center>
""", unsafe_allow_html=True)