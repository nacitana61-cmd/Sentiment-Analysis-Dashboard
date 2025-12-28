import streamlit as st
import pandas as pd
from transformers import pipeline
import altair as alt
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Brand Sentiment Dashboard",
    layout="wide"
)

# --------------------------------------------------
# CONSTANTS (SAFE LIMITS FOR RENDER)
# --------------------------------------------------
MAX_REVIEWS = 200          # reduce if memory issues persist
SENTIMENT_BATCH_SIZE = 16 # small batches = low memory

# --------------------------------------------------
# LOAD SENTIMENT MODEL (CACHED, CPU ONLY)
# --------------------------------------------------
@st.cache_resource
def load_sentiment_model():
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=-1  # force CPU (important for Render)
    )

sentiment_analyzer = load_sentiment_model()

# --------------------------------------------------
# DATA LOADING (CACHED)
# --------------------------------------------------
@st.cache_data
def load_reviews():
    df = pd.read_csv("reviews.csv")
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df["month"] = pd.to_datetime(df["date"]).dt.month_name()
    return df

@st.cache_data
def load_products():
    return pd.read_csv("products.csv")

@st.cache_data
def load_testimonials():
    return pd.read_csv("testimonials.csv")

# --------------------------------------------------
# SENTIMENT BATCHING FUNCTION
# --------------------------------------------------
def run_sentiment_in_batches(texts, batch_size=SENTIMENT_BATCH_SIZE):
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        results.extend(sentiment_analyzer(batch))
    return results

# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------
st.sidebar.title("Navigation")
source_choice = st.sidebar.radio(
    "Go to:",
    ["Products", "Testimonials", "Reviews"]
)

# --------------------------------------------------
# PRODUCTS VIEW
# --------------------------------------------------
if source_choice == "Products":
    st.title("📦 Scraped Products")
    df_prod = load_products()
    st.dataframe(df_prod, use_container_width=True)

# --------------------------------------------------
# TESTIMONIALS VIEW
# --------------------------------------------------
elif source_choice == "Testimonials":
    st.title("💬 Scraped Testimonials")
    df_test = load_testimonials()
    st.dataframe(df_test, use_container_width=True)

# --------------------------------------------------
# REVIEWS + SENTIMENT VIEW
# --------------------------------------------------
elif source_choice == "Reviews":
    st.title("⭐ Reviews Sentiment Analysis")

    df_rev = load_reviews()

    # Month selector
    months = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ]

    selected_month = st.select_slider(
        "Select a month in 2023:",
        options=months,
        value="May"
    )

    filtered_df = df_rev[df_rev["month"] == selected_month].copy()
    filtered_df = filtered_df.head(MAX_REVIEWS)  # MEMORY LIMIT

    if filtered_df.empty:
        st.warning(f"No reviews found for {selected_month}.")
        st.stop()

    # --------------------------------------------------
    # RUN SENTIMENT (BUTTON CONTROLLED)
    # --------------------------------------------------
    if st.button("Run Sentiment Analysis"):
        with st.spinner("Analyzing sentiment..."):
            results = run_sentiment_in_batches(
                filtered_df["review"].tolist()
            )

            filtered_df["Sentiment"] = [r["label"] for r in results]
            filtered_df["Confidence"] = [r["score"] for r in results]

        # --------------------------------------------------
        # BAR CHART
        # --------------------------------------------------
        st.subheader(f"Sentiment Results for {selected_month} 2023")

        chart_data = (
            filtered_df
            .groupby("Sentiment")
            .agg(
                Count=("Sentiment", "count"),
                Confidence=("Confidence", "mean")
            )
            .reset_index()
        )

        bar_chart = alt.Chart(chart_data).mark_bar().encode(
            x="Sentiment:N",
            y="Count:Q",
            color="Sentiment:N",
            tooltip=[
                "Sentiment",
                "Count",
                alt.Tooltip("Confidence", format=".2%")
            ]
        ).properties(
            width=600,
            height=400
        )

        st.altair_chart(bar_chart, use_container_width=True)

        # --------------------------------------------------
        # WORD CLOUD
        # --------------------------------------------------
        st.subheader("🧠 Review Word Cloud")

        text_blob = " ".join(filtered_df["review"].tolist())

        wordcloud = WordCloud(
            width=600,
            height=300,
            background_color="white",
            max_words=100
        ).generate(text_blob)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.imshow(wordcloud)
        ax.axis("off")
        st.pyplot(fig)

        # --------------------------------------------------
        # RAW DATA TABLE
        # --------------------------------------------------
        st.subheader("📄 Detailed Breakdown")
        st.dataframe(
            filtered_df[["date", "review", "Sentiment", "Confidence"]],
            use_container_width=True
        )
