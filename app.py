import streamlit as st
import pandas as pd
from transformers import pipeline
import plotly.express as px
from datetime import datetime
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import random

# ---------------------------
# LOAD DATA
# ---------------------------
@st.cache_data
def load_data():
    products = pd.read_csv("products.csv")
    testimonials = pd.read_csv("testimonials.csv")
    reviews = pd.read_csv("reviews.csv")

    # Ensure reviews have dates
    if "date" not in reviews.columns:
        reviews["date"] = [f"2023-{(i%12)+1:02d}-{random.randint(1,28):02d}" for i in range(len(reviews))]

    reviews["date"] = pd.to_datetime(reviews["date"])
    reviews["month"] = reviews["date"].dt.month
    return products, testimonials, reviews

products_df, testimonials_df, reviews_df = load_data()

# ---------------------------
# SIDEBAR NAVIGATION
# ---------------------------
st.sidebar.title("Navigation")
section = st.sidebar.radio("Choose Section", ["Products", "Testimonials", "Reviews"])

# ---------------------------
# PRODUCTS SECTION
# ---------------------------
if section == "Products":
    st.title("Products")
    st.dataframe(products_df)

# ---------------------------
# TESTIMONIALS SECTION
# ---------------------------
elif section == "Testimonials":
    st.title("Testimonials")
    st.dataframe(testimonials_df)

# ---------------------------
# REVIEWS SECTION
# ---------------------------
elif section == "Reviews":
    st.title("Reviews (Sentiment Analysis)")

    # ---------------------------
    # MONTH SELECTION
    # ---------------------------
    month_options = list(range(1, 13))
    month_name_map = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                      7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
    selected_month = st.select_slider(
        "Select Month (2023)",
        options=month_options,
        format_func=lambda x: month_name_map[x]
    )

    # FILTER REVIEWS
    filtered_reviews = reviews_df[reviews_df["month"] == selected_month]
    st.write(f"Total Reviews in {month_name_map[selected_month]} 2023: {len(filtered_reviews)}")
    st.dataframe(filtered_reviews)

    if len(filtered_reviews) > 0:
        # ---------------------------
        # SENTIMENT ANALYSIS
        # ---------------------------
        st.write("Performing Sentiment Analysis...")
        sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

        results = sentiment_pipeline(list(filtered_reviews["review"]))

        filtered_reviews["sentiment"] = [r["label"] for r in results]
        filtered_reviews["confidence"] = [r["score"] for r in results]

        # ---------------------------
        # VISUALIZATION (BAR CHART)
        # ---------------------------
        sentiment_summary = filtered_reviews.groupby("sentiment").agg(
            count=("review","count"),
            avg_confidence=("confidence","mean")
        ).reset_index()

        fig = px.bar(
            sentiment_summary,
            x="sentiment",
            y="count",
            color="sentiment",
            text="count",
            hover_data={"avg_confidence":True},
            title=f"Sentiment Count in {month_name_map[selected_month]} 2023"
        )
        st.plotly_chart(fig)

        # ---------------------------
        # WORD CLOUD
        # ---------------------------
        st.subheader("Word Cloud of Reviews")
        text = " ".join(filtered_reviews["review"])
        wc = WordCloud(width=800, height=400, background_color="white").generate(text)

        plt.figure(figsize=(10,5))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        st.pyplot(plt)
