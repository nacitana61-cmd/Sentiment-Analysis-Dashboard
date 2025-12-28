import streamlit as st
import pandas as pd
from transformers import pipeline
import altair as alt
import os
import gc # Garbage collection to clear memory

st.set_page_config(page_title="Brand Sentiment Dashboard", layout="wide")

# 1. Improved Memory-Saving Model Loader
@st.cache_resource
def load_sentiment_model():
    # Force the model to load in the leanest way possible
    return pipeline(
        "sentiment-analysis", 
        model="distilbert-base-uncased-finetuned-sst-2-english",
        model_kwargs={"low_cpu_mem_usage": True},
        device=-1 # Ensure CPU only
    )

# 2. Sidebar Navigation
st.sidebar.title("Navigation")
source_choice = st.sidebar.radio("Go to:", ["Products", "Testimonials", "Reviews"])

def load_data(filename):
    df = pd.read_csv(filename)
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.month_name()
    return df

if source_choice == "Products":
    st.title("📦 Scraped Products")
    st.dataframe(pd.read_csv("products.csv"), use_container_width=True)

elif source_choice == "Testimonials":
    st.title("💬 Scraped Testimonials")
    st.dataframe(pd.read_csv("testimonials.csv"), use_container_width=True)

elif source_choice == "Reviews":
    st.title("⭐ Reviews Sentiment Analysis")
    
    # Load the model ONLY when we need it
    sentiment_analyzer = load_sentiment_model()
    
    df_rev = load_data("reviews.csv")
    months = ["January", "February", "March", "April", "May", "June", 
              "July", "August", "September", "October", "November", "December"]
    selected_month = st.select_slider("Select a month:", options=months, value="May")
    
    filtered_df = df_rev[df_rev['month'] == selected_month].copy()

    if not filtered_df.empty:
        with st.spinner('Analyzing sentiment...'):
            # Process reviews
            results = sentiment_analyzer(filtered_df['review'].tolist())
            filtered_df['Sentiment'] = [res['label'] for res in results]
            filtered_df['Confidence'] = [res['score'] for res in results]

        st.subheader(f"Results for {selected_month}")
        
        # Chart Logic
        chart_data = filtered_df.groupby('Sentiment').agg({'Sentiment': 'count', 'Confidence': 'mean'}).rename(columns={'Sentiment': 'Count'}).reset_index()
        bar_chart = alt.Chart(chart_data).mark_bar().encode(
            x='Sentiment', y='Count', color='Sentiment',
            tooltip=['Sentiment', 'Count', alt.Tooltip('Confidence', format='.2%')]
        ).properties(width=600, height=400)

        st.altair_chart(bar_chart, use_container_width=True)
        st.write("Detailed Breakdown:")
        st.dataframe(filtered_df[['date', 'review', 'Sentiment', 'Confidence']], use_container_width=True)
        
        # Manually clear memory after processing
        gc.collect() 
    else:
        st.warning(f"No reviews found for {selected_month}.")
