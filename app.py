import streamlit as st
import pandas as pd
from transformers import pipeline
import altair as alt
import os
import gc

st.set_page_config(page_title="Brand Sentiment Dashboard", layout="wide")

# REQUIRED: Hugging Face Transformer Pipeline
@st.cache_resource
def load_sentiment_model():
    # Using TinyBERT: A real Transformer that fits in 512MB
    return pipeline(
        "sentiment-analysis", 
        model="cross-encoder/ms-marco-TinyBERT-L-2-v2",
        device=-1
    )

# Sidebar
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
    
    # Load model only when needed
    with st.spinner("Loading Transformer Model..."):
        sentiment_analyzer = load_sentiment_model()

    df_rev = load_data("reviews.csv")
    months = ["January", "February", "March", "April", "May", "June", 
              "July", "August", "September", "October", "November", "December"]
    selected_month = st.select_slider("Select a month:", options=months, value="May")
    
    filtered_df = df_rev[df_rev['month'] == selected_month].copy()

    if not filtered_df.empty:
        with st.spinner('Transformer is analyzing sentiment...'):
            results = sentiment_analyzer(filtered_df['review'].tolist())
            
            # Map Transformer scores to Positive/Negative labels
            labels = []
            for res in results:
                # ms-marco-TinyBERT uses a score: > 0 is Positive
                if res['score'] > 0:
                    labels.append("POSITIVE")
                else:
                    labels.append("NEGATIVE")
            
            filtered_df['Sentiment'] = labels
            filtered_df['Confidence'] = [abs(res['score']) for res in results]

        st.subheader(f"Sentiment Results (Transformer Model)")
        
        # Chart
        chart_data = filtered_df.groupby('Sentiment').size().reset_index(name='Count')
        bar_chart = alt.Chart(chart_data).mark_bar().encode(
            x='Sentiment', y='Count', color='Sentiment'
        ).properties(width=600, height=400)

        st.altair_chart(bar_chart, use_container_width=True)
        
        # Table
        st.write("Detailed Breakdown:")
        st.dataframe(filtered_df[['date', 'review', 'Sentiment', 'Confidence']], use_container_width=True)
        
        # Force memory cleanup
        gc.collect()
    else:
        st.warning(f"No reviews found for {selected_month}.")
        # FINAL CLEANUP
        gc.collect() 
    else:
        st.warning("No data.")

