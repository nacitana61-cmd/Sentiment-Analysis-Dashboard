import streamlit as st
import pandas as pd
from transformers import pipeline
import os
import gc  # Garbage Collector
import altair as alt

st.set_page_config(page_title="Sentiment Dashboard", layout="wide")

# 1. THE ENGINE: Optimized for RAM
@st.cache_resource
def load_sentiment_model():
    # We use the 'good' model but tell it to be as lean as possible
    return pipeline(
        "sentiment-analysis", 
        model="distilbert-base-uncased-finetuned-sst-2-english",
        model_kwargs={"low_cpu_mem_usage": True}, # Essential for 512MB
        device=-1
    )

# 2. DATA LOADING: Optimized to not keep double copies
def load_data(filename):
    df = pd.read_csv(filename)
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.month_name()
    return df

# Sidebar
st.sidebar.title("Navigation")
source_choice = st.sidebar.radio("Go to:", ["Products", "Testimonials", "Reviews"])

if source_choice == "Products":
    st.title("📦 Products")
    st.dataframe(pd.read_csv("products.csv"), use_container_width=True)

elif source_choice == "Testimonials":
    st.title("💬 Testimonials")
    st.dataframe(pd.read_csv("testimonials.csv"), use_container_width=True)

elif source_choice == "Reviews":
    st.title("⭐ High-Accuracy Analysis")
    
    # Load model only when the user visits this tab
    sentiment_analyzer = load_sentiment_model()
    
    df_rev = load_data("reviews.csv")
    months = ["January", "February", "March", "April", "May", "June", 
              "July", "August", "September", "October", "November", "December"]
    selected_month = st.select_slider("Select Month:", options=months, value="May")
    
    filtered_df = df_rev[df_rev['month'] == selected_month].copy()
    
    # Immediately delete the big dataframe to free space
    del df_rev 
    gc.collect()

    if not filtered_df.empty:
        with st.spinner('AI is thinking...'):
            # Analyze reviews
            texts = filtered_df['review'].tolist()
            results = sentiment_analyzer(texts)
            
            filtered_df['Sentiment'] = [res['label'] for res in results]
            filtered_df['Confidence'] = [res['score'] for res in results]

        # DISPLAY RESULTS
        st.subheader(f"Results for {selected_month}")
        
        # Simple Chart (Altair is lighter than Matplotlib)
        chart_data = filtered_df.groupby('Sentiment').size().reset_index(name='Count')
        st.altair_chart(alt.Chart(chart_data).mark_bar().encode(
            x='Sentiment', y='Count', color='Sentiment'
        ).properties(height=300), use_container_width=True)

        st.dataframe(filtered_df[['date', 'review', 'Sentiment', 'Confidence']], use_container_width=True)
        
        # FINAL CLEANUP
        gc.collect() 
    else:
        st.warning("No data.")
