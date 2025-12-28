import streamlit as st
import pandas as pd
from transformers import pipeline
import altair as alt
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os
import gc

st.set_page_config(page_title="Brand Sentiment Dashboard", layout="wide")

# 1. Optimized Transformer Loader
@st.cache_resource
def load_sentiment_model():
    # This is the EXACT model your professor requested
    # We use low_cpu_mem_usage to prevent the 512MB crash
    return pipeline(
        "sentiment-analysis", 
        model="distilbert-base-uncased-finetuned-sst-2-english",
        model_kwargs={"low_cpu_mem_usage": True},
        device=-1
    )

# 2. Navigation
st.sidebar.title("Navigation")
source_choice = st.sidebar.radio("Go to:", ["Products", "Testimonials", "Reviews"])

def load_data(filename):
    if not os.path.exists(filename): return pd.DataFrame()
    df = pd.read_csv(filename)
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.month_name()
    return df

if source_choice == "Products":
    st.title("📦 Scraped Products")
    if os.path.exists("products.csv"):
        st.dataframe(pd.read_csv("products.csv"), use_container_width=True)

elif source_choice == "Testimonials":
    st.title("💬 Scraped Testimonials")
    if os.path.exists("testimonials.csv"):
        st.dataframe(pd.read_csv("testimonials.csv"), use_container_width=True)

elif source_choice == "Reviews":
    st.title("⭐ Reviews Sentiment Analysis")
    
    with st.spinner("Loading Transformer..."):
        sentiment_analyzer = load_sentiment_model()

    df_rev = load_data("reviews.csv")
    if not df_rev.empty:
        months = ["January", "February", "March", "April", "May", "June", 
                  "July", "August", "September", "October", "November", "December"]
        selected_month = st.select_slider("Select a month:", options=months, value="May")
        
        filtered_df = df_rev[df_rev['month'] == selected_month].copy()

        if not filtered_df.empty:
            with st.spinner('Analyzing Sentiment...'):
                # This will now correctly show POSITIVE and NEGATIVE
                results = sentiment_analyzer(filtered_df['review'].tolist())
                filtered_df['Sentiment'] = [res['label'] for res in results]
                filtered_df['Confidence'] = [res['score'] for res in results]

            # --- Visualization Section ---
            st.subheader(f"Analysis for {selected_month}")
            
            # 1. Chart
            chart_data = filtered_df.groupby('Sentiment').size().reset_index(name='Count')
            bar_chart = alt.Chart(chart_data).mark_bar().encode(
                x='Sentiment', y='Count', color='Sentiment'
            ).properties(height=300)
            st.altair_chart(bar_chart, use_container_width=True)

            # 2. Word Cloud (The "Word Bubble")
            st.subheader("Word Cloud")
            text = " ".join(filtered_df['review'].astype(str).tolist())
            if text.strip():
                wc = WordCloud(width=800, height=400, background_color='white').generate(text)
                fig, ax = plt.subplots()
                ax.imshow(wc, interpolation='bilinear')
                ax.axis("off")
                st.pyplot(fig)

            # 3. Table
            st.write("Detailed Breakdown:")
            st.dataframe(filtered_df[['date', 'review', 'Sentiment', 'Confidence']], use_container_width=True)
            
            gc.collect()
        else:
            st.warning(f"No reviews found for {selected_month}.")
