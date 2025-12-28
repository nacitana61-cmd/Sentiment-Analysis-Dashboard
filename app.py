import streamlit as st
import pandas as pd
from transformers import pipeline
import altair as alt
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os
import gc

# 1. Page Config
st.set_page_config(page_title="Brand Sentiment Dashboard", layout="wide")

# 2. Transformer Requirement
@st.cache_resource
def load_sentiment_model():
    # Real DistilBERT Transformer - Emotion version is slightly more memory-stable
    return pipeline(
        "sentiment-analysis", 
        model="bhadresh-savani/distilbert-base-uncased-emotion",
        device=-1
    )

# 3. Sidebar
st.sidebar.title("Navigation")
source_choice = st.sidebar.radio("Go to:", ["Products", "Testimonials", "Reviews"])

def load_data(filename):
    if not os.path.exists(filename): return pd.DataFrame()
    df = pd.read_csv(filename)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['month'] = df['date'].dt.month_name()
    return df

# Main Dashboard
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
    
    with st.spinner("Loading Transformer Model..."):
        sentiment_analyzer = load_sentiment_model()

    df_rev = load_data("reviews.csv")
    if not df_rev.empty:
        months = ["January", "February", "March", "April", "May", "June", 
                  "July", "August", "September", "October", "November", "December"]
        selected_month = st.select_slider("Select a month:", options=months, value="May")
        
        filtered_df = df_rev[df_rev['month'] == selected_month].copy()

        if not filtered_df.empty:
            with st.spinner('Transformer is analyzing sentiment...'):
                results = sentiment_analyzer(filtered_df['review'].tolist())
                
                # Mapping specific Transformer labels to Positive/Negative for the assignment
                processed_sentiments = []
                for res in results:
                    label = res['label'].lower()
                    if label in ['joy', 'love', 'surprise']:
                        processed_sentiments.append('POSITIVE')
                    else:
                        processed_sentiments.append('NEGATIVE')
                
                filtered_df['Sentiment'] = processed_sentiments
                filtered_df['Confidence'] = [res['score'] for res in results]

            # --- Layout: Data table at the TOP ---
            st.subheader(f"Detailed Breakdown for {selected_month}")
            st.dataframe(filtered_df[['date', 'review', 'Sentiment', 'Confidence']], use_container_width=True)
            
            # --- Layout: Charts & Word Bubble ---
            st.divider()
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Sentiment Distribution")
                chart_data = filtered_df.groupby('Sentiment').size().reset_index(name='Count')
                bar_chart = alt.Chart(chart_data).mark_bar().encode(
                    x='Sentiment', y='Count', color='Sentiment'
                ).properties(height=350)
                st.altair_chart(bar_chart, use_container_width=True)

            with col2:
                st.subheader("Word Bubble (Word Cloud)")
                text = " ".join(filtered_df['review'].astype(str).tolist())
                if text.strip():
                    wc = WordCloud(width=500, height=350, background_color='white').generate(text)
                    fig, ax = plt.subplots()
                    ax.imshow(wc, interpolation='bilinear')
                    ax.axis("off")
                    st.pyplot(fig)
            
            gc.collect() # Clean up memory
        else:
            st.warning(f"No reviews found for {selected_month}.")
