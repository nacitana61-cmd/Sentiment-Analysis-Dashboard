import streamlit as st
import pandas as pd
from transformers import pipeline
import altair as alt
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os

# 1. Page Configuration
st.set_page_config(page_title="Brand Sentiment Dashboard", layout="wide")

# 2. Optimized Sentiment Analysis Function
# Using a fine-tuned TinyBERT model to stay under 512MB RAM
@st.cache_resource
def load_sentiment_model():
    return pipeline(
        "sentiment-analysis", 
        model="cross-encoder/ms-marco-TinyBERT-L-2-v2", 
        device=-1 
    )

# 3. Sidebar Navigation
st.sidebar.title("Navigation")
source_choice = st.sidebar.radio("Go to:", ["Products", "Testimonials", "Reviews"])

# Helper function to load data
def load_data(filename):
    if not os.path.exists(filename):
        return pd.DataFrame()
    df = pd.read_csv(filename)
    # Ensure date column is properly formatted
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['month'] = df['date'].dt.month_name()
    return df

# --- MAIN DASHBOARD LOGIC ---

if source_choice == "Products":
    st.title("📦 Scraped Products")
    if os.path.exists("products.csv"):
        st.dataframe(pd.read_csv("products.csv"), use_container_width=True)
    else:
        st.error("File 'products.csv' not found. Please upload it to your GitHub.")

elif source_choice == "Testimonials":
    st.title("💬 Scraped Testimonials")
    if os.path.exists("testimonials.csv"):
        st.dataframe(pd.read_csv("testimonials.csv"), use_container_width=True)
    else:
        st.error("File 'testimonials.csv' not found. Please upload it to your GitHub.")

elif source_choice == "Reviews":
    st.title("⭐ Reviews Sentiment Analysis")
    
    # Lazy Load Model: Only loads when user visits this tab to save RAM
    with st.spinner("Initializing AI Model..."):
        sentiment_analyzer = load_sentiment_model()

    df_rev = load_data("reviews.csv")
    
    if not df_rev.empty:
        # Month Filter Slider
        months = ["January", "February", "March", "April", "May", "June", 
                  "July", "August", "September", "October", "November", "December"]
        selected_month = st.select_slider("Select a month to analyze:", options=months, value="May")
        
        # Filter Data
        filtered_df = df_rev[df_rev['month'] == selected_month].copy()

        if not filtered_df.empty:
            with st.spinner(f'AI is analyzing {len(filtered_df)} reviews for {selected_month}...'):
                # Run the prediction
                results = sentiment_analyzer(filtered_df['review'].tolist())
                
                # Map scores to labels for the report
                sentiments = []
                confidences = []
                for res in results:
                    # This specific model uses raw scores; > 0 is generally positive
                    if res['score'] > 0:
                        sentiments.append('POSITIVE')
                    else:
                        sentiments.append('NEGATIVE')
                    confidences.append(abs(res['score']))
                
                filtered_df['Sentiment'] = sentiments
                filtered_df['Confidence'] = confidences

            # 4. DATA VISUALIZATION
            st.subheader(f"Sentiment Results: {selected_month} 2023")
            
            # Sentiment Bar Chart
            chart_data = filtered_df.groupby('Sentiment').size().reset_index(name='Count')
            bar_chart = alt.Chart(chart_data).mark_bar().encode(
                x='Sentiment',
                y='Count',
                color='Sentiment'
            ).properties(height=350)
            st.altair_chart(bar_chart, use_container_width=True)

            # Bonus: Word Cloud
            st.subheader("Key Topics (Word Cloud)")
            text = " ".join(filtered_df['review'].astype(str).tolist())
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
            fig, ax = plt.subplots()
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis("off")
            st.pyplot(fig)
            
            # Detailed Data Table
            st.write("Detailed Breakdown:")
            st.dataframe(filtered_df[['date', 'review', 'Sentiment', 'Confidence']], use_container_width=True)
        else:
            st.warning(f"No reviews found in the dataset for {selected_month}.")
    else:
        st.error("reviews.csv is missing or empty.")
