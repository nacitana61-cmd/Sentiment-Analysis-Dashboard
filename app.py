import streamlit as st
import pandas as pd
from transformers import pipeline
import altair as alt
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os

# 1. Page Config
st.set_page_config(page_title="Brand Sentiment Dashboard", layout="wide")

# 2. Optimized Sentiment Analysis (Memory Saving)
@st.cache_resource
def load_sentiment_model():
    # 'low_cpu_mem_usage' and 'device=-1' (Force CPU) help stay under 512MB
    return pipeline(
        "sentiment-analysis", 
        model="distilbert-base-uncased-finetuned-sst-2-english",
        model_kwargs={"low_cpu_mem_usage": True},
        device=-1 
    )

sentiment_analyzer = load_sentiment_model()

# 3. Sidebar Navigation
st.sidebar.title("Navigation")
source_choice = st.sidebar.radio("Go to:", ["Products", "Testimonials", "Reviews"])

# Helper function to load and process dates
def load_data(filename):
    if not os.path.exists(filename):
        st.error(f"File {filename} not found!")
        return pd.DataFrame()
    df = pd.read_csv(filename)
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.month_name()
    return df

# Section Behavior
if source_choice == "Products":
    st.title("📦 Scraped Products")
    if os.path.exists("products.csv"):
        st.dataframe(pd.read_csv("products.csv"), use_container_width=True)
    else:
        st.error("products.csv missing!")

elif source_choice == "Testimonials":
    st.title("💬 Scraped Testimonials")
    if os.path.exists("testimonials.csv"):
        st.dataframe(pd.read_csv("testimonials.csv"), use_container_width=True)
    else:
        st.error("testimonials.csv missing!")

elif source_choice == "Reviews":
    st.title("⭐ Reviews Sentiment Analysis")
    df_rev = load_data("reviews.csv")
    
    if not df_rev.empty:
        # Month Selection
        months = ["January", "February", "March", "April", "May", "June", 
                  "July", "August", "September", "October", "November", "December"]
        selected_month = st.select_slider("Select a month in 2023:", options=months, value="May")
        
        # Filter by selected month
        filtered_df = df_rev[df_rev['month'] == selected_month].copy()

        if not filtered_df.empty:
            # Run Sentiment Analysis
            with st.spinner('Analyzing sentiment...'):
                results = sentiment_analyzer(filtered_df['review'].tolist())
                filtered_df['Sentiment'] = [res['label'] for res in results]
                filtered_df['Confidence'] = [res['score'] for res in results]

            # 4. Visualizations
            st.subheader(f"Sentiment Results for {selected_month} 2023")
            
            # Bar Chart
            chart_data = filtered_df.groupby('Sentiment').agg({'Sentiment': 'count', 'Confidence': 'mean'}).rename(columns={'Sentiment': 'Count'}).reset_index()
            bar_chart = alt.Chart(chart_data).mark_bar().encode(
                x='Sentiment',
                y='Count',
                color='Sentiment',
                tooltip=['Sentiment', 'Count', alt.Tooltip('Confidence', format='.2%')]
            ).properties(width=600, height=400)
            st.altair_chart(bar_chart, use_container_width=True)

            # --- BONUS: WORD CLOUD ---
            st.subheader("Word Cloud")
            text = " ".join(filtered_df['review'].tolist())
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
            fig, ax = plt.subplots()
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis("off")
            st.pyplot(fig)
            
            # Raw data
            st.write("Detailed Breakdown:")
            st.dataframe(filtered_df[['date', 'review', 'Sentiment', 'Confidence']], use_container_width=True)
        else:
            st.warning(f"No reviews found for {selected_month}.")
