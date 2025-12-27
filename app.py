import streamlit as st
import pandas as pd
from transformers import pipeline
import altair as alt
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os

# 1. Page Config
st.set_page_config(page_title="Brand Sentiment Dashboard", layout="wide")

# 2. Optimized Sentiment Analysis (Tiny Model for Memory)
@st.cache_resource
def load_sentiment_model():
    # Bert-Tiny is much smaller than DistilBERT and fits the 512MB limit easily
    return pipeline(
        "sentiment-analysis", 
        model="prajjwal1/Bert-Tiny", 
        device=-1 
    )

# 3. Sidebar Navigation
st.sidebar.title("Navigation")
source_choice = st.sidebar.radio("Go to:", ["Products", "Testimonials", "Reviews"])

def load_data(filename):
    if not os.path.exists(filename):
        return pd.DataFrame()
    df = pd.read_csv(filename)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['month'] = df['date'].dt.month_name()
    return df

# Main Logic
if source_choice == "Products":
    st.title("📦 Scraped Products")
    if os.path.exists("products.csv"):
        st.dataframe(pd.read_csv("products.csv"), use_container_width=True)
    else:
        st.error("products.csv not found in GitHub.")

elif source_choice == "Testimonials":
    st.title("💬 Scraped Testimonials")
    if os.path.exists("testimonials.csv"):
        st.dataframe(pd.read_csv("testimonials.csv"), use_container_width=True)
    else:
        st.error("testimonials.csv not found in GitHub.")

elif source_choice == "Reviews":
    st.title("⭐ Reviews Sentiment Analysis")
    
    # Lazy Loading: The AI only takes memory when this tab is clicked
    with st.spinner("Initializing AI..."):
        sentiment_analyzer = load_sentiment_model()

    df_rev = load_data("reviews.csv")
    if not df_rev.empty:
        months = ["January", "February", "March", "April", "May", "June", 
                  "July", "August", "September", "October", "November", "December"]
        selected_month = st.select_slider("Select Month:", options=months, value="May")
        
        filtered_df = df_rev[df_rev['month'] == selected_month].copy()

        if not filtered_df.empty:
            with st.spinner('Analyzing sentiment...'):
                results = sentiment_analyzer(filtered_df['review'].tolist())
                
                # Cleaning up labels for the report
                sentiments = []
                for res in results:
                    # Logic to ensure labels are user-friendly
                    if res['label'] in ['LABEL_1', 'POSITIVE']:
                        sentiments.append('POSITIVE')
                    else:
                        sentiments.append('NEGATIVE')
                
                filtered_df['Sentiment'] = sentiments
                filtered_df['Confidence'] = [res['score'] for res in results]

            st.subheader(f"Sentiment Analysis: {selected_month} 2023")
            
            # Chart
            chart_data = filtered_df.groupby('Sentiment').size().reset_index(name='Count')
            bar_chart = alt.Chart(chart_data).mark_bar().encode(
                x='Sentiment', y='Count', color='Sentiment'
            ).properties(height=300)
            st.altair_chart(bar_chart, use_container_width=True)

            # Bonus Word Cloud
            st.subheader("Word Cloud")
            text = " ".join(filtered_df['review'].astype(str).tolist())
            wc = WordCloud(width=800, height=400, background_color='white').generate(text)
            fig, ax = plt.subplots()
            ax.imshow(wc, interpolation='bilinear')
            ax.axis("off")
            st.pyplot(fig)
            
            st.dataframe(filtered_df[['date', 'review', 'Sentiment', 'Confidence']], use_container_width=True)
        else:
            st.warning(f"No reviews found for {selected_month}.")
