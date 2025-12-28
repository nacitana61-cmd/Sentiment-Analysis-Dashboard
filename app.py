import streamlit as st
import pandas as pd
from transformers import pipeline
import altair as alt

st.set_page_config(page_title="Brand Sentiment Dashboard", layout="wide")

# 3. Sentiment Analysis (Hugging Face)
@st.cache_resource
def load_sentiment_model():
    # Added low_cpu_mem_usage and device=-1 to fit your code into 512MB
    return pipeline(
        "sentiment-analysis", 
        model="distilbert-base-uncased-finetuned-sst-2-english",
        low_cpu_mem_usage=True,
        device=-1
    )

sentiment_analyzer = load_sentiment_model()

# 2. Sidebar Navigation
st.sidebar.title("Navigation")
source_choice = st.sidebar.radio("Go to:", ["Products", "Testimonials", "Reviews"])

# Helper function to load and process dates
def load_data(filename):
    df = pd.read_csv(filename)
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.month_name()
    return df

# Section Behavior
if source_choice == "Products":
    st.title("📦 Scraped Products")
    df_prod = pd.read_csv("products.csv")
    st.dataframe(df_prod, use_container_width=True)

elif source_choice == "Testimonials":
    st.title("💬 Scraped Testimonials")
    df_test = pd.read_csv("testimonials.csv")
    st.dataframe(df_test, use_container_width=True)

elif source_choice == "Reviews":
    st.title("⭐ Reviews Sentiment Analysis")
    df_rev = load_data("reviews.csv")
    
    # Month Selection (Core Feature)
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

        # 4. Visualization
        st.subheader(f"Sentiment Results for {selected_month} 2023")
        
        # Aggregate data for Bar Chart
        chart_data = filtered_df.groupby('Sentiment').agg({'Sentiment': 'count', 'Confidence': 'mean'}).rename(columns={'Sentiment': 'Count'}).reset_index()

        # Advanced Chart with Tooltip for Confidence Score
        bar_chart = alt.Chart(chart_data).mark_bar().encode(
            x='Sentiment',
            y='Count',
            color='Sentiment',
            tooltip=['Sentiment', 'Count', alt.Tooltip('Confidence', format='.2%')]
        ).properties(width=600, height=400)

        st.altair_chart(bar_chart, use_container_width=True)
        
        # Display raw data table
        st.write("Detailed Breakdown:")
        st.dataframe(filtered_df[['date', 'review', 'Sentiment', 'Confidence']], use_container_width=True)
    else:
        st.warning(f"No reviews found for {selected_month}.")
