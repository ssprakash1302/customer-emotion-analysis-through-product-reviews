

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
from datetime import datetime

st.set_page_config(layout="wide", page_title="Emotion Analysis Dashboard")

df = pd.read_csv("Final_Merged_Dataset.csv", parse_dates=['Time'])

df = df[(df["Time"].dt.year >= 2006) & (df["Time"].dt.year <= 2015)]

df["Detected_Emotions"] = df["Detected_Emotions"].dropna().apply(lambda x: x.split(",")[0].strip())

st.sidebar.header("🔍 Filters")

start_date = st.sidebar.date_input("Start Date", datetime(2006, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime(2015, 12, 31))

df_filtered = df[(df["Time"] >= pd.to_datetime(start_date)) & (df["Time"] <= pd.to_datetime(end_date))]

sentiment_filter_list = ["Positive", "Negative", "Neutral"]
selected_sentiments = st.sidebar.multiselect(
    "Filter by Sentiment:", sentiment_filter_list, default=sentiment_filter_list
)

selected_emotions = st.sidebar.multiselect(
    "Filter by Emotion:", df_filtered['Detected_Emotions'].unique().tolist(),
    default=df_filtered['Detected_Emotions'].unique().tolist()
)

adorescore_range = st.sidebar.slider("Filter by Adorescore", -100, 100, (-100, 100))

search_theme = st.sidebar.text_input("Search Theme")
selected_themes = st.sidebar.multiselect("Select Themes:", df_filtered['Main_Topic'].unique())
selected_subtopics = st.sidebar.multiselect("Select Subtopics:", df_filtered['Sub_Topic'].unique())

df_filtered = df_filtered[
    (df_filtered['Sentiment'].isin(selected_sentiments)) &
    (df_filtered['Detected_Emotions'].isin(selected_emotions)) &
    (df_filtered["Adorescore"] >= adorescore_range[0]) &
    (df_filtered["Adorescore"] <= adorescore_range[1])
]

if search_theme:
    df_filtered = df_filtered[df_filtered["Main_Topic"].str.contains(search_theme, case=False, na=False)]
if selected_themes:
    df_filtered = df_filtered[df_filtered["Main_Topic"].isin(selected_themes)]
if selected_subtopics:
    df_filtered = df_filtered[df_filtered["Sub_Topic"].isin(selected_subtopics)]

def analyze_text(input_text):
    api_url = "http://127.0.0.1:8000/analyze"  # Change if deployed elsewhere
    response = requests.post(api_url, json={"text": input_text})
    if response.status_code == 200:
        return response.json()
    return None

st.subheader("🔍 Real-Time Emotion Analysis")
user_input = st.text_area("Enter text to analyze emotions:")

if st.button("Analyze Text"):
    if user_input.strip():
        result = analyze_text(user_input)
        if result:
            st.subheader("🧠 Emotion Analysis Result:")
            
            primary_emotion = result['emotions']['primary']['emotion']
            secondary_emotion = result['emotions']['secondary']['emotion']

            st.write(f"**Primary Emotion:** {primary_emotion}")
            st.write(f"**Secondary Emotion:** {secondary_emotion}")
            st.write(f"**Main Topics Identified:** {', '.join(result['topics']['main'])}")
            st.write(f"**Adorescore:** {result['adorescore']['overall']}")
        else:
            st.error("Failed to analyze text. Please try again.")
    else:
        st.warning("⚠ Please enter some text before analyzing.")

def overall_emotion_plot(df_filtered):
    emotion_counts = df_filtered['Detected_Emotions'].value_counts().reset_index()
    emotion_counts.columns = ['Emotion', 'Count']
    fig = px.pie(emotion_counts, values='Count', names='Emotion', title='Overall Emotion Distribution',
                 hole=0.4, color_discrete_sequence=px.colors.sequential.RdBu)
    return fig

st.subheader("📊 Overall Emotion Distribution")
st.plotly_chart(overall_emotion_plot(df_filtered), use_container_width=True)

adorescore = int(df_filtered["Adorescore"].mean()) if not df_filtered.empty else 0

st.subheader("🌡 *Adorescore Gauge Meter*")
if not df_filtered.empty:
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=adorescore,
        title={'text': "Adorescore", 'font': {'size': 24}},
        gauge={
            'axis': {'range': [-100, 100], 'tickwidth': 1, 'tickcolor': "darkgray"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'steps': [
                {'range': [-100, -50], 'color': 'red'},
                {'range': [-50, 0], 'color': 'orange'},
                {'range': [0, 50], 'color': 'yellowgreen'},
                {'range': [50, 100], 'color': 'green'}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': adorescore
            }
        }
    ))
    fig_gauge.update_layout(height=300, margin=dict(t=20, b=0, l=0, r=0))
    st.plotly_chart(fig_gauge, use_container_width=True)
else:
    st.warning("⚠ *No data available for the selected filters. Try adjusting them.*")

st.markdown("## 💡 Sentiment & Emotion Analysis")

if not df_filtered.empty:
    sentiment_counts = df_filtered['Sentiment'].value_counts().reset_index()
    sentiment_counts.columns = ['Sentiment', 'Count']
    total_entries = sentiment_counts["Count"].sum()

    st.subheader("😊 *Sentiment Distribution*")
    for _, row in sentiment_counts.iterrows():
        percentage = (row['Count'] / total_entries) * 100
        st.write(f"🔹 *{row['Sentiment']}*: {row['Count']} entries ({percentage:.2f}%)")

    top_emotion = df_filtered['Detected_Emotions'].mode()[0]
    top_emotion_count = df_filtered['Detected_Emotions'].value_counts()[top_emotion]
    top_emotion_percentage = (top_emotion_count / total_entries) * 100

    st.subheader("🎭 *Top Emotion*")
    st.write(f"💬 The most frequent emotion is *{top_emotion}* with *{top_emotion_count} mentions* "
             f"({top_emotion_percentage:.2f}% of the filtered data).")

else:
    st.warning("⚠ *No data available for the selected filters. Try adjusting them.*")

theme_counts = df_filtered['Main_Topic'].value_counts().reset_index()
theme_counts.columns = ['Theme', 'Count']

col4, col5 = st.columns([1, 2])
with col4:
    st.metric(label="Adorescore", value=f"{adorescore}", help="Overall sentiment score (-100 to +100)")
with col5:
    st.subheader("Top Themes in Dataset")
    for i, row in theme_counts.iterrows():
        st.write(f"🔹 *{row['Theme']}* - {row['Count']} mentions")

st.subheader("📌 Themes & Snippets")
selected_theme = st.selectbox("Select Theme", theme_counts['Theme'].unique() if not theme_counts.empty else ["None"])
filtered_df = df_filtered[df_filtered['Main_Topic'] == selected_theme]

st.write(f"Showing data for *{selected_theme}* ({len(filtered_df)} entries)")
for _, row in filtered_df.head(15).iterrows():
    st.write(f"💬 {row['Text']}")

st.write("🔎 *Use filters to refine analysis!*")
