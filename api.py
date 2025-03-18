
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import re
import pickle

app = FastAPI()

try:
    with open('sia.pkl', 'rb') as f:
        sia = pickle.load(f)
    with open('emotion_pipeline.pkl', 'rb') as f:
        emotion_pipeline = pickle.load(f)
    with open('topic_classifier.pkl', 'rb') as f:
        topic_classifier = pickle.load(f)
    with open('embedding_model.pkl', 'rb') as f:
        embedding_model = pickle.load(f)
    with open('bertopic_model.pkl', 'rb') as f:
        bertopic_model = pickle.load(f)
    print("✅ All models loaded successfully!")
except Exception as e:
    print(f"Model loading failed: {str(e)}")

try:
    df = pd.read_csv("Final_Main_Subtopics.csv")
    processed_texts = df['Processed_Text'].astype(str).tolist()
    bertopic_model.fit(processed_texts)
    print("BERTopic model refitted successfully!")
except Exception as e:
    print(f"BERTopic refitting failed: {str(e)}")

possible_topics = [
    "Product Quality", "Durability", "Defective Item", "Wrong Item Sent", "Packaging Issue",
    "Late Delivery", "Fast Delivery", "Damaged During Shipping",
    "Customer Support Response", "Refund & Returns", "Replacement Process",
    "Expensive Product", "Discounts & Deals", "Value for Money",
    "Website Usability", "Trust in Brand", "Fake or Counterfeit Product", "Delivery", "Quality", "Clothes"
]

class ReviewInput(BaseModel):
    text: str

@app.post("/analyze")
def analyze_review(review: ReviewInput):
    try:
        review_text = review.text.lower()
        review_text = re.sub(r'\W', ' ', review_text)
        review_text = re.sub(r'\s+', ' ', review_text).strip()

        sentiment_score = sia.polarity_scores(review_text)['compound']

        
        emotions_data = emotion_pipeline(review_text[:512])

        if not isinstance(emotions_data, list) or len(emotions_data) == 0 or not isinstance(emotions_data[0], list):
            raise ValueError("Emotion analysis returned unexpected format.")

        emotions_data = emotions_data[0]  # Extract first list inside the list of lists

        primary_emotion = emotions_data[0] if len(emotions_data) > 0 else {"label": "Unknown", "score": 0}
        secondary_emotion = emotions_data[1] if len(emotions_data) > 1 else {"label": "Unknown", "score": 0}

        main_topics = topic_classifier(review_text, possible_topics, multi_label=True)
        selected_main_topics = [
            topic for topic, score in zip(main_topics['labels'], main_topics['scores']) if score > 0.5
        ]
        if not selected_main_topics:
            selected_main_topics = ["General Feedback"]

        topics, _ = bertopic_model.transform([review_text])
        sub_topic_labels = [
            ", ".join([word for word, _ in bertopic_model.get_topic(topic_id)])
            if topic_id != -1 else "Unknown"
            for topic_id in topics
        ]
        subtopics = {main_topic: [sub_topic_labels[0]] for main_topic in selected_main_topics}

        emotion_weights = {
            "joy": 1.0, "trust": 0.8, "anticipation": 0.7, "surprise": 0.6,
            "sadness": -1.0, "fear": -0.8, "anger": -0.9, "disgust": -0.7,
            "ecstasy": 1.2, "admiration": 1.1, "terror": -1.2, "loathing": -1.1
        }
        adorescore_overall = sum(
            emotion_weights.get(emotion['label'].lower(), 0) * emotion['score']
            for emotion in emotions_data
        )
        adorescore_overall = round(adorescore_overall * 100)

        adorescore_breakdown = {main_topic: round(sentiment_score * 100) for main_topic in selected_main_topics}

        return {
            "emotions": {
                "primary": {
                    "emotion": primary_emotion.get('label', 'Unknown'),
                    "activation": "High" if primary_emotion.get('score', 0) > 0.75 else "Medium" if primary_emotion.get('score', 0) > 0.5 else "Low",
                    "intensity": round(primary_emotion.get('score', 0), 2)
                },
                "secondary": {
                    "emotion": secondary_emotion.get('label', 'Unknown'),
                    "activation": "High" if secondary_emotion.get('score', 0) > 0.75 else "Medium" if secondary_emotion.get('score', 0) > 0.5 else "Low",
                    "intensity": round(secondary_emotion.get('score', 0), 2)
                }
            },
            "topics": {
                "main": selected_main_topics,
                "subtopics": subtopics
            },
            "adorescore": {
                "overall": adorescore_overall,
                "breakdown": adorescore_breakdown
            }
        }

    except Exception as e:
        print(f"Error in analyze_review(): {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
