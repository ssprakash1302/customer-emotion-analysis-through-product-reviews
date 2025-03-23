# Customer Emotion Analysis System

## 📌 Project Overview
The **Customer Emotion Analysis System** is an AI-powered tool that analyzes customer reviews to extract emotions, sentiment trends, and key topics. It helps businesses gain actionable insights by identifying customer sentiments, root causes of negative feedback, and forecasting future trends. The system integrates **BERTopic, Hugging Face emotion models, AI-powered root cause analysis, and time-series forecasting**, visualized through an interactive **Streamlit dashboard**.

## ✨ Key Features
- **Emotion Classification**: Detects emotions in customer reviews using a **Hugging Face emotion model**.
- **Topic Analysis**: Extracts key themes from reviews using **BERTopic**.
- **AI-Powered Root Cause Analysis**: Identifies root causes of negative feedback using **NLP and clustering techniques**.
- **Adorescore Trends**: Tracks customer satisfaction trends over time.
- **AI-Based Forecasting**: Predicts future sentiment trends using **time-series models**.
- **Interactive Dashboard**: Visualizes insights using **Streamlit**.
- **Search & Filtering**: Allows searching and filtering of reviews based on sentiment and topics.

## 🛠️ Tech Stack
- **Python** (Data Processing, Model Training, API Development)
- **Streamlit** (Dashboard Visualization)
- **Hugging Face Transformers** (Emotion Detection)
- **BERTopic** (Topic Modeling)
- **Scikit-learn, Pandas, Matplotlib, Seaborn** (Data Processing & Visualization)
- **Flask** (API for model inference)
- **Docker** (Containerization for deployment)

## 🚀 Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/customer-emotion-analysis.git
   cd customer-emotion-analysis
   ```
2. Create a virtual environment:
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
4. Download and place the pre-trained models (`bertopic_model`, `emotion_model`, `processed_data.pkl`) in the `models/` directory.
5. Run the Streamlit dashboard:
   ```sh
   streamlit run app.py
   ```

## 📊 Usage
- **Upload customer reviews** (CSV/Text format).
- **View overall emotion distribution** with visualizations.
- **Analyze root causes of negative reviews**.
- **Explore trending topics** using BERTopic.
- **Forecast future customer sentiment trends**.
- **Search and filter reviews** for targeted analysis.

## 📌 API Endpoints (Flask)
- `/predict-emotion` → Predicts emotion from text.
- `/get-topics` → Extracts topics from reviews.
- `/root-cause-analysis` → Provides AI-driven root cause insights.


## 🔹 Note on Model Files (PKL)

Due to file size limitations, the model files (`.pkl`) are not included in this repository.  
You can download them from the following Google Drive link:

📥 [Download Model Files]~(https://drive.google.com/file/d/18L_aecEZB22jGqNrn2ZRvTeotTBIagmX/view?usp=sharing)

After downloading, place the `.pkl` files in the appropriate directory before running the project.

## 🏆 Unique Aspects
✅ **AI-Powered Root Cause Analysis** for negative feedback.  
✅ **Time-Series Forecasting** to predict sentiment trends.  
✅ **Advanced Topic Modeling** with BERTopic.  
✅ **User-Friendly Dashboard** for insights visualization.  

## 🤖 Future Enhancements
- Improve model accuracy with fine-tuning.
- Add real-time review fetching from social media & e-commerce sites.
- Deploy the system using **Docker & cloud services**.

## 📜 License
This project is **open-source** and available under the [MIT License](LICENSE).

---
Made with ❤️ by **Shanmuga Prakash S** 🚀

