# SentimentSnap - AI Sentiment Analysis

## 📌 Overview
**SentimentSnap** is an AI-powered sentiment analysis application built with **Streamlit**. It can analyze text sentiment, translate non-English text, and classify the sentiment as **Positive, Negative, or Neutral**. The app leverages **Natural Language Processing (NLP)** and **Machine Learning** for high-accuracy sentiment detection.

## 🚀 Features
- ✅ **Text Sentiment Analysis** (Positive, Negative, Neutral)
- ✅ **Multilingual Support** with automatic translation (via Google Translate)
- ✅ **Text Preprocessing** (Stopword removal, Lemmatization, Cleaning)
- ✅ **Confidence Score & Probability Distribution**
- ✅ **Simple & Interactive UI with Streamlit**

## 📂 Project Structure
```
SentimentSnap/
│-- sentiment_model.joblib       # Pre-trained ML model
│-- tfidf_vectorizer.joblib      # TF-IDF vectorizer
│-- app.py                       # Streamlit app
│-- requirements.txt             # Required dependencies
│-- README.md                    # Documentation
```

## 🛠 Installation & Setup
### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/yourusername/SentimentSnap.git
cd SentimentSnap
```

### **2️⃣ Create a Virtual Environment (Optional but Recommended)**
```bash
python -m venv venv
source venv/bin/activate   # For macOS/Linux
venv\Scripts\activate      # For Windows
```

### **3️⃣ Install Dependencies**
```bash
pip install -r requirements.txt
```

### **4️⃣ Run the Application**
```bash
streamlit run app.py
```

## 🎯 How to Use
1. Enter text in the input box.
2. Click on **Analyze Sentiment**.
3. The app will **translate (if needed), clean, and classify** the text.
4. View **sentiment results, confidence scores, and probability bars**.

## 🔧 Technologies Used
- **Python** (NLTK, Scikit-learn, BeautifulSoup, Joblib)
- **Streamlit** (UI Framework)
- **Google Translate API** (Language Detection & Translation)

## ⚠️ Disclaimer
🚀 *This AI-powered sentiment analysis tool is for educational purposes only. The results may not always be 100% accurate, and users should interpret them with discretion.*

## 📌 Author
- **[Mohamed Nihal]** – *ML Engineer*

---
⭐ *If you like this project, don't forget to star ⭐ the repository!*

