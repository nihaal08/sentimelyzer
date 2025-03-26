import streamlit as st
import joblib
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from googletrans import Translator, LANGUAGES
import os

# Set up page configuration
st.set_page_config(
    page_title="SentimentSnap - AI Sentiment Analysis",
    page_icon="✨",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Function to download NLTK data
def download_nltk_data():
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
    except Exception as e:
        st.error(f"Error downloading NLTK data: {str(e)}")

# Load models
@st.cache_resource
def load_models():
    try:
        if not os.path.exists('sentiment_model.joblib') or not os.path.exists('tfidf_vectorizer.joblib'):
            raise FileNotFoundError("Model or vectorizer file not found")
        model = joblib.load('sentiment_model.joblib')
        vectorizer = joblib.load('tfidf_vectorizer.joblib')
        return model, vectorizer
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None

# Text translation function
def translate_text(text, target_lang='en'):
    try:
        translator = Translator()
        translated = translator.translate(text, dest=target_lang)
        return translated.text, translated.src
    except Exception as e:
        st.error(f"Error in translation: {str(e)}")
        return text, None

# Cleaning function
def clean_text(text):
    try:
        text = BeautifulSoup(text, 'html.parser').get_text()
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
        return ' '.join(tokens)
    except Exception as e:
        st.error(f"Error in text cleaning: {str(e)}")
        return ""

# Prediction function
def predict_sentiment(text, model, vectorizer):
    try:
        if not text:
            return None
        cleaned_text = clean_text(text)
        text_tfidf = vectorizer.transform([cleaned_text])
        prediction = model.predict(text_tfidf)[0]
        probability = model.predict_proba(text_tfidf)[0]
        
        return {
            'cleaned_text': cleaned_text,
            'sentiment': 'Positive' if prediction == 1 else 'Negative',
            'confidence': max(probability),
            'positive_prob': probability[1],
            'negative_prob': probability[0]
        }
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
        return None

# Main application
def main():
    st.markdown("<h1 style='text-align: center; color: #007BFF;'>SentimentSnap</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: #555;'>AI-Powered Sentiment Analysis with Translation</h4>", unsafe_allow_html=True)
    st.write("\nEnter text to analyze its sentiment")

    # Download NLTK data
    download_nltk_data()

    # Load models
    model, vectorizer = load_models()
    
    if model is None or vectorizer is None:
        st.error("Cannot proceed without models. Please check the error messages above.")
        return

    # Text input
    user_input = st.text_area("Enter your text here", height=150)
    
    if st.button("Analyze Sentiment"):
        if user_input:
            with st.spinner("Analyzing..."):
                translated_text, detected_lang = translate_text(user_input)
                
                if detected_lang and detected_lang != 'en':
                    st.write(f"**Detected Language:** {LANGUAGES.get(detected_lang, 'Unknown').capitalize()}")
                    st.write(f"**Translated Text:** {translated_text}")
                
                result = predict_sentiment(translated_text, model, vectorizer)
                
                if result:
                    # Display results
                    st.subheader("Analysis Results")
                    
                    # Sentiment indicator
                    if result['sentiment'] == 'Positive':
                        st.success(f"Sentiment: {result['sentiment']}")
                    else:
                        st.error(f"Sentiment: {result['sentiment']}")
                    
                    # Detailed results
                    st.write(f"**Cleaned Text:** {result['cleaned_text']}")
                    st.write(f"**Confidence Score:** {result['confidence']:.3f}")
                    
                    # Probability bars
                    col1, col2 = st.columns(2)
                    with col1:
                        st.progress(result['positive_prob'])
                        st.write(f"**Positive:** {result['positive_prob']:.3f}")
                    with col2:
                        st.progress(result['negative_prob'])
                        st.write(f"**Negative:** {result['negative_prob']:.3f}")
                    
                    # Sentiment emoji
                    emoji = "😊" if result['sentiment'] == 'Positive' else "😞"
                    st.write(f"**Feeling:** {emoji}")
                else:
                    st.warning("Analysis failed. Check error messages above.")
        else:
            st.warning("Please enter some text to analyze!")

if __name__ == '__main__':
    main()

# Footer Disclaimer
st.markdown("""
    <hr>
    <p style="text-align: center; font-size: 12px; color: gray;">
        🚀 <b>Disclaimer:</b> This AI-powered sentiment analysis tool is for educational purposes only. 
        The results may not always be 100% accurate, and users should interpret them with discretion.  
    </p>
""", unsafe_allow_html=True)
