import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
import emoji
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import sqlite3
import nltk
from concurrent.futures import ThreadPoolExecutor
import time
from wordcloud import WordCloud
from googletrans import Translator

# Set up Streamlit page layout
st.set_page_config(layout="wide")
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('vader_lexicon', quiet=True)
try:
    STOPWORDS = set(stopwords.words('english'))
except Exception as e:
    st.error(f"Error loading stopwords: {e}")
    STOPWORDS = set()

# Load CSS for styling
def load_css():
    st.markdown(
        """
        <style>
            body {
                color: green;               
                font-family: Arial, sans-serif;
            }
            
            h1, h2, h3, h4, h5, h6 {
                color: green;               
                text-transform: uppercase; 
            }
            .stSidebar {
                background-color: black;  
                color: darkgreen;                
            }
            .stButton > button {
                background-color: green;    
                color: black;             
                border: none;             
                padding: 10px;           
                border-radius: 5px;     
                cursor: pointer;         
                width: 100%;             
                text-align: center;      
                font-size: 16px;         
            }
            .stButton > button:hover {
                background-color: lightgreen;
                color: black;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

load_css()

# Initialize the message state for chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize the page state
if 'page' not in st.session_state:
    st.session_state.page = "Home"

# Chat and Help Section
def chat_and_help_section():
    st.title("Chat & Help Assistant")

    # Display previous messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    if prompt := st.chat_input("How can I assist you? (Type 'help' for options)"):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        response = generate_response(prompt)
        with st.chat_message("assistant"):
            st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})

# Extend the generate_response function
def generate_response(prompt):
    prompt = prompt.lower()  # Normalize input
    help_responses = {
        'scrape reviews': (
            "### Scrape Reviews\n"
            "To scrape reviews, please follow these steps:\n"
            "1. Paste the Amazon product review URL.\n"
            "2. Specify the number of pages to scrape.\n"
            "3. Click the 'SCRAPE REVIEWS' button.\n"
            "Make sure the URL leads to a valid product page with reviews!"
        ),
        'upload dataset': (
            "### Upload Dataset\n"
            "To upload a CSV dataset:\n"
            "1. Ensure it contains these columns: Id, ProductId, UserId, ProfileName, HelpfulnessNumerator, "
            "HelpfulnessDenominator, Score, Time, Summary, Text.\n"
            "2. Click 'UPLOAD DATASET' to analyze your reviews."
        ),
        'text analysis': (
            "### Text Analysis\n"
            "Simply enter the text you wish to analyze in the provided area. You'll receive sentiment results "
            "for positive, negative, or neutral sentiment."
        ),
        'fake review detection': (
            "### Fake Review Detection\n"
            "1. Upload a CSV file that includes reviews.\n"
            "2. The system will identify likely fake reviews based on ratings and content.\n"
            "3. Make sure the CSV file has these essential columns: category, rating, label, text_."
        ),
        'history': (
            "### History\n"
            "You can review all scraped and uploaded datasets here. Filter, download, and analyze past reviews’ "
            "sentiments in various ways."
        ),
        'help': (
            "### Need Help?\n"
            "Ask me specific questions or request assistance about features like:\n"
            "- Scraping reviews\n"
            "- Uploading datasets\n"
            "- Analyzing text\n"
            "If you're unsure where to start, just type 'Getting Started' for tips!"
        ),
        'getting started': (
            "### Getting Started\n"
            "To kick off, you might want to:\n"
            "1. Scrape some product reviews: Go to 'Scrape Reviews'.\n"
            "2. Analyze your dataset: Try 'Upload Dataset'.\n"
            "3. Check out 'Text Analysis' for sentiment on custom text!"
        ),
        'tutorial': (
            "### Interactive Tutorial\n"
            "This dashboard offers functionalities like:\n"
            "1. Scraping reviews\n"
            "2. Uploading datasets\n"
            "3. Analyzing custom text\n"
            "Feel free to navigate through the sidebar for access to various sections."
        )
    }

    # Add a variety of responses for better understanding
    response = help_responses.get(prompt, 
    "I didn’t understand your question. You can type:\n- Scrape Reviews\n- Upload Dataset\n- Text Analysis\n- Getting Started\n\n"
    "Or type 'help' for options and guidance.")

    return response

# Database and Helper Functions
def initialize_scraped_database():
    conn = sqlite3.connect('scraped_sentiment_analysis.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS scraped_reviews (
            id INTEGER PRIMARY KEY,
            name TEXT,
            rating TEXT,
            title TEXT,
            description TEXT,
            sentiment TEXT, 
            translated_description TEXT  
        )
    ''')
    conn.commit()
    conn.close()

def initialize_uploaded_database():
    conn = sqlite3.connect('uploaded_sentiment_analysis.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS uploaded_reviews (
            id INTEGER PRIMARY KEY,
            product_id TEXT,
            user_id TEXT,
            profile_name TEXT,
            helpfulness_numerator INTEGER,
            helpfulness_denominator INTEGER,
            score INTEGER,
            time INTEGER,
            summary TEXT,
            text TEXT,
            processed_text TEXT,
            sentiment TEXT
        )
    ''')
    conn.commit()
    conn.close()

def initialize_uploaded_output_database():
    conn = sqlite3.connect('uploaded_output_analysis.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS output_reviews (
            id INTEGER PRIMARY KEY,
            product_id TEXT,
            user_id TEXT,
            profile_name TEXT,
            helpfulness_numerator INTEGER,
            helpfulness_denominator INTEGER,
            score INTEGER,
            time INTEGER,
            summary TEXT,
            text TEXT,
            processed_text TEXT,
            sentiment TEXT
        )
    ''')
    conn.commit()
    conn.close()

def translate_text(text):
    translator = Translator()
    try:
        translated = translator.translate(text, dest='en')
        return translated.text
    except Exception as e:
        st.error(f"Translation Error: {e}")
        return text

def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def filter_unwanted_comments(reviews, unwanted_keywords):
    filtered_reviews = []
    for review in reviews:
        if not any(keyword.lower() in review['Description'].lower() for keyword in unwanted_keywords):
            filtered_reviews.append(review)
    return filtered_reviews

def get_request_headers():
    return {
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36'
    }

def single_page_scrape(url, page_number, encountered_reviews):
    reviews = []
    try:
        response = requests.get(f"{url}&pageNumber={page_number}", headers=get_request_headers())
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        boxes = soup.select('div[data-hook="review"]')
        
        for box in boxes:
            review_title = box.select_one('[data-hook="review-title"]').text.strip()
            # Remove the rating from the title (e.g., "X out of Y")
            review_title = re.sub(r'\d+(\.\d+)? out of \d+', '', review_title).strip()  # This will remove "4.0 out of 5"

            review_description = box.select_one('[data-hook="review-body"]').text.strip()
            # Remove the "Read more" text if present
            review_description = review_description.replace("Read more", "").strip()
            
            # Use title and description or unique identifier to check for duplicates
            identifier = f"{review_title}_{review_description}"
            if identifier in encountered_reviews:
                continue  # Skip duplicate review
            
            encountered_reviews.add(identifier)  # Mark this review as seen

            review = {
                'Name': box.select_one('[class="a-profile-name"]').text if box.select_one('[class="a-profile-name"]') else 'N/A',
                'Rating': box.select_one('[data-hook="review-star-rating"]').text.split(' out')[0] if box.select_one('[data-hook="review-star-rating"]') else 'N/A',
                'Title': review_title,
                'Description': review_description,
            }
            
            review['Description'] = clean_text(review['Description'])
            review['Translated_Description'] = translate_text(review['Description'])
            review['Sentiment'] = analyze_sentiment(review['Translated_Description'])

            reviews.append(review)
    except requests.exceptions.RequestException as e:
        st.error(f"Error on page {page_number}: {e}")
    return reviews

@st.cache_data(show_spinner=False)
def scrape_reviews(url, pages):
    reviews = []
    encountered_reviews = set()  # Set to track unique reviews
    with ThreadPoolExecutor() as executor:
        futures = []
        for page_number in range(1, pages + 1):
            futures.append(executor.submit(single_page_scrape, url, page_number, encountered_reviews))
            time.sleep(1)

        for future in futures:
            reviews.extend(future.result())

    unwanted_keywords = ['fake', 'unverified', 'not helpful', 'spam']
    reviews = filter_unwanted_comments(reviews, unwanted_keywords)
    return reviews

lem = WordNetLemmatizer()

@st.cache_data(show_spinner=False)
def preprocess_text(text):
    text = emoji.demojize(text)
    text = clean_text(text)
    tokens = word_tokenize(text.lower())
    cleaned_tokens = [lem.lemmatize(token) for token in tokens if token not in STOPWORDS]
    return ' '.join(cleaned_tokens)

analyzer = SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    negations = ['not', 'no', 'never', 'without', 'barely']
    sentences = nltk.tokenize.sent_tokenize(text)

    overall_sentiment = 0

    for sentence in sentences:
        sentiment_scores = analyzer.polarity_scores(sentence)
        overall_sentiment += sentiment_scores['compound']

        if any(negation in sentence.lower() for negation in negations):
            overall_sentiment -= 0.2

    if overall_sentiment > len(sentences) * 0.05:
        return 'Positive'
    elif overall_sentiment < -len(sentences) * 0.05:
        return 'Negative'
    else:
        return 'Neutral'

def generate_insights(data):
    positive_reviews = data[data['Sentiment'] == 'Positive']
    negative_reviews = data[data['Sentiment'] == 'Negative']
    neutral_reviews = data[data['Sentiment'] == 'Neutral']

    insights = []
    insights.append(f"{len(positive_reviews)} positive reviews found.")
    insights.append(f"{len(negative_reviews)} negative reviews found.")
    insights.append(f"{len(neutral_reviews)} neutral reviews found.")
    return insights

def insert_scraped_review(name, rating, title, description, sentiment, translated_description):
    try:
        conn = sqlite3.connect('scraped_sentiment_analysis.db')
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO scraped_reviews (name, rating, title, description, sentiment, translated_description) 
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (name, rating, title, description, sentiment, translated_description))
        conn.commit()
    except Exception as e:
        st.error(f"Error inserting review into scraped_reviews: {e}")
    finally:
        conn.close()

def insert_uploaded_output_review(product_id, user_id, profile_name, helpfulness_numerator, helpfulness_denominator,
                                   score, time, summary, text, processed_text, sentiment):
    try:
        conn = sqlite3.connect('uploaded_output_analysis.db')
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO output_reviews (product_id, user_id, profile_name, helpfulness_numerator, 
                                         helpfulness_denominator, score, time, summary, text, processed_text, sentiment) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (product_id, user_id, profile_name, helpfulness_numerator, helpfulness_denominator,
              score, time, summary, text, processed_text, sentiment))
        conn.commit()
    except Exception as e:
        st.error(f"Error inserting review into output_reviews: {e}")
    finally:
        conn.close()

def fetch_all_reviews(table_name):
    if 'scraped' in table_name:
        db_name = 'scraped_sentiment_analysis.db'
    else:
        db_name = 'uploaded_output_analysis.db'

    try:
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()
        cursor.execute(f'SELECT * FROM {table_name}')
        data = cursor.fetchall()
    except Exception as e:
        st.error(f"Error fetching reviews from {table_name}: {e}")
        return []
    finally:
        conn.close()
    return data

def clear_scraped_database():
    try:
        conn = sqlite3.connect('scraped_sentiment_analysis.db')
        cursor = conn.cursor()
        cursor.execute('DELETE FROM scraped_reviews')
        conn.commit()
        st.success("All scraped reviews have been cleared.")
    except Exception as e:
        st.error(f"Error clearing scraped database: {e}")
    finally:
        conn.close()

def clear_uploaded_output_database():
    try:
        conn = sqlite3.connect('uploaded_output_analysis.db')
        cursor = conn.cursor()
        cursor.execute('DELETE FROM output_reviews')
        conn.commit()
        st.success("All uploaded output reviews have been cleared.")
    except Exception as e:
        st.error(f"Error clearing uploaded output database: {e}")
    finally:
        conn.close()

def export_to_csv(data, filename):
    csv = data.to_csv(index=False)
    st.download_button(label="Download Results as CSV", data=csv, file_name=filename, mime='text/csv')

def show_tutorial():
    st.subheader("Interactive Tutorial")
    st.markdown("""
    Welcome to the Sentiment Analysis Dashboard! Here’s how to get started:
    - **Home**: Overview of functionalities and quick insights.
    - **Scrape Reviews**: Collect Amazon product reviews with ease.
    - **Upload Dataset**: Analyze your own CSV files for sentiment.
    - **Text Analysis**: Input custom text to understand sentiment.
    - **Fake Review Detection**: Identify potentially fake reviews from your dataset.
    - **History**: View past analyses and results for comparison.
    
    **Explore More** using the sidebar to navigate through the features of this dashboard!
    """)
    st.markdown("### Quick Tips:")
    st.markdown("- Use the clear buttons in History to manage your databases.")
    st.markdown("- Visualizations provide clear insights into data trends and sentiments.")

def display_navbar():
    st.sidebar.image('LOGO.png', use_column_width=True)
    if st.sidebar.button("Home", key="home_button"):
        st.session_state.page = "Home"
    if st.sidebar.button("Scrape Reviews", key="scrape_reviews_button"):
        st.session_state.page = "Scrape Reviews"
    if st.sidebar.button("Upload Dataset", key="dataset_upload_button"):
        st.session_state.page = "Dataset Upload"
    if st.sidebar.button("Text Analysis", key="text_analysis_button"):
        st.session_state.page = "Text Analysis"
    if st.sidebar.button("Fake Review Detection", key="fake_review_button"):
        st.session_state.page = "Fake Review Detection"
    if st.sidebar.button("History", key="history_button"):
        st.session_state.page = "History"
    if st.sidebar.button("Support", key="support_button"):
        st.session_state.page = "Support"  # Combined option for Chat & Help and Tutorial

display_navbar()

if st.session_state.page == "Home":
    st.title("WELCOME TO THE SENTIMENT ANALYSIS DASHBOARD!")

    # Updated content for the home page
    st.markdown("""
        ### Analyze Amazon Product Reviews Effortlessly!
        This interactive dashboard is designed to help you scrape, analyze, and visualize Amazon reviews for effective sentiment analysis.
        
        #### Get Started with These Features:
        - **Scrape Reviews** from Amazon by just entering the product URL.
        - **Upload Your Datasets** in CSV format to analyze reviews on the go.
        - **Analyze Custom Text** inputs to gauge sentiments.
        - **Visual Insights** through charts and graphs to help you understand data better.
        
        #### Interactive and User-Friendly
        Explore the sidebar for a seamless experience, and take advantage of our support capabilities for guidance and assistance.
        
        ### Why Sentiment Analysis?
        Sentiment analysis helps businesses understand customer emotions, enhance product offerings, and improve overall sentiment through constructive feedback.
    """)

# Interactive Tutorial Page
if st.session_state.page == "Support":
    # Show both tutorial and chat functionality here
    show_tutorial()
    st.write("---")
    chat_and_help_section()

# Scrape Reviews Section
if st.session_state.page == "Scrape Reviews":
    st.header("SCRAPE REVIEWS FROM AMAZON")
    url_input = st.text_input("ENTER AMAZON REVIEW URL:")
    pages_input = st.number_input("PAGES TO SCRAPE:", 1, 50, 1)

    if st.button("SCRAPE REVIEWS", key="start_scrape"):
        if url_input:
            with st.spinner('SCRAPING DATA...'):
                scraped_reviews = scrape_reviews(url_input, pages_input)
            st.success("DATA SCRAPING COMPLETE!")

            df_reviews = pd.DataFrame(scraped_reviews)
            st.write("### SCRAPED REVIEWS")
            st.write(df_reviews)

            if not df_reviews.empty:
                for _, row in df_reviews.iterrows():
                    insert_scraped_review(row['Name'], row['Rating'], row['Title'], 
                                          row['Description'], row['Sentiment'], 
                                          row['Translated_Description'])

                # Export functionality
                export_to_csv(df_reviews, "scraped_reviews.csv")

                st.write("### SENTIMENT DISTRIBUTION")
                sentiment_counts = df_reviews['Sentiment'].value_counts()
                sentiment_counts_df = pd.DataFrame(sentiment_counts).reset_index()
                sentiment_counts_df.columns = ['Sentiment', 'Counts']

                st.write("#### Pie Chart of Sentiment Distribution")
                fig_pie = go.Figure(data=[go.Pie(labels=sentiment_counts_df['Sentiment'],
                                                   values=sentiment_counts_df['Counts'],
                                                   hole=0.3,
                                                   marker=dict(colors=['green' if sentiment == 'Positive'
                                                                       else 'red' if sentiment == 'Negative'
                                                                       else 'yellow'
                                                                       for sentiment in sentiment_counts_df['Sentiment']]))])
                st.plotly_chart(fig_pie)

                # Generate word clouds
                positive_reviews_text = ' '.join(df_reviews[df_reviews['Sentiment'] == 'Positive']['Description'])
                negative_reviews_text = ' '.join(df_reviews[df_reviews['Sentiment'] == 'Negative']['Description'])

                with st.container():
                    st.write("### WORD CLOUD FOR REVIEWS")
                    col1, col2 = st.columns(2)

                    with col1:
                        st.subheader("Positive Reviews Word Cloud")
                        if positive_reviews_text:
                            plt.figure(figsize=(4, 4))
                            wordcloud_pos = WordCloud(width=200, height=200, background_color='black').generate(positive_reviews_text)
                            plt.imshow(wordcloud_pos, interpolation='bilinear')
                            plt.axis('off')
                            st.pyplot(plt)
                            plt.close()
                        else:
                            st.write("*No positive reviews available to generate a word cloud.*")
                        
                    with col2:
                        st.subheader("Negative Reviews Word Cloud")
                        if negative_reviews_text:
                            plt.figure(figsize=(4, 4))
                            wordcloud_neg = WordCloud(width=200, height=200, background_color='black').generate(negative_reviews_text)
                            plt.imshow(wordcloud_neg, interpolation='bilinear')
                            plt.axis('off')
                            st.pyplot(plt)
                            plt.close()
                        else:
                            st.write("*No negative reviews available to generate a word cloud.*")

                st.write("### Bar Chart of Ratings by Sentiment")
                rating_count = df_reviews.groupby(['Sentiment', 'Rating']).size().reset_index(name='Counts')
                fig_bar = px.bar(rating_count, x='Rating', y='Counts', color='Sentiment', barmode='group',
                                 title='Bar Chart of Ratings by Sentiment', 
                                 color_discrete_map={
                                     'Positive': 'green',
                                     'Negative': 'red',
                                     'Neutral': 'yellow'
                                 },
                                 labels={'Counts': 'Number of Reviews', 'Rating': 'Rating'})
                st.plotly_chart(fig_bar)

                insights = generate_insights(df_reviews)
                st.write("### INSIGHTS")
                for insight in insights:
                    st.write(insight)
                
            else:
                st.write("*NO REVIEWS FOUND DURING SCRAPING.*")

# Dataset Upload Section
if st.session_state.page == "Dataset Upload":
    st.header("UPLOAD DATASET")
    uploaded_file = st.file_uploader("CHOOSE A CSV FILE", type="csv")
    
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("### UPLOADED DATA")
        st.write(data)

        required_columns = ['Id', 'ProductId', 'UserId', 'ProfileName', 'HelpfulnessNumerator', 
                            'HelpfulnessDenominator', 'Score', 'Time', 'Summary', 'Text']
        
        if all(col in data.columns for col in required_columns):
            # Process the dataset
            data['Processed_Text'] = data['Text'].apply(preprocess_text)
            data['Sentiment'] = data['Processed_Text'].apply(analyze_sentiment)

            for _, row in data.iterrows():
                insert_uploaded_output_review(row['ProductId'], row['UserId'], row['ProfileName'], 
                                              row['HelpfulnessNumerator'], row['HelpfulnessDenominator'],
                                              row['Score'], row['Time'], row['Summary'], 
                                              row['Text'], row['Processed_Text'], row['Sentiment'])

            st.success("DATA UPLOADED AND INSERTED INTO OUTPUT DATABASE!")

            # Export functionality
            export_to_csv(data, "uploaded_reviews.csv")

            # Check if data was successfully uploaded to the output database
            uploaded_reviews = fetch_all_reviews('output_reviews')
            if uploaded_reviews:
                st.write("### REVIEW RECORDS IN OUTPUT DATABASE:")
                db_df = pd.DataFrame(uploaded_reviews, columns=["ID", "ProductId", "UserId", "ProfileName", 
                                                                "HelpfulnessNumerator", "HelpfulnessDenominator", 
                                                                "Score", "Time", "Summary", "Text", "Processed_Text", "Sentiment"])
                st.write(db_df)
            else:
                st.write("*NO RECORDS FOUND IN THE OUTPUT DATABASE.*")

            st.write("### SENTIMENT DISTRIBUTION")
            sentiment_counts = data['Sentiment'].value_counts()
            sentiment_counts_df = pd.DataFrame(sentiment_counts).reset_index()
            sentiment_counts_df.columns = ['Sentiment', 'Counts']
            
            st.write("#### Pie Chart of Sentiment Distribution")
            fig_pie = go.Figure(data=[go.Pie(labels=sentiment_counts_df['Sentiment'],
                                               values=sentiment_counts_df['Counts'],
                                               hole=0.3, 
                                               marker=dict(colors=['green' if sentiment == 'Positive' 
                                                                   else 'red' if sentiment == 'Negative'
                                                                   else 'yellow'
                                                                   for sentiment in sentiment_counts_df['Sentiment']]))])
            st.plotly_chart(fig_pie)

            st.write("### Bar Chart of Scores by Sentiment")
            score_count = data.groupby(['Sentiment', 'Score']).size().reset_index(name='Counts')
            fig_bar = px.bar(score_count, x='Score', y='Counts', color='Sentiment', barmode='group',
                             title='Bar Chart of Scores by Sentiment', 
                             color_discrete_map={
                                 'Positive': 'green',
                                 'Negative': 'red',
                                 'Neutral': 'yellow'
                             },
                             labels={'Counts': 'Number of Reviews', 'Score': 'Score'})
            st.plotly_chart(fig_bar)

            insights = generate_insights(data)
            st.write("### INSIGHTS")
            for insight in insights:
                st.write(insight)

        else:
            st.write("*UPLOADED CSV MUST CONTAIN THE FOLLOWING COLUMNS: Id, ProductId, UserId, ProfileName, HelpfulnessNumerator, HelpfulnessDenominator, Score, Time, Summary, Text.*")

# Text Analysis Section
if st.session_state.page == "Text Analysis":
    st.header("ANALYZE CUSTOM TEXT")
    user_input_text = st.text_area("ENTER TEXT:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Analyze in English"):
            if user_input_text:
                sentiment_result = analyze_sentiment(user_input_text)
                st.write(f"*SENTIMENT OF TEXT:* {sentiment_result}")
            else:
                st.write("*PLEASE ENTER TEXT TO ANALYZE.*")

    with col2:
        if st.button("Analyze in Other Languages"):
            if user_input_text:
                translated_text = translate_text(user_input_text)
                st.write("### TRANSLATED TEXT")
                st.write(translated_text)
                sentiment_result = analyze_sentiment(translated_text)
                st.write(f"*SENTIMENT OF TRANSLATED TEXT:* {sentiment_result}")
            else:
                st.write("*PLEASE ENTER TEXT TO ANALYZE.*")

# History Section
if st.session_state.page == "History":
    st.header("History of Reviews")

    # Display scraped reviews
    st.subheader("Scraped Reviews")
    scraped_reviews = fetch_all_reviews('scraped_reviews')
    
    if scraped_reviews:
        df_scraped = pd.DataFrame(scraped_reviews, columns=["ID", "Name", "Rating", "Title", "Description", "Sentiment", "Translated_Description"])
        st.write("### ALL SCRAPED REVIEWS")
        st.write(df_scraped)
    else:
        st.write("*NO SCRAPED REVIEWS FOUND.*")

    # Display uploaded output reviews
    st.subheader("Uploaded Output Reviews")
    uploaded_output_reviews = fetch_all_reviews('output_reviews')
    
    if uploaded_output_reviews:
        df_uploaded_output = pd.DataFrame(uploaded_output_reviews, columns=["ID", "ProductId", "UserId", "ProfileName", 
                                                                           "HelpfulnessNumerator", "HelpfulnessDenominator", 
                                                                           "Score", "Time", "Summary", "Text", "Processed_Text", "Sentiment"])
        st.write("### ALL UPLOADED OUTPUT REVIEWS")
        st.write(df_uploaded_output)
    else:
        st.write("*NO UPLOADED OUTPUT REVIEWS FOUND.*")

    if st.button("Clear Scraped Reviews Database"):
        clear_scraped_database()

    if st.button("Clear Uploaded Output Reviews Database"):
        clear_uploaded_output_database()

# Fake Review Detection Page
if st.session_state.page == "Fake Review Detection":
    st.title("Fake Review Detection")
    st.header("Upload Reviews for Fake Review Identification")

    fake_review_uploaded_file = st.file_uploader("CHOOSE A CSV FILE WITH REVIEWS", type="csv")
    
    if fake_review_uploaded_file is not None:
        fake_reviews_data = pd.read_csv(fake_review_uploaded_file)
        
        st.write("### UPLOADED REVIEWS")
        st.write(fake_reviews_data)

        required_columns = ['category', 'rating', 'label', 'text_']
        if all(col in fake_reviews_data.columns for col in required_columns):
            fake_reviews_data['Is_Fake'] = fake_reviews_data['rating'].apply(lambda x: 'Fake' if x < 3 else 'Real')
            st.write("#### Fake Review Detection Results")
            results_df = fake_reviews_data[['category', 'rating', 'text_']].copy()
            results_df['Is_Fake'] = fake_reviews_data['Is_Fake']
            st.write(results_df)

            fake_count = fake_reviews_data[fake_reviews_data['Is_Fake'] == 'Fake'].shape[0]
            real_count = fake_reviews_data[fake_reviews_data['Is_Fake'] == 'Real'].shape[0]
            st.write("### ANALYSIS INSIGHTS")
            st.write(f"Total Reviews: {len(fake_reviews_data)}")
            st.write(f"Fake Reviews Detected: {fake_count}")
            st.write(f"Real Reviews Detected: {real_count}")

            fig = px.pie(names=['Fake', 'Real'], values=[fake_count, real_count],
                         title='Distribution of Fake vs Real Reviews',
                         color_discrete_sequence=['green', 'red'])
            st.plotly_chart(fig)

        else:
            st.write("*UPLOADED CSV MUST CONTAIN THE FOLLOWING COLUMNS: category, rating, label, text_*")

# Initialize the databases
initialize_scraped_database()
initialize_uploaded_database()
initialize_uploaded_output_database()
