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
import asyncio
from concurrent.futures import ThreadPoolExecutor
from wordcloud import WordCloud
from googletrans import Translator

st.set_page_config(layout="wide")

nltk.download('punkt_tab', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('vader_lexicon', quiet=True)

STOPWORDS = set(stopwords.words('english'))

def load_css():
    st.markdown("""
        <style>
            body {
                color: #D2C9A3;               
                font-family: Arial, sans-serif;
            }
            h1, h2, h3, h4, h5, h6 {
                color: brown;               
                text-transform: uppercase; 
            }
            .stSidebar {
                background-color: black;  
                color: #D2C9A3;   
                text-align:center;
                font-size:10px
            }
            .stButton > button {
                background-color: #D2C9A3;   
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
                background-color: brown;
                color: black;
            }
        </style>
    """, unsafe_allow_html=True)

load_css()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if 'page' not in st.session_state:
    st.session_state.page = "Home"

# Preprocessing Text Function
def preprocess_text(text):
    """ Preprocesses the given text by demojizing, cleaning, and tokenizing."""
    text = emoji.demojize(text)  # Convert emojis to text
    text = clean_text(text)  # Clean the text
    tokens = word_tokenize(text.lower())  # Tokenize and lowercase
    lem = WordNetLemmatizer()  # Initialize lemmatizer
    cleaned_tokens = [lem.lemmatize(token) for token in tokens if token not in STOPWORDS]
    return ' '.join(cleaned_tokens)  # Return cleaned tokens as a single string

def clean_text(text):
    """Cleans input text by removing unwanted characters."""
    text = re.sub(r'[^\w\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

def filter_unwanted_comments(reviews, unwanted_keywords):
    """Filters out reviews containing any of the unwanted keywords."""
    filtered_reviews = []
    for review in reviews:
        if not any(keyword in review['Description'].lower() for keyword in unwanted_keywords):
            filtered_reviews.append(review)
    return filtered_reviews

def chat_and_help_section():
    st.title("Chat & Help Assistant")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("How can I assist you? (Type 'help' for options)"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        response = generate_response(prompt)
        with st.chat_message("assistant"):
            st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})

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
            "1. Ensure it contains the following columns: Id, ProductId, UserId, ProfileName, HelpfulnessNumerator, "
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
            "2. The system will identify likely fake reviews based on ratings and content using NLP techniques.\n"
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
            "- Scrape reviews\n"
            "- Upload datasets\n"
            "- Analyze text\n"
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
            "This dashboard includes functionalities like:\n"
            "1. Scrape reviews.\n"
            "2. Upload datasets.\n"
            "3. Text Analysis.\n"
            "Feel free to navigate through the sidebar for access to various sections."
        )
    }

    response = help_responses.get(prompt, 
    "I didn’t understand your question. You can type:\n- Scrape Reviews\n- Upload Dataset\n- Text Analysis\n- Getting Started\n\n"
    "Or type 'help' for options and guidance.")

    return response

def initialize_database(db_name, create_statement):
    try:
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()
        cursor.execute(create_statement)
        conn.commit()
    except Exception as e:
        st.error(f"Error initializing database {db_name}: {e}")
    finally:
        conn.close()

# Table creation statements
initialize_database(
    'scraped_sentiment_analysis.db',
    '''
    CREATE TABLE IF NOT EXISTS scraped_reviews (
        id INTEGER PRIMARY KEY,
        name TEXT,
        rating TEXT,
        title TEXT,
        description TEXT,
        sentiment TEXT, 
        translated_description TEXT  
    )
    '''
)

initialize_database(
    'uploaded_sentiment_analysis.db',
    '''
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
    '''
)

initialize_database(
    'uploaded_output_analysis.db',
    '''
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
    '''
)

def translate_text(text):
    translator = Translator()
    try:
        translated = translator.translate(text, dest='en')
        return translated.text
    except Exception as e:
        st.error(f"Translation Error: {e}")
        return text

def get_request_headers():
    return {
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36'
    }

async def single_page_scrape(url, page_number, encountered_reviews):
    reviews = []
    try:
        response = requests.get(f"{url}&pageNumber={page_number}", headers=get_request_headers())
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        boxes = soup.select('div[data-hook="review"]')

        for box in boxes:
            review_title = box.select_one('[data-hook="review-title"]').text.strip() if box.select_one('[data-hook="review-title"]') else 'N/A'
            review_description = box.select_one('[data-hook="review-body"]').text.strip() if box.select_one('[data-hook="review-body"]') else 'N/A'
            review_title = re.sub(r'\b\d+\.\d+\s+out of\s+\d+\s+stars\b', '', review_title, flags=re.IGNORECASE).strip()
            identifier = f"{review_title}_{review_description}"

            if identifier in encountered_reviews:
                continue
            encountered_reviews.add(identifier)

            review = {
                'Name': box.select_one('[class="a-profile-name"]').text if box.select_one('[class="a-profile-name"]') else 'N/A',
                'Rating': box.select_one('[data-hook="review-star-rating"]').text.split(' out')[0] if box.select_one('[data-hook="review-star-rating"]') else 'N/A',
                'Title': review_title,
                'Description': clean_text(review_description),
            }
            review['Translated_Description'] = translate_text(review['Description'])
            review['Sentiment'] = analyze_sentiment(review['Translated_Description'])

            reviews.append(review)
    except requests.exceptions.RequestException as e:
        st.error(f"Error on page {page_number}: {e}")
    return reviews

async def scrape_reviews(url, pages):
    reviews = []
    encountered_reviews = set()
    tasks = []

    for page_number in range(1, pages + 1):
        tasks.append(single_page_scrape(url, page_number, encountered_reviews))

    responses = await asyncio.gather(*tasks)
    for response in responses:
        reviews.extend(response)

    unwanted_keywords = ['fake', 'unverified', 'not helpful', 'spam']
    return filter_unwanted_comments(reviews, unwanted_keywords)

def analyze_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    sentences = nltk.tokenize.sent_tokenize(text)

    overall_sentiment = 0

    for sentence in sentences:
        sentiment_scores = analyzer.polarity_scores(sentence)
        overall_sentiment += sentiment_scores['compound']

    if overall_sentiment > 0.05:
        return 'Positive'
    elif overall_sentiment < -0.05:
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

def clear_database(table_name):
    """Clear all entries from the specified database table."""
    if 'scraped' in table_name:
        db_name = 'scraped_sentiment_analysis.db'
    else:
        db_name = 'uploaded_output_analysis.db'

    try:
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()
        cursor.execute(f'DELETE FROM {table_name}') 
        conn.commit()
        st.success(f"All entries in {table_name} have been cleared.")
    except Exception as e:
        st.error(f"Error clearing {table_name}: {e}")
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
    - **Upload Dataset**: Analyze your own CSV files for sentiment.
    - **Text Analysis**: Input custom text to understand sentiment.
    - **Scrape Reviews**: Collect Amazon product reviews with ease.
    - **Fake Review Detection**: Identify potentially fake reviews from your dataset using NLP.
    - **History**: View past analyses and results for comparison.
    
    **Explore More** using the sidebar to navigate through the features of this dashboard!
    """)
    st.markdown("### Quick Tips:")
    st.markdown("- Use the clear buttons in History to manage your databases.")
    st.markdown("- Visualizations provide clear insights into data trends and sentiments.")

def display_navbar():
    st.sidebar.write("") 
    st.sidebar.image('sentimelyzer.png', use_container_width=True)
    st.sidebar.write("") 
    st.sidebar.write("") 
    if st.sidebar.button("Home", key="home_button"):
        st.session_state.page = "Home"
    if st.sidebar.button("About", key="about_button"):
        st.session_state.page = "About"
    if st.sidebar.button("Text Analysis", key="text_analysis_button"):
        st.session_state.page = "Text Analysis"
    if st.sidebar.button("Upload Dataset", key="dataset_upload_button"):
        st.session_state.page = "Dataset Upload"
    if st.sidebar.button("Scrape Reviews", key="scrape_reviews_button"):
        st.session_state.page = "Scrape Reviews"
    if st.sidebar.button("Fake Review Detection", key="fake_review_button"):
        st.session_state.page = "Fake Review Detection"
    if st.sidebar.button("History", key="history_button"):
        st.session_state.page = "History"
    if st.sidebar.button("Support", key="support_button"):
        st.session_state.page = "Support"

display_navbar()

if st.session_state.page == "Home":
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.markdown("""
        <h1 style="text-align: center;">Welcome To SentimelyzeR</h1>
        <h3 style="text-align: center;">Your Gateway to Understanding Sentiments</h3>
    """, unsafe_allow_html=True)

if st.session_state.page == "About":
    st.title("About")
    st.markdown("""
        Sentimelyzer is your go-to tool for effortlessly analyzing Amazon product reviews to uncover customer sentiment. 
        With features like scraping reviews directly from Amazon URLs, uploading your own CSV datasets for analysis, and 
        evaluating custom text inputs, you can gain valuable insights into consumer opinions. 
        Our interactive visualizations make it easy to interpret data trends and improve your product offerings based on real feedback.
    """)
    st.write("### Features of SentimelyzeR:")
    st.write("- **Scrape Reviews:** Instantly gather reviews from Amazon products.")
    st.write("- **Upload Dataset:** Analyze your own CSV files containing product reviews.")
    st.write("- **Text Analysis:** Understand the sentiment of any text you provide.")
    st.write("- **Fake Review Detection:** Identify potentially fake reviews using NLP.")
    st.write("- **History:** Access your previous analyses and results.")
    
if st.session_state.page == "Support":
    show_tutorial()
    st.write("---")
    chat_and_help_section()

if st.session_state.page == "Scrape Reviews":
    st.header("SCRAPE REVIEWS FROM AMAZON")
    url_input = st.text_input("ENTER AMAZON REVIEW URL:")
    pages_input = st.number_input("PAGES TO SCRAPE:", 1, 50, 1)

    if st.button("SCRAPE REVIEWS", key="start_scrape"):
        if url_input:
            with st.spinner('SCRAPING DATA...'):
                scraped_reviews = asyncio.run(scrape_reviews(url_input, pages_input))
            st.success("DATA SCRAPING COMPLETE!")

            df_reviews = pd.DataFrame(scraped_reviews)
            st.write("### SCRAPED REVIEWS")
            st.write(df_reviews)

            if not df_reviews.empty:
                for _, row in df_reviews.iterrows():
                    insert_scraped_review(row['Name'], row['Rating'], row['Title'], 
                                          row['Description'], row['Sentiment'], 
                                          row['Translated_Description'])

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

if st.session_state.page == "Dataset Upload":
    st.header("UPLOAD DATASET")
    uploaded_file = st.file_uploader("CHOOSE A CSV FILE", type="csv")
    
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("### UPLOADED DATA")
        st.write(data)

        required_columns = ['Id', 'ProductId', 'UserId', 'ProfileName', 
                            'HelpfulnessNumerator', 'HelpfulnessDenominator', 
                            'Score', 'Time', 'Summary', 'Text']
        
        if all(col in data.columns for col in required_columns):
            data['Processed_Text'] = data['Text'].apply(preprocess_text)
            data['Sentiment'] = data['Processed_Text'].apply(analyze_sentiment)

            for _, row in data.iterrows():
                insert_uploaded_output_review(row['ProductId'], row['UserId'], row['ProfileName'], 
                                              row['HelpfulnessNumerator'], row['HelpfulnessDenominator'],
                                              row['Score'], row['Time'], row['Summary'], 
                                              row['Text'], row['Processed_Text'], row['Sentiment'])

            st.success("DATA UPLOADED AND INSERTED INTO OUTPUT DATABASE!")

            export_to_csv(data, "uploaded_reviews.csv")

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

if st.session_state.page == "Text Analysis":
    st.header("ANALYZE CUSTOM TEXT")
    user_input_text = st.text_area("ENTER TEXT:")

    language_option = st.selectbox("Select Language for Analysis:", ["English", "Other Language"])

    if st.button("Analyze Text"):
        if user_input_text:
            if language_option == "English":
                sentiment_result = analyze_sentiment(user_input_text)
                explanation = ""
                
                if sentiment_result == 'Positive':
                    explanation = "The text expresses positive sentiment likely due to the use of uplifting words and phrases, indicating a favorable perception."
                elif sentiment_result == 'Negative':
                    explanation = "The text conveys a negative sentiment, which may arise from the presence of critical or adverse vocabulary, suggesting dissatisfaction."
                else:
                    explanation = "The sentiment is neutral, signifying that the text lacks strong emotional indicators, often presenting a balanced view without bias."

                st.write(f"*Sentiment of Text:* {sentiment_result}")
                st.write(f"*Reason:* {explanation}")

            elif language_option == "Other Language":
                translated_text = translate_text(user_input_text)
                sentiment_result = analyze_sentiment(translated_text)
                explanation = ""
                
                if sentiment_result == 'Positive':
                    explanation = "The translated text expresses positive sentiment likely due to the use of uplifting words and phrases, indicating a favorable perception."
                elif sentiment_result == 'Negative':
                    explanation = "The translated text conveys a negative sentiment, which may arise from the presence of critical or adverse vocabulary, suggesting dissatisfaction."
                else:
                    explanation = "The translated text has a neutral sentiment, signifying that it lacks strong emotional indicators, often presenting a balanced view without bias."

                st.write("### TRANSLATED TEXT")
                st.write(translated_text)
                st.write(f"*Sentiment of Translated Text:* {sentiment_result}")
                st.write(f"*Reason:* {explanation}")
        else:
            st.write("*Please enter text for analysis.*")
            
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
        clear_database('scraped_reviews')  # Use the table name

    if st.button("Clear Uploaded Output Reviews Database"):
        clear_database('output_reviews')  # Use the table name

if st.session_state.page == "Fake Review Detection":
    st.title("Fake Review Detection")
    st.header("Upload Reviews for Fake Review Identification")

    fake_review_uploaded_file = st.file_uploader("CHOOSE A CSV FILE WITH REVIEWS", type="csv")
    
    if fake_review_uploaded_file is not None:
        # Read the uploaded file
        fake_reviews_data = pd.read_csv(fake_review_uploaded_file)

        st.write("### UPLOADED REVIEWS")
        st.write(fake_reviews_data)

        # Required columns
        required_columns = ['Id', 'ProductId', 'UserId', 'ProfileName', 
                            'HelpfulnessNumerator', 'HelpfulnessDenominator', 
                            'Score', 'Time', 'Summary', 'Text']
        
        # Check if all required columns are present
        if all(col in fake_reviews_data.columns for col in required_columns):
            # Initialize Sentiment Analyzer
            sia = SentimentIntensityAnalyzer()
            
            # Function to analyze sentiment and classify as Fake/Real
            def classify_fake_reviews(row):
                sentiment_score = sia.polarity_scores(row['Text'])['compound']  # Analyze based on the 'Text' column
                rating = float(row['Score'])
                
                # Fake logic: Low rating with a positive sentiment score
                if rating < 3 and sentiment_score >= 0.05:
                    return 'Fake'
                elif rating >= 3:
                    return 'Real'
                return 'Uncertain'

            # Apply the function and create a new column
            fake_reviews_data['Is_Fake'] = fake_reviews_data.apply(classify_fake_reviews, axis=1)

            st.write("#### Fake Review Detection Results")
            results_df = fake_reviews_data[['Id', 'ProductId', 'UserId', 'ProfileName', 
                                              'Score', 'Text', 'Is_Fake']].copy()
            st.write(results_df)

            # Count of each category
            fake_count = fake_reviews_data[fake_reviews_data['Is_Fake'] == 'Fake'].shape[0]
            real_count = fake_reviews_data[fake_reviews_data['Is_Fake'] == 'Real'].shape[0]
            uncertain_count = fake_reviews_data[fake_reviews_data['Is_Fake'] == 'Uncertain'].shape[0]

            st.write("### ANALYSIS INSIGHTS")
            st.write(f"Total Reviews: {len(fake_reviews_data)}")
            st.write(f"Fake Reviews Detected: {fake_count}")
            st.write(f"Real Reviews Detected: {real_count}")
            st.write(f"Uncertain Reviews: {uncertain_count}")

            # Pie chart of results
            fig = px.pie(names=['Fake', 'Real', 'Uncertain'], values=[fake_count, real_count, uncertain_count],
                         title='Distribution of Fake vs Real Reviews',
                         color_discrete_sequence=['green', 'red', 'yellow'])
            st.plotly_chart(fig)

        else:
            st.write("*UPLOADED CSV MUST CONTAIN THE FOLLOWING COLUMNS: Id, ProductId, UserId, ProfileName, HelpfulnessNumerator, HelpfulnessDenominator, Score, Time, Summary, Text.*")
