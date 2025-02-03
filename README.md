# SentimelyzeR

## Overview

SentimelyzeR is an interactive dashboard for sentiment analysis, primarily focused on analyzing Amazon product reviews. It offers functionalities such as scraping reviews directly from Amazon URLs, uploading custom datasets in CSV format for sentiment analysis, and evaluating sentiment from user-provided text. The application uses Natural Language Processing (NLP) techniques and provides visualizations to aid users in understanding sentiment trends and insights derived from the reviews.

## Features

- **Scrape Reviews:** Instantly gather reviews from Amazon products via URLs.
- **Upload Dataset:** Analyze your own CSV files containing product reviews for sentiment analysis.
- **Text Analysis:** Input custom text to analyze and understand sentiment dynamics.
- **Fake Review Detection:** Identify potentially fake reviews using NLP and sentiment analysis techniques.
- **Sentiment Distribution Visualization:** Visualize sentiment distribution using pie charts and bar graphs.
- **Interactive Word Cloud:** Generate word clouds for positive and negative reviews to visualize common terms.
- **History:** Access previously scraped and uploaded reviews with filtering options.
- **Chat & Help Assistant:** An interactive assistant to guide users in using the dashboard features.

## Getting Started

### Prerequisites

- Python 3.x
- Required libraries (installed via `pip`):
    ```bash
    pip install streamlit pandas requests beautifulsoup4 nltk matplotlib plotly sqlite3 wordcloud googletrans
    ```

### Running the Application

To run the SentimelyzeR dashboard:

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/sentimelyzer.git
    cd sentimelyzer
    ```

2. Start the Streamlit server:
    ```bash
    streamlit run app.py
    ```

3. Open your web browser and navigate to `http://localhost:8501`.

## User Guide

### Scraping Reviews

1. Navigate to the "Scrape Reviews" section.
2. Enter the Amazon product review URL and the number of pages to scrape.
3. Click the "SCRAPE REVIEWS" button to start scraping. The resulting reviews will be displayed with sentiment analysis results.

### Uploading Dataset

1. Go to the "Upload Dataset" section.
2. Upload a CSV file that contains the necessary columns.
3. The uploaded data will be processed, and sentiment analysis will be performed.

### Analyzing Custom Text

1. Navigate to the "Text Analysis" section.
2. Input any text you wish to analyze.
3. Select the language for analysis (English or other) and click the "Analyze Text" button.

### Fake Review Detection

1. Visit the "Fake Review Detection" section.
2. Upload a CSV file with reviews to analyze their authenticity.
3. Results will display which reviews are classified as fake, real, or uncertain.

### History

- In the "History" section, you can view all past analyses and results, with options to clear databases.

## Development

### Directory Structure

```
sentimelyzer/
│
├── app.py                   # Main application file
├── requirements.txt          # Required libraries
└── sentimelyzer.png         # Application logo
```

### Database Initialization

The application initializes three SQLite databases to store scraped reviews, uploaded analyses, and outputs. Ensure the appropriate permissions are set to allow database reading/writing.

### Acknowledgments

- **NLTK & VADER:** For natural language processing and sentiment analysis.
- **BeautifulSoup:** For parsing HTML and web scraping.
- **Plotly & Matplotlib:** For generating visualizations.
- **WordCloud:** For creating word clouds based on text inputs.

## License

This project is licensed under the MIT License.

---

For questions or feedback, please reach out via the repository's issues page or through contact details provided in the project documentation. Happy coding!
