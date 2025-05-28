from IPython.display import display
import pandas as pd
import numpy as np
import re
import nltk
from datetime import datetime, timedelta
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from tqdm import tqdm

# Download NLTK resources
nltk.download(['stopwords', 'wordnet', 'omw-1.4', 'vader_lexicon'])
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# ====================== FINANCIAL RELEVANCE ======================
finrelevance_tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
finrelevance_model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

def is_finance_related(text, threshold=0.5):
    """Hybrid financial relevance detection with debugging"""
    try:
        # Expanded keyword check
        financial_keywords = {
            'stock', 'market', 'price', 'invest', 'portfolio', 'trade',
            'dollar', 'crypto', 'bitcoin', 'etf', 'bull', 'bear', 'dividend',
            'ipo', 'valuation', 'merger', 'acquisition', 'earnings',
            'update', 'pricing', 'subscription', 'revenue', 'profit'
        }
        keywords_found = financial_keywords & set(re.findall(r'\w+', text.lower()))
        if len(keywords_found) > 0:
            print(f"Keyword match for '{text[:50]}...': {keywords_found}")
            return True

        # Model-based check
        inputs = finrelevance_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        outputs = finrelevance_model(**inputs)
        prob = outputs.logits.softmax(dim=1)[0][0].item()
        if prob > threshold:
            print(f"FinBERT match for '{text[:50]}...': {prob}")
            return True
        else:
            print(f"Rejected '{text[:50]}...': FinBERT prob {prob} < {threshold}")
            return False
    except Exception as e:
        print(f"Error in is_finance_related for '{text[:61]}...': {e}")
        return False

# ====================== TEXT PROCESSING ======================
def preprocess_text(text, custom_lexicon=None):
    if pd.isna(text) or text == "[NON-FINANCIAL]":
        return ""

    text = str(text).lower()

    # Lexicon replacement
    if custom_lexicon:
        for term, replacement in custom_lexicon.items():
            text = re.sub(r'\b' + term + r'\b', replacement, text)

    # Clean text
    text = re.sub(r'@\w+|https?://\S+|www\.\S+', '', text)  # Remove mentions/URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special chars
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace

    # Lemmatization
    words = [lemmatizer.lemmatize(word, pos='v')
            for word in text.split() if word not in stop_words]

    return ' '.join(words)

# ====================== SENTIMENT ANALYSIS ======================
finbert_tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
finbert_model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
finbert = pipeline("text-classification",
                 model=finbert_model,
                 tokenizer=finbert_tokenizer,
                 truncation=True,
                 max_length=512)

def detect_financial_sarcasm(text):
    """Rule-based financial sarcasm detection"""
    text = text.lower()
    patterns = [
        r'(great|excellent|perfect).*\s(crash|drop|plummet|tank)',
        r'(congrats|congratulations).*\s(loss|fail|bankrupt)',
        r'\b(wow|awesome).*\s(dump|collapse)',
        r'(\b[A-Z]{4,}\b)',  # All-caps words
        r'\b\w+[!]{2,}\b'    # Excessive punctuation
    ]
    return sum(1 for p in patterns if re.search(p, text)) >= 2

def get_enhanced_sentiment(text, confidence_threshold=0.6):
    try:
        if text == "[NON-FINANCIAL]":
            return "Neutral", 0.0, False

        # Detect sarcasm
        is_sarcastic = detect_financial_sarcasm(text)

        # Get base sentiment
        result = finbert(text[:512])[0]
        label = result['label']
        confidence = result['score']

        # Handle sarcasm
        if is_sarcastic:
            label = 'Negative' if label == 'Positive' else 'Positive'
            confidence = min(confidence * 1.2, 1.0)

        # VADER validation
        sia = SentimentIntensityAnalyzer()
        vader_scores = sia.polarity_scores(text)
        if abs(vader_scores['compound']) > 0.7 and confidence < 0.4:
            label = 'Positive' if vader_scores['compound'] > 0 else 'Negative'

        # Market context override
        if 'short' in text.lower() and label == 'Positive':
            label = 'Negative'

        return label, confidence, is_sarcastic
    except Exception as e:
        print(f"Sentiment error for '{text[:50]}...': {e}")
        return "Neutral", 0.0, False

# ====================== TEMPORAL FEATURES ======================
def create_temporal_features(df, date_col='random_datetime', text_col='Post'):
    """Robust temporal feature creation with empty handling"""
    if df.empty:
        return pd.DataFrame()

    features = {
        'avg_sentiment': ('sentiment_score', 'mean'),
        'sentiment_volatility': ('sentiment_score', 'std'),
        'tweet_count': (text_col, 'count'),  # Use text_col instead of 'Text'
        'sarcasm_rate': ('is_sarcastic', 'mean')
    }

    try:
        if 'sentiment_score' not in df.columns:
            print(f"Warning: 'sentiment_score' not found in df. Columns available: {df.columns.tolist()}")
            df['sentiment_score'] = 0  # Fallback
        temporal_df = df.groupby(pd.Grouper(key=date_col, freq='D')).agg(**features)

        # Handle momentum with low data
        temporal_df['sentiment_momentum'] = temporal_df['avg_sentiment'].rolling(
            3, min_periods=1).mean()

        # Safe volatility ratio
        temporal_df['volatility_ratio'] = temporal_df['sentiment_volatility'] / \
            temporal_df['tweet_count'].replace(0, 1)

        return temporal_df.fillna(0)
    except Exception as e:
        print(f"Temporal feature error: {e}")
        return pd.DataFrame()

# ====================== VISUALIZATIONS ======================
def generate_visualizations(temporal_df, raw_df):
    """Error-proof visualizations for all data scenarios"""
    plt.figure(figsize=(15, 10))

    # 1. Financial Distribution
    plt.subplot(2, 2, 1)
    if 'is_financial' in raw_df.columns and not raw_df.empty:
        counts = raw_df['is_financial'].value_counts()
        labels = [l for l, p in zip(['Financial', 'Non-Financial'], [True, False]) if p in counts.index]
        plt.pie(counts, labels=labels, autopct='%1.1f%%', colors=['#4CAF50', '#F44336'])
        plt.title('Tweet Distribution')
    else:
        plt.text(0.5, 0.5, 'No Financial Data', ha='center', va='center')

    # 2. Sentiment Analysis
    plt.subplot(2, 2, 2)
    if 'sentiment' in raw_df.columns and not raw_df.empty:
        counts = raw_df['sentiment'].value_counts()
        colors = {'Positive': '#4CAF50', 'Negative': '#F44336', 'Neutral': '#9E9E9E'}
        present = [s for s in colors if s in counts.index]
        plt.pie(counts[present], labels=present, autopct='%1.1f%%', colors=[colors[s] for s in present])
        plt.title('Sentiment Distribution')
    else:
        plt.text(0.5, 0.5, 'No Sentiment Data', ha='center', va='center')

    # 3. Temporal Trends
    plt.subplot(2, 2, 3)
    if not temporal_df.empty:
        sns.lineplot(x=temporal_df.index, y='avg_sentiment', data=temporal_df, label='Daily')
        sns.lineplot(x=temporal_df.index, y='sentiment_momentum', data=temporal_df, label='3D Momentum')
        plt.title('Sentiment Trends')
        plt.xticks(rotation=45)
    else:
        plt.text(0.5, 0.5, 'No Temporal Data', ha='center', va='center')

    # 4. Word Cloud
    plt.subplot(2, 2, 4)
    if not raw_df.empty and 'cleaned_text' in raw_df.columns:
        text = ' '.join(raw_df['cleaned_text'])
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Key Terms')
    else:
        plt.text(0.5, 0.5, 'No Text Data', ha='center', va='center')

    plt.tight_layout()
    plt.show()

# ====================== MAIN ANALYSIS ======================
def analyze_twitter_data(df, text_col='Post', date_col='Window_End_Date', target_date='2025-04-20', custom_lexicon=None, confidence_threshold=0.6):
    """Robust analysis pipeline handling all data scenarios"""

    # Validate input DataFrame
    if df.empty:
        print("Input DataFrame is empty")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # Skip financial filtering and date filtering as per executive decision
    financial_df = df.copy()  # Use all tweets

    # Date range after coercion (for info only, no filtering)
    financial_df[date_col] = pd.to_datetime(financial_df[date_col], errors='coerce')
    financial_df = financial_df.dropna(subset=[date_col])
    if financial_df.empty:
        print("No valid dates after coercion")
        return df, pd.DataFrame(), pd.DataFrame()
    print(f"Date range after coercion: {financial_df[date_col].min()} to {financial_df[date_col].max()}")

    # Sentiment analysis
    financial_df['cleaned_text'] = financial_df[text_col].apply(
        lambda x: preprocess_text(x, custom_lexicon=custom_lexicon))

    tqdm.pandas(desc="Sentiment Analysis")
    sentiment_results = financial_df['cleaned_text'].progress_apply(
        lambda x: pd.Series(get_enhanced_sentiment(x, confidence_threshold)))
    financial_df[['sentiment', 'confidence', 'is_sarcastic']] = sentiment_results
    print(f"Sentiment values: {financial_df['sentiment'].unique()}")  # Debugging

    # Numeric mapping with error handling
    sentiment_map = {'Positive': 1, 'Negative': -1, 'Neutral': 0}
    try:
        financial_df['sentiment_score'] = financial_df['sentiment'].map(sentiment_map)
        print(f"Sentiment score mapping successful. Sample: {financial_df['sentiment_score'].head().to_dict()}")
    except KeyError as e:
        print(f"Error mapping sentiment to score: {e}. Checking sentiment values: {financial_df['sentiment'].unique()}")
        financial_df['sentiment_score'] = 0

    financial_df['confidence'] = financial_df['confidence'].fillna(0)

    # Temporal features
    try:
        temporal_df = create_temporal_features(financial_df, date_col=date_col, text_col=text_col) if not financial_df.empty else pd.DataFrame()
    except KeyError as e:
        print(f"Temporal feature error: {e}. Using financial_df as fallback.")
        temporal_df = financial_df[[date_col, 'sentiment_score']].copy()
    except Exception as e:
        print(f"Temporal feature error: {e}. Using financial_df as fallback.")
        temporal_df = financial_df[[date_col, 'sentiment_score']].copy()
    rl_output = temporal_df.reset_index() if not temporal_df.empty else pd.DataFrame()

    # Ensure rl_output has required columns
    if not rl_output.empty and date_col in rl_output.columns:
        rl_output = rl_output.rename(columns={date_col: 'random_datetime'})
        if 'sentiment_score' in rl_output.columns:
            rl_output['avg_sentiment'] = rl_output['sentiment_score']
        else:
            rl_output['avg_sentiment'] = 0  # Fallback
    else:
        rl_output = pd.DataFrame(columns=['random_datetime', 'avg_sentiment'])  # Ensure columns exist

    return df, temporal_df, rl_output

# ====================== EXECUTION ======================
if __name__ == "__main__":
    financial_lexicon = {
        'bullish': 'very_positive',
        'bearish': 'very_negative',
        'short': 'negative_position',
        'long': 'positive_position'
    }

    user_date = input("Enter target date (YYYY-MM-DD): ")
    raw_df, temporal_df, rl_df = analyze_twitter_data(
        '/content/stock_data_with_dates.csv',
        custom_lexicon=financial_lexicon,
        confidence_threshold=0.65,
        target_date=user_date
    )

    print("\nRL Input Features:")
    display(rl_df.tail() if not rl_df.empty else "No RL features generated")

    print("\nTemporal Statistics:")
    display(temporal_df.describe() if not temporal_df.empty else "No temporal data")

    generate_visualizations(temporal_df, raw_df)