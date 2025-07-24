"""
Sentiment Analysis Module for Financial Text Processing.
This module provides enhanced sentiment analysis specifically designed for financial content.
"""

import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta
from typing import Tuple, Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Try to import transformers and NLTK
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.sentiment import SentimentIntensityAnalyzer
    NLTK_AVAILABLE = True
    
    # Download NLTK resources quietly
    try:
        nltk.download(['stopwords', 'wordnet', 'omw-1.4', 'vader_lexicon'], quiet=True)
    except:
        pass
except ImportError:
    NLTK_AVAILABLE = False

# Import simple sentiment analyzer as fallback
try:
    from .simple_sentiment import SimpleSentimentAnalyzer
except ImportError:
    try:
        from simple_sentiment import SimpleSentimentAnalyzer
    except ImportError:
        SimpleSentimentAnalyzer = None


class FinancialSentimentAnalyzer:
    """Enhanced sentiment analyzer for financial text with robust fallbacks."""
    
    def __init__(self, confidence_threshold: float = 0.6):
        self.confidence_threshold = confidence_threshold
        self.use_simple_analyzer = False
        self.simple_analyzer = None
        
        # Initialize components based on availability
        if NLTK_AVAILABLE:
            try:
                self.stop_words = set(stopwords.words('english'))
                self.lemmatizer = WordNetLemmatizer()
            except:
                self.stop_words = set()
                self.lemmatizer = None
        else:
            self.stop_words = set()
            self.lemmatizer = None
        
        # Initialize models
        self._init_models()
        
        # Financial keywords for relevance detection
        self.financial_keywords = {
            'stock', 'market', 'price', 'invest', 'portfolio', 'trade',
            'dollar', 'crypto', 'bitcoin', 'etf', 'bull', 'bear', 'dividend',
            'ipo', 'valuation', 'merger', 'acquisition', 'earnings',
            'update', 'pricing', 'subscription', 'revenue', 'profit',
            'finance', 'financial', 'economy', 'economic', 'nasdaq',
            'dow', 'sp500', 'futures', 'options', 'bond', 'yield'
        }
    
    def _init_models(self):
        """Initialize the required models with fallbacks."""
        if not TRANSFORMERS_AVAILABLE and not NLTK_AVAILABLE:
            print("⚠️  Neither transformers nor NLTK available, using simple analyzer")
            self._init_simple_analyzer()
            return
        
        try:
            if TRANSFORMERS_AVAILABLE:
                # Try to initialize FinBERT for sentiment
                print("🔄 Attempting to load transformers models...")
                self.finbert_tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
                self.finbert_model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
                self.finbert = pipeline("text-classification",
                                      model=self.finbert_model,
                                      tokenizer=self.finbert_tokenizer,
                                      truncation=True,
                                      max_length=512)
                
                # Financial relevance model
                self.finrelevance_tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
                self.finrelevance_model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
                print("✅ Transformers models loaded successfully")
            else:
                self.finbert = None
                self.finrelevance_model = None
            
            # VADER sentiment analyzer
            if NLTK_AVAILABLE:
                self.vader_analyzer = SentimentIntensityAnalyzer()
            else:
                self.vader_analyzer = None
                
        except Exception as e:
            print(f"❌ Could not initialize transformers models: {e}")
            print("🔄 Falling back to simple sentiment analyzer...")
            self._init_simple_analyzer()
    
    def _init_simple_analyzer(self):
        """Initialize simple sentiment analyzer as fallback."""
        if SimpleSentimentAnalyzer is not None:
            self.simple_analyzer = SimpleSentimentAnalyzer()
            self.use_simple_analyzer = True
            print("✅ Simple sentiment analyzer initialized")
        else:
            print("❌ No sentiment analysis capability available")
            # Create minimal functionality
            self.use_simple_analyzer = True
            self.simple_analyzer = None
    
    def is_finance_related(self, text: str, threshold: float = 0.5) -> bool:
        """
        Determine if text is finance-related using hybrid approach.
        
        Args:
            text: Input text
            threshold: Confidence threshold for model-based detection
            
        Returns:
            Boolean indicating if text is finance-related
        """
        try:
            # Quick keyword check
            keywords_found = self.financial_keywords & set(re.findall(r'\w+', text.lower()))
            if len(keywords_found) > 0:
                return True
            
            # Model-based check if available
            if self.finrelevance_model:
                inputs = self.finrelevance_tokenizer(text, return_tensors="pt", 
                                                   truncation=True, max_length=512)
                outputs = self.finrelevance_model(**inputs)
                prob = outputs.logits.softmax(dim=1)[0][0].item()
                return prob > threshold
            
            return False
            
        except Exception:
            return len(keywords_found) > 0 if 'keywords_found' in locals() else False
    
    def preprocess_text(self, text: str, custom_lexicon: Optional[Dict] = None) -> str:
        """
        Preprocess text for sentiment analysis.
        
        Args:
            text: Input text
            custom_lexicon: Optional dictionary for term replacement
            
        Returns:
            Preprocessed text
        """
        if pd.isna(text) or text == "[NON-FINANCIAL]":
            return ""
        
        text = str(text).lower()
        
        # Apply custom lexicon
        if custom_lexicon:
            for term, replacement in custom_lexicon.items():
                text = re.sub(r'\\b' + term + r'\\b', replacement, text)
        
        # Clean text
        text = re.sub(r'@\\w+|https?://\\S+|www\\.\\S+', '', text)  # Remove mentions/URLs
        text = re.sub(r'[^a-zA-Z\\s]', '', text)  # Remove special chars
        text = re.sub(r'\\s+', ' ', text).strip()  # Remove extra whitespace
        
        # Lemmatization and stop word removal
        if self.lemmatizer:
            words = [self.lemmatizer.lemmatize(word, pos='v')
                    for word in text.split() if word not in self.stop_words]
        else:
            words = [word for word in text.split() if word not in self.stop_words]
        
        return ' '.join(words)
    
    def analyze_sentiment(self, text: str) -> float:
        """
        Main sentiment analysis method with fallback support.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Sentiment score between -1 and 1
        """
        if not text or not isinstance(text, str):
            return 0.0
        
        # Use simple analyzer if models failed to load
        if self.use_simple_analyzer:
            if self.simple_analyzer:
                return self.simple_analyzer.analyze_sentiment(text)
            else:
                # Minimal fallback - count positive/negative words
                positive_words = ['good', 'great', 'excellent', 'positive', 'bullish', 'growth']
                negative_words = ['bad', 'poor', 'terrible', 'negative', 'bearish', 'decline']
                
                text_lower = text.lower()
                pos_count = sum(1 for word in positive_words if word in text_lower)
                neg_count = sum(1 for word in negative_words if word in text_lower)
                
                if pos_count > neg_count:
                    return 0.5
                elif neg_count > pos_count:
                    return -0.5
                else:
                    return 0.0
        
        # Use advanced models if available
        try:
            if self.finbert:
                result = self.finbert(text[:512])[0]
                label = result['label'].lower()
                confidence = result['score']
                
                if 'positive' in label:
                    return confidence
                elif 'negative' in label:
                    return -confidence
                else:
                    return 0.0
            elif self.vader_analyzer:
                scores = self.vader_analyzer.polarity_scores(text)
                return scores['compound']
            else:
                return 0.0
                
        except Exception as e:
            print(f"⚠️  Sentiment analysis error: {e}")
            return 0.0
    
    def detect_financial_sarcasm(self, text: str) -> bool:
        """
        Detect sarcasm in financial text using rule-based patterns.
        
        Args:
            text: Input text
            
        Returns:
            Boolean indicating if sarcasm is detected
        """
        text = text.lower()
        patterns = [
            r'(great|excellent|perfect).*\\s(crash|drop|plummet|tank)',
            r'(congrats|congratulations).*\\s(loss|fail|bankrupt)',
            r'\\b(wow|awesome).*\\s(dump|collapse)',
            r'(\\b[A-Z]{4,}\\b)',  # All-caps words
            r'\\b\\w+[!]{2,}\\b'    # Excessive punctuation
        ]
        return sum(1 for p in patterns if re.search(p, text)) >= 2
    
    def get_enhanced_sentiment(self, text: str) -> Tuple[str, float, bool]:
        """
        Get enhanced sentiment analysis for financial text.
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (sentiment_label, confidence, is_sarcastic)
        """
        try:
            if text == "[NON-FINANCIAL]" or not text.strip():
                return "Neutral", 0.0, False
            
            # Detect sarcasm
            is_sarcastic = self.detect_financial_sarcasm(text)
            
            # Get base sentiment from FinBERT if available
            if self.finbert:
                result = self.finbert(text[:512])[0]
                label = result['label']
                confidence = result['score']
            else:
                # Fallback to VADER
                vader_scores = self.vader_analyzer.polarity_scores(text)
                compound = vader_scores['compound']
                if compound > 0.05:
                    label, confidence = 'Positive', abs(compound)
                elif compound < -0.05:
                    label, confidence = 'Negative', abs(compound)
                else:
                    label, confidence = 'Neutral', 1 - abs(compound)
            
            # Handle sarcasm
            if is_sarcastic:
                label = 'Negative' if label == 'Positive' else 'Positive'
                confidence = min(confidence * 1.2, 1.0)
            
            # VADER validation if FinBERT is available
            if self.finbert and confidence < 0.4:
                vader_scores = self.vader_analyzer.polarity_scores(text)
                if abs(vader_scores['compound']) > 0.7:
                    label = 'Positive' if vader_scores['compound'] > 0 else 'Negative'
            
            # Market context override
            if 'short' in text.lower() and label == 'Positive':
                label = 'Negative'
            
            return label, confidence, is_sarcastic
            
        except Exception as e:
            return "Neutral", 0.0, False
    
    def create_temporal_features(self, df: pd.DataFrame, date_col: str = 'date', 
                               text_col: str = 'text') -> pd.DataFrame:
        """
        Create temporal sentiment features.
        
        Args:
            df: DataFrame with text data
            date_col: Name of date column
            text_col: Name of text column
            
        Returns:
            DataFrame with temporal features
        """
        if df.empty:
            return pd.DataFrame()
        
        features = {
            'avg_sentiment': ('sentiment_score', 'mean'),
            'sentiment_volatility': ('sentiment_score', 'std'),
            'text_count': (text_col, 'count'),
            'sarcasm_rate': ('is_sarcastic', 'mean')
        }
        
        try:
            temporal_df = df.groupby(pd.Grouper(key=date_col, freq='D')).agg(**features)
            
            # Add momentum features
            temporal_df['sentiment_momentum'] = temporal_df['avg_sentiment'].rolling(
                3, min_periods=1).mean()
            
            # Volatility ratio
            temporal_df['volatility_ratio'] = temporal_df['sentiment_volatility'] / \
                temporal_df['text_count'].replace(0, 1)
            
            return temporal_df.fillna(0)
            
        except Exception as e:
            print(f"Temporal feature error: {e}")
            return pd.DataFrame()
    
    def analyze_sentiment_batch(self, texts: List[str], 
                              custom_lexicon: Optional[Dict] = None) -> pd.DataFrame:
        """
        Analyze sentiment for a batch of texts.
        
        Args:
            texts: List of text strings
            custom_lexicon: Optional custom lexicon for preprocessing
            
        Returns:
            DataFrame with sentiment analysis results
        """
        results = []
        
        for text in texts:  # Remove tqdm dependency
            # Preprocess
            cleaned_text = self.preprocess_text(text, custom_lexicon)
            
            # Check financial relevance
            is_financial = self.is_finance_related(text)
            
            if is_financial:
                # Get sentiment
                sentiment, confidence, is_sarcastic = self.get_enhanced_sentiment(cleaned_text)
                
                # Convert to numeric score
                sentiment_map = {'Positive': 1, 'Negative': -1, 'Neutral': 0}
                sentiment_score = sentiment_map.get(sentiment, 0)
            else:
                sentiment, confidence, is_sarcastic = "Neutral", 0.0, False
                sentiment_score = 0
            
            results.append({
                'original_text': text,
                'cleaned_text': cleaned_text,
                'is_financial': is_financial,
                'sentiment': sentiment,
                'sentiment_score': sentiment_score,
                'confidence': confidence,
                'is_sarcastic': is_sarcastic
            })
        
        return pd.DataFrame(results)


def analyze_text_sentiment(texts: List[str], custom_lexicon: Optional[Dict] = None) -> pd.DataFrame:
    """
    Convenience function for sentiment analysis.
    
    Args:
        texts: List of text strings to analyze
        custom_lexicon: Optional custom lexicon
        
    Returns:
        DataFrame with sentiment analysis results
    """
    analyzer = FinancialSentimentAnalyzer()
    return analyzer.analyze_sentiment_batch(texts, custom_lexicon)


# Compatibility class for main usage
class SentimentAnalyzer:
    """Simple compatibility wrapper for the sentiment analyzer."""
    
    def __init__(self):
        """Initialize the sentiment analyzer."""
        self.analyzer = FinancialSentimentAnalyzer()
    
    def analyze_sentiment(self, text: str) -> float:
        """
        Analyze sentiment of text.
        
        Args:
            text: Input text
            
        Returns:
            Sentiment score between -1 and 1
        """
        return self.analyzer.analyze_sentiment(text)
