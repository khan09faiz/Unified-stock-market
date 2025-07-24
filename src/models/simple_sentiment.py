"""
Simple offline sentiment analyzer for demonstration when transformers models can't be loaded.
"""

import re
import string
from typing import Dict, List
import numpy as np


class SimpleSentimentAnalyzer:
    """Simple rule-based sentiment analyzer for offline use."""
    
    def __init__(self):
        """Initialize with predefined word lists."""
        # Positive financial terms
        self.positive_words = {
            'excellent', 'outstanding', 'strong', 'growth', 'profit', 'bullish', 'rally',
            'surge', 'gains', 'beat', 'exceeded', 'optimistic', 'positive', 'upgraded',
            'buy', 'overweight', 'outperform', 'breakthrough', 'success', 'expansion',
            'revenue', 'earnings', 'dividend', 'bonus', 'acquisition', 'merger',
            'innovation', 'launch', 'partnership', 'agreement', 'approval', 'recovery'
        }
        
        # Negative financial terms
        self.negative_words = {
            'decline', 'loss', 'bearish', 'crash', 'fall', 'drop', 'plunge',
            'weak', 'poor', 'disappointing', 'missed', 'below', 'downgrade',
            'sell', 'underweight', 'underperform', 'concern', 'risk', 'uncertainty',
            'volatile', 'pressure', 'challenge', 'problem', 'debt', 'lawsuit',
            'investigation', 'scandal', 'bankruptcy', 'layoffs', 'closure', 'recession'
        }
        
        # Intensity modifiers
        self.intensifiers = {
            'very': 1.5, 'extremely': 2.0, 'highly': 1.5, 'significantly': 1.5,
            'substantially': 1.5, 'dramatically': 2.0, 'slightly': 0.5, 'somewhat': 0.7
        }
        
        # Negation words
        self.negations = {'not', 'no', 'never', 'nothing', 'neither', 'nor', 'none'}
    
    def preprocess_text(self, text: str) -> List[str]:
        """
        Preprocess text for analysis.
        
        Args:
            text: Input text
            
        Returns:
            List of cleaned tokens
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation except negation-related
        text = re.sub(r'[^\w\s\']', ' ', text)
        
        # Split into words
        words = text.split()
        
        # Remove extra whitespace
        words = [word.strip() for word in words if word.strip()]
        
        return words
    
    def analyze_sentiment(self, text: str) -> float:
        """
        Analyze sentiment of text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Sentiment score between -1 and 1
        """
        if not text or not isinstance(text, str):
            return 0.0
        
        words = self.preprocess_text(text)
        
        if not words:
            return 0.0
        
        total_score = 0.0
        word_count = 0
        
        i = 0
        while i < len(words):
            word = words[i]
            score = 0.0
            intensity = 1.0
            negated = False
            
            # Check for intensifiers before the word
            if i > 0 and words[i-1] in self.intensifiers:
                intensity = self.intensifiers[words[i-1]]
            
            # Check for negations before the word (within 2-3 words)
            for j in range(max(0, i-3), i):
                if words[j] in self.negations:
                    negated = True
                    break
            
            # Calculate base sentiment score
            if word in self.positive_words:
                score = 1.0
            elif word in self.negative_words:
                score = -1.0
            
            # Apply intensity and negation
            if score != 0:
                score *= intensity
                if negated:
                    score *= -1
                
                total_score += score
                word_count += 1
            
            i += 1
        
        # Calculate average sentiment
        if word_count == 0:
            return 0.0
        
        avg_sentiment = total_score / word_count
        
        # Normalize to [-1, 1] range
        return max(-1.0, min(1.0, avg_sentiment))
    
    def analyze_multiple_texts(self, texts: List[str]) -> Dict:
        """
        Analyze sentiment for multiple texts.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            Dictionary with analysis results
        """
        if not texts:
            return {'sentiment_score': 0.0, 'sentiment_scores': [], 'analysis_count': 0}
        
        scores = []
        for text in texts:
            score = self.analyze_sentiment(text)
            scores.append(score)
        
        avg_score = np.mean(scores) if scores else 0.0
        
        return {
            'sentiment_score': float(avg_score),
            'sentiment_scores': scores,
            'analysis_count': len(scores),
            'model_used': 'simple_rule_based'
        }


def create_sample_financial_texts() -> List[str]:
    """Create sample financial news texts for testing."""
    return [
        "Company reports strong quarterly earnings beating analyst expectations by 15%",
        "Stock price surges after successful product launch and positive market reception",
        "Quarterly revenue declined due to challenging market conditions and increased competition",
        "New partnership agreement expected to drive significant growth in the coming quarters",
        "Regulatory concerns weigh on sector performance amid uncertainty about future policies"
    ]


def test_simple_sentiment_analyzer():
    """Test the simple sentiment analyzer."""
    analyzer = SimpleSentimentAnalyzer()
    test_texts = create_sample_financial_texts()
    
    print("Testing Simple Sentiment Analyzer:")
    print("=" * 50)
    
    for i, text in enumerate(test_texts, 1):
        score = analyzer.analyze_sentiment(text)
        sentiment_label = "Positive" if score > 0.1 else "Negative" if score < -0.1 else "Neutral"
        
        print(f"\nText {i}: {text[:60]}...")
        print(f"Score: {score:.3f} ({sentiment_label})")
    
    # Test multiple texts
    results = analyzer.analyze_multiple_texts(test_texts)
    print(f"\nOverall Analysis:")
    print(f"Average Sentiment: {results['sentiment_score']:.3f}")
    print(f"Texts Analyzed: {results['analysis_count']}")


if __name__ == "__main__":
    test_simple_sentiment_analyzer()
