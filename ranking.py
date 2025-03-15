import pandas as pd
from model import classify_news
import nltk  # Ensure this import is at the top
from nltk.sentiment import SentimentIntensityAnalyzer

# Ensure vader_lexicon is downloaded
nltk.download('vader_lexicon')

# Initialize Sentiment Intensity Analyzer
sia = SentimentIntensityAnalyzer()

# Sentiment impact weights
sentiment_weights = {"NEGATIVE": -1, "NEUTRAL": 0, "POSITIVE": 1}

def rank_news(news_list):
    """Ranks news based on sentiment, impact score, and compound sentiment analysis."""
    data = []

    for news in news_list:
        # Get predicted sentiment and impact score
        result = classify_news([news])
        print("classify_news output:", result)  # Debugging
        
        # Ensure the output is extracted properly
        if isinstance(result, list) and len(result) > 0:
            result = result[0]  # Extract first element if it's a list

        if isinstance(result, tuple) and len(result) == 2:
            sentiment, impact_score = result
        else:
            sentiment = "NEUTRAL"  # Default sentiment
            impact_score = 0  # Default impact score

        # Apply sentiment weighting
        sentiment_weight = sentiment_weights.get(sentiment, 0)
        
        # Calculate initial importance score
        investment_importance = sentiment_weight * impact_score

        # Perform sentiment intensity analysis
        sentiment_scores = sia.polarity_scores(news)
        
        # Calculate the final rank score with compound sentiment
        rank_score = (
            (sentiment_scores['pos'] * 2) +
            (sentiment_scores['neu'] * 1) -
            (sentiment_scores['neg'] * 3) +
            (sentiment_scores['compound'] * 5) + 
            investment_importance
        )

        data.append({
            "news": news,
            "sentiment": sentiment,
            "impact_score": impact_score,
            "importance": investment_importance,
            "rank_score": rank_score
        })

    # Sort by rank score
    ranked_df = pd.DataFrame(data).sort_values(by="rank_score", ascending=False)
    return ranked_df.to_dict(orient="records")

# Example Usage
news_articles = ["Stock market surges after positive earnings.", "Recession fears grow due to inflation."]
ranked_news = rank_news(news_articles)
print(ranked_news)

