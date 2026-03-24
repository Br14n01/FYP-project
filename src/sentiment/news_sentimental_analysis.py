"""
FinBERT sentiment scoring with singleton model loading and batch support.

Usage:
    from src.sentiment.news_sentimental_analysis import SentimentScorer
    scorer = SentimentScorer()
    result  = scorer.score("Apple beats earnings expectations")
    results = scorer.score_batch(["headline1", "headline2", ...])
"""

import os
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline


class SentimentScorer:
    """Loads ProsusAI/finbert once, reuses for all scoring calls."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        print("Loading FinBERT model (one-time) ...")
        tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        self._pipeline = pipeline(
            "sentiment-analysis", model=model, tokenizer=tokenizer
        )
        self._initialized = True

    def score(self, text: str) -> dict:
        """
        Score a single text.

        Returns dict with keys: label, score.
        """
        result = self._pipeline(text, truncation=True)[0]
        return {"label": result["label"], "score": result["score"]}

    def score_batch(self, texts: list[str], batch_size: int = 32) -> list[dict]:
        """
        Score a list of texts in batches.

        Returns list of dicts, each with keys: label, score.
        """
        results = self._pipeline(texts, truncation=True, batch_size=batch_size)
        return [{"label": r["label"], "score": r["score"]} for r in results]


def scoring(text):
    """Legacy API: kept for backward compatibility with main.py."""
    scorer = SentimentScorer()
    r = scorer.score(text)
    return [r]


def score_news_csv(
    news_csv_path: str,
    output_path: str | None = None,
    headline_col: str = "headline",
) -> pd.DataFrame:
    """
    Read a news CSV (from finnhub_news), score every headline with FinBERT,
    and save the results.

    Parameters
    ----------
    news_csv_path : str
        Path to the CSV containing at least columns: date, headline.
    output_path : str or None
        Where to write the scored CSV. If None, derived from news_csv_path.
    headline_col : str
        Column name containing the text to score.

    Returns
    -------
    pd.DataFrame
        Original data plus 'sentiment_label' and 'sentiment_score' columns.
    """
    df = pd.read_csv(news_csv_path)
    if df.empty:
        print(f"No data in {news_csv_path}")
        return df

    headlines = df[headline_col].fillna("").tolist()

    scorer = SentimentScorer()
    print(f"Scoring {len(headlines)} headlines ...")
    scored = scorer.score_batch(headlines)

    df["sentiment_label"] = [s["label"] for s in scored]
    df["sentiment_score"] = [s["score"] for s in scored]

    # Convert to a numeric sentiment value: positive=+score, negative=-score, neutral=0
    def _to_numeric(row):
        if row["sentiment_label"] == "positive":
            return row["sentiment_score"]
        elif row["sentiment_label"] == "negative":
            return -row["sentiment_score"]
        return 0.0

    df["sentiment_numeric"] = df.apply(_to_numeric, axis=1)

    if output_path is None:
        output_path = news_csv_path.replace("_news.csv", "_sentiment.csv")
    df.to_csv(output_path, index=False)
    print(f"Saved scored data -> {output_path}")

    return df
