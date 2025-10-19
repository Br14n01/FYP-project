from dotenv import load_dotenv
import os
from src.financial_news import get_news_finnhub
from src.news_sentimental_analysis import scoring

def main():
    load_dotenv()
    api_key = os.getenv("FINNHUB_API_KEY")

    ticker = input("Enter the company ticker: ")

    news = get_news_finnhub(api_key, ticker)

    result = scoring(news)

    print(f"News summary:\n{news}")
    print(f"label: {result[0]['label']}")
    print(f"score: {result[0]['score']}")

if __name__ == "__main__":
    main()