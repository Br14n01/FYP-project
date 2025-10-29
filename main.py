from dotenv import load_dotenv
import os
from src.financial_news import *
from src.news_sentimental_analysis import scoring

def main():

    queries = [
        "Apple Inc",
        "AAPL"
    ]
    articles_per_query = 5
    articles = []

    for query in queries:
        print("Fetching news from query...")
        article = fetch_news(query=query, num_articles=articles_per_query)
        articles.extend(article)

    for idx, article in enumerate(articles, 1):
        print(f"Article {idx}: {article['title']}")
        print(f"Link: {article['link']}")
        print(f"Published: {article['published']}")

        result = scoring(article['title'])
        print(f"Label: {result[0]['label']}")

    # finnhub
    # load_dotenv()
    # api_key = os.getenv("FINNHUB_API_KEY")
    # ticker = input("Enter the company ticker: ")
    # news = get_news_finnhub(api_key, ticker)
    # result = scoring(news)

    # print(f"News summary:\n{news}")
    # print(f"label: {result[0]['label']}")
    # print(f"score: {result[0]['score']}")

if __name__ == "__main__":
    main()