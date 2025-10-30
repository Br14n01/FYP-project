from dotenv import load_dotenv
import os
from src.financial_news import *
from src.news_sentimental_analysis import scoring

def main():

    queries = [
        "AAPL"
    ]
    articles_per_query = 3
    all_articles = []

    for query in queries:
        print("Fetching news from query...")
        articles = fetch_news(query=query, num_articles=articles_per_query)
        all_articles.extend(articles)

    for idx, article in enumerate(all_articles, 1):
        print(f"******* Article {idx}: {article['title']} *******")
        print(f"Article {idx}: {article['title']}")
        # print(f"Link: {article['link']}")
        print(f"Published: {article['published']}")
        print(f"Content: {article['content'][:100]}...")

        result = scoring(article['title'])
        print(f"Label: {result[0]['label']}")
        print(f"Score: {result[0]['score']}")

if __name__ == "__main__":
    main()