import finnhub
import feedparser 
from urllib.parse import urlparse, parse_qs, quote
import requests
from bs4 import BeautifulSoup

def get_news_finnhub(api_key, ticker='AAPL'):
    finnhub_client = finnhub.Client(api_key=api_key)

    company_news = finnhub_client.company_news(ticker, 
                                            _from="2025-10-15"
                                            ,to="2025-10-18"
                                            )

    return company_news[0]['summary']

def fetch_article_content(url):
    """
    Fetch article text from the resolved publisher URL.
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # Grab all paragraph text
        paragraphs = soup.find_all('p')
        content = ' '.join([p.get_text() for p in paragraphs])
        return content.strip() if content else "No readable content found."
    except requests.RequestException:
        return "Content not retrieved."

def fetch_news(query, num_articles=10):
    """
    Retrieve recent news articles from Google News RSS for a given search query.

    Parameters
    ----------
    query : str
        The search term to query in Google News.
    num_articles : int, optional (default=10)
        The maximum number of articles to retrieve.
    
    Returns
    -------
    article : list of dict
        A list of article metadata, where each dictionary has the keys:
        - "title" (str): The headline of the article.
        - "link" (str): The URL to the article.
        - "published" (str): The publication date as provided by the feed.
    """
    rss_url = f"https://news.google.com/rss/search?q={quote(query)}"
    feed = feedparser.parse(rss_url)
    news_items = feed.entries[:num_articles]

    articles = []
    for item in news_items:
        title = item.title
        link = item.link
        published = item.published
        content = fetch_article_content(link)
        
        articles.append({
            "title": title,
            "link": link,
            "published": published,
            "content": content
        })

    return articles

