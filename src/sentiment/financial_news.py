import feedparser 
from urllib.parse import urlparse, parse_qs, quote
import requests
from bs4 import BeautifulSoup
import json

def resolve_google_news_url(google_url):
    resp = requests.get(google_url)
    data = BeautifulSoup(resp.text, 'html.parser').select_one('c-wiz[data-p]').get('data-p')
    obj = json.loads(data.replace('%.@.', '["garturlreq",'))

    payload = {
        'f.req': json.dumps([[['Fbv4je', json.dumps(obj[:-6] + obj[-2:]), 'null', 'generic']]])
    }

    headers = {
    'content-type': 'application/x-www-form-urlencoded;charset=UTF-8',
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36',
    }

    url = "https://news.google.com/_/DotsSplashUi/data/batchexecute"
    response = requests.post(url, headers=headers, data=payload)
    array_string = json.loads(response.text.replace(")]}'", ""))[0][2]
    article_url = json.loads(array_string)[1]
    return article_url

def fetch_article_content(url):
    try:
        response = requests.get(url, timeout=10,
                                headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # Grab all paragraph text
        paragraphs = soup.find_all('p')
        content = ' '.join([p.get_text() for p in paragraphs])
        # Remove common boilerplate phrases
        boilerplate_phrases = [
            "Oops, something went wrong",
            "Something went wrong"
        ]
        for phrase in boilerplate_phrases:
            content = content.replace(phrase, "")
        return content.strip() if content else "No readable content found."
    except requests.RequestException:
        return "Content not retrieved."

def fetch_news(query, num_articles=5):
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
        link = resolve_google_news_url(item.link)
        published = item.published
        content = fetch_article_content(link)
        articles.append({
            "title": title,
            "link": link,
            "published": published,
            "content": content
        })

    return articles

