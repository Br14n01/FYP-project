import finnhub
import yfinance as yf

def get_news_finnhub(api_key, ticker='AAPL'):
    finnhub_client = finnhub.Client(api_key=api_key)

    company_news = finnhub_client.company_news(ticker, 
                                            _from="2025-10-15"
                                            ,to="2025-10-18"
                                            )

    return company_news[0]['summary']

def get_news_yf(ticker='AAPL'):
    news = yf.Search(ticker, news_count=10).news

    return news