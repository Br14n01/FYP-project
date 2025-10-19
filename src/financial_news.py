import finnhub

def get_news(api_key, ticker='AAPL'):
    

    finnhub_client = finnhub.Client(api_key=api_key)

    company_news = finnhub_client.company_news(ticker, 
                                            _from="2025-10-15"
                                            ,to="2025-10-18"
                                            )

    return company_news[1]['summary']
