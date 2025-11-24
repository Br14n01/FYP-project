# Hybrid Sentiment and Technical Indicator-Based Trading Strategy Using Machine Learning 

This is a Final Year Project for CUHK Computer Science and Engineering Undergraduate

## Introduction

This project aims to design and implement a machine learning-based trading system that integrates 
both market sentiment analysis and technical indicators to generate more accurate and reliable trading signals. 

## Approach

We sorted a list of securities and stocks in different fields for the system to track on.

| Symbol | Name                          | Sector                |
|--------|-------------------------------|-----------------------|
| VOO    | Vanguard S&P 500 ETF          | Exchange-Traded Fund  |
| QQQ    | Invesco QQQ Trust             | Exchange-Traded Fund  |
| AAPL   | Apple Inc.                    | Technology            |
| MSFT   | Microsoft Corporation         | Technology            |
| NVDA   | NVIDIA Corporation            | Technology            |
| TSLA   | Tesla Inc.                   | Technology            |
| AMZN   | Amazon.com Inc.               | Technology / E-commerce|
| JPM    | JPMorgan Chase & Co.          | Financials            |
| BAC    | Bank of America Corporation   | Financials            |
| GS     | Goldman Sachs Group Inc.      | Financials            |
| CVX    | Chevron Corporation           | Energy                |
| OXY    | Occidental Petroleum Corporation | Energy            |
| UNH    | UnitedHealth Group Incorporated | Healthcare          |
| JNJ    | Johnson & Johnson             | Healthcare            |
| PG     | Procter & Gamble Company      | Consumer Staples      |
| KO     | Coca-Cola Company             | Consumer Staples      |
| CAT    | Caterpillar Inc.              | Industrials           |
| T      | AT&T Inc.                    | Industrials           |
| BTC    | Bitcoin                      | Cryptocurrency        |
| ETH    | Ethereum                    | Cryptocurrency        |

### Data Collection

1. Scrape from Google News
2. Fetching news from financial data API (yahoo finance python API etc.)

