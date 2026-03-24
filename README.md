# Hybrid Sentiment and Technical Indicator-Based Trading Strategy Using Machine Learning 

Final Year Project – CUHK Computer Science and Engineering

## Introduction

This project designs and implements a machine learning-based trading system that integrates
both market sentiment analysis and technical indicators to generate more accurate and reliable
trading signals.

The system combines:
- **FinBERT** sentiment scoring of financial news (via Finnhub API)
- **30+ technical indicators** computed from OHLCV price data (via pandas_ta)
- **XGBoost** and **Random Forest** classifiers with walk-forward validation

## Architecture

```
Data Sources              Feature Engineering          ML Pipeline
─────────────            ────────────────────         ──────────────
yfinance API   ──────>   Technical Indicators   ──┐
                         (RSI, MACD, BB, etc.)     ├──>  XGBoost / RF  ──>  Backtest
Finnhub News   ──────>   FinBERT Sentiment      ──┘     Walk-Forward
  API                    (daily aggregates)              Validation
```

## Tracked Securities

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

## Setup

### Prerequisites
- Python >= 3.12
- [uv](https://github.com/astral-sh/uv) package manager
- Finnhub API key (free tier: https://finnhub.io/)

### Installation

```bash
# Install dependencies
uv sync

# Set up environment variables
echo "FINNHUB_API_KEY=your_key_here" > .env
```

### Data Collection

The system collects data from two sources:

1. **Price data**: Downloaded via the `yfinance` Python API (no key required)
2. **News data**: Fetched from the Finnhub `/company-news` endpoint, which provides
   historical news articles with timestamps for any given date range

### ML Model

We chose **XGBoost** as the primary classifier because:
- Best-in-class performance on tabular/structured data
- Handles mixed feature types (continuous technical indicators + sentiment scores)
- Built-in feature importance for interpretability
- Compatible with `TimeSeriesSplit` for walk-forward validation

**Random Forest** serves as a baseline comparison.

## Usage

```bash
# Interactive mode
uv run python main.py

# Full pipeline (collect news, score, train, evaluate, backtest)
uv run python main.py   # then select option 4
```

## Project Structure

```
src/
├── sentiment/
│   ├── finnhub_news.py              # Finnhub historical news collection
│   ├── financial_news.py            # Google News scraper (live demo)
│   ├── news_sentimental_analysis.py # FinBERT sentiment scoring
│   ├── sentiment_features.py        # Daily sentiment feature aggregation
│   ├── entity_extraction.py         # NER with GLiNER
│   └── text_correlation.py          # Sentence similarity (demo)
├── technical/
│   ├── stock_data.py                # yfinance data download
│   ├── indicators.py                # Manual indicator implementations
│   ├── strategy.py                  # CSV data loader for backtesting
│   └── backtest.py                  # Backtesting strategies
└── ml/
    ├── features.py                  # Feature engineering + merging
    ├── train.py                     # Training pipeline + walk-forward
    └── RandomForest.py              # RF baseline reference

financial_data.ipynb                 # Exploratory notebook (XGBoost prototype)
main.py                              # CLI entry point
```

## Results

Results are saved to the `results/` directory:
- `{symbol}_comparison.csv` – accuracy/F1 across all configurations
- `{symbol}_confusion_matrix.png` – normalized confusion matrix
- `{symbol}_roc_curves.png` – ROC curves per class
- `{symbol}_shap_importance.png` – SHAP feature importance (top 20)
- `{symbol}_backtest_comparison.csv` – strategy return comparison
- `cross_ticker_summary.csv` – all tickers side-by-side
