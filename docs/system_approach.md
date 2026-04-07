# Hybrid Sentiment and Technical Indicator-Based Trading Strategy Using Machine Learning

## 1. Introduction and Motivation

Financial markets are inherently noisy, non-stationary, and influenced by a combination of quantitative signals (price, volume, momentum) and qualitative information (news, investor sentiment, macroeconomic events). Traditional trading strategies that rely solely on technical analysis or fundamental analysis each capture only a partial view of market dynamics. A **hybrid approach** — one that fuses numerical technical indicators with natural language-derived sentiment signals — is motivated by the hypothesis that these two information channels are complementary: technical indicators capture endogenous market microstructure (trends, mean-reversion, volatility regimes), while sentiment features capture exogenous information shocks that may precede or amplify price movements.

This system implements such a hybrid strategy by combining:

1. **Sentiment Analysis** — FinBERT-based scoring of financial news headlines, aggregated into daily sentiment features.
2. **Technical Indicator Engineering** — A rich set of momentum, trend, volatility, and volume indicators derived from OHLCV (Open, High, Low, Close, Volume) price data.
3. **Machine Learning Classification** — An XGBoost gradient-boosted tree classifier that ingests both feature channels and outputs a three-class trading signal: *Bullish* (Buy), *Bearish* (Sell), or *Neutral* (Hold).

The remainder of this document explains the rationale and methodology behind each component, with particular emphasis on the XGBoost model — its training pipeline, validation strategy, and avenues for improving prediction accuracy.

---

## 2. System Architecture Overview

The system follows a modular pipeline architecture with clearly separated concerns:

```
Raw Data Sources                  Feature Engineering               ML Pipeline                Deployment
─────────────────                 ───────────────────               ───────────                ──────────
                                                                                              
 Finnhub API  ──► News CSVs ──►  FinBERT Scoring ──► Daily        ┌─────────────────┐        Live Signal
 (headlines)      per ticker      (per headline)     Sentiment  ──►│                 │        Prediction
                                                     Features     │  XGBoost         │──────► (BUY/SELL/
                                                                  │  Classifier      │        HOLD)
 Yahoo Finance ► OHLCV Data ──►  Technical          Technical  ──►│  (3-class)       │
 (price/volume)                   Indicators          Features     │                 │──────► Backtest
                                                                  └─────────────────┘        Simulation
                                  Label Generation ──► Target
                                  (forward returns)    Labels
```

### Pipeline Stages

| Stage | Module | Purpose |
|-------|--------|---------|
| **News Collection** | `finnhub_news.py` | Fetch historical company news from the Finnhub API in weekly chunks |
| **Sentiment Scoring** | `news_sentimental_analysis.py` | Score each headline using ProsusAI/FinBERT |
| **Sentiment Aggregation** | `sentiment_features.py` | Aggregate per-headline scores into daily features |
| **Price Data** | `features.py` | Download OHLCV data via Yahoo Finance (`yfinance`) |
| **Technical Indicators** | `features.py` | Compute momentum, trend, volatility, and volume indicators |
| **Label Generation** | `features.py` | Create supervised learning targets from forward-looking returns |
| **Model Training** | `train.py` | Train XGBoost/Random Forest with walk-forward validation |
| **Evaluation** | `evaluation.py` | Per-sector and per-ticker performance analysis |
| **Backtesting** | `backtest.py`, `simulation.py` | Day-by-day trading simulation with portfolio tracking |

---

## 3. Sentiment Analysis Pipeline (Brief Overview)

> *Note: The sentiment analysis component was covered in detail in the previous semester's report. This section provides a concise summary for completeness.*

### 3.1 Why Sentiment Analysis?

Market prices do not move purely on technical patterns — they react to new information. Financial news encodes forward-looking expectations, risk perceptions, and narrative shifts that technical indicators alone cannot capture. By quantifying the sentiment polarity of news flow, we provide the model with a proxy for the information environment surrounding a stock on any given day.

### 3.2 FinBERT Scoring

The system uses **FinBERT** (`ProsusAI/finbert`), a BERT-based language model fine-tuned on financial text. Each headline is classified into one of three sentiment classes — *positive*, *negative*, or *neutral* — with an associated confidence score. The raw FinBERT output is converted into a signed numeric value:

- **Positive** headlines → `+score`
- **Negative** headlines → `-score`
- **Neutral** headlines → `0.0`

### 3.3 Daily Sentiment Feature Aggregation

Since the ML model operates on daily granularity (one prediction per trading day), individual headline scores must be aggregated into daily features. The aggregation produces **11 sentiment features** per day:

| Feature | Description |
|---------|-------------|
| `sent_mean` | Mean sentiment score across all headlines that day |
| `sent_std` | Standard deviation of scores (measures disagreement) |
| `sent_count` | Number of articles (proxy for attention/coverage) |
| `sent_max` / `sent_min` | Extreme sentiment values |
| `sent_positive_ratio` | Fraction of headlines classified as positive |
| `sent_negative_ratio` | Fraction of headlines classified as negative |
| `sent_momentum_3d/5d/10d` | Rolling mean of `sent_mean` (captures sentiment trends) |
| `sent_vol_5d` | Rolling volatility of daily sentiment (sentiment stability) |

These sentiment features are joined to the price-based feature matrix via a left join on the date index, with forward-filling and zero-imputation for dates without news coverage.

---

## 4. Technical Indicator Engineering

### 4.1 Why Technical Indicators?

While raw OHLCV prices contain the fundamental price information, they are **non-stationary** (trending upward or downward over time) and **scale-dependent** (a $1 move on a $10 stock is very different from a $1 move on a $500 stock). Technical indicators transform raw prices into **derived features** that capture underlying market dynamics — momentum, trend strength, volatility, and volume patterns — in a more stationary and informative representation that machine learning models can exploit.

### 4.2 Indicator Categories

The system computes approximately **40+ technical indicators** across four categories:

#### Momentum Indicators
Momentum indicators measure the rate of price change, helping identify whether a trend is accelerating or decelerating.

- **RSI** (Relative Strength Index) at periods 5, 10, and 15 — measures overbought/oversold conditions
- **ROC** (Rate of Change, 10-day) — percentage price change over a fixed window
- **MOM** (Momentum, 10-day) — absolute price change
- **Stochastic RSI** — RSI applied to itself, yielding a more sensitive oscillator
- **KST** (Know Sure Thing) — weighted rate-of-change oscillator

#### Trend Indicators
Trend indicators reveal the direction and strength of the prevailing price movement.

- **SMA** (Simple Moving Average) at 5, 10, 20 periods
- **EMA** (Exponential Moving Average) at 5, 10, 20 periods
- **VWMA** (Volume-Weighted Moving Average, 20-period)
- **MACD** (Moving Average Convergence Divergence) — difference between 12-period and 26-period EMA

#### Volatility Indicators
Volatility indicators quantify the degree of price uncertainty.

- **Bollinger Bands** (20-period) — upper/lower/mid bands and bandwidth
- **ATR** (Average True Range, 14-period) — mean of daily true range
- **Keltner Channels** (20-period) — ATR-based envelope around the EMA

#### Volume Indicators
Volume indicators incorporate trading volume to confirm or diverge from price signals.

- **OBV** (On-Balance Volume) — cumulative volume flow
- **A/D** (Accumulation/Distribution) — volume-weighted price position indicator
- **EFI** (Elder's Force Index) — price change × volume
- **NVI/PVI** (Negative/Positive Volume Index) — tracks price changes on low/high volume days

### 4.3 Scale-Invariant (Relative) Features for Cross-Stock Generalisation

A key challenge in building a **universal model** that generalises across multiple stocks is that raw indicator values are inherently scale-dependent. For example, the absolute value of SMA(20) for AAPL at ~$170 is not comparable to JPM at ~$220. To enable cross-stock training, the system computes an additional set of **scale-invariant relative features**:

- **MA Ratios**: `close / SMA(n) - 1`, `close / EMA(n) - 1`, `SMA(5) / SMA(20) - 1` — express the current price relative to its moving averages as a percentage deviation
- **Bollinger %B**: `(close - lower_band) / bandwidth` — position within the Bollinger Band envelope (0 = at lower band, 1 = at upper band)
- **Keltner %B**: analogous to Bollinger %B but for Keltner Channels
- **ATR as % of price**: `ATR / close` — normalises volatility by price level
- **Log returns** at 1, 5, 10, 20-day horizons — inherently scale-free
- **Realised volatility**: rolling standard deviation of 1-day log returns at 5, 10, 20-day windows
- **Volume ratios**: `volume / SMA(volume, n)` — relative volume surge detection
- **Calendar features**: cyclical encodings of day-of-week and month using sine/cosine transformations, allowing the model to capture weekly and seasonal patterns without introducing artificial ordinal relationships

These relative features are critical for the universal model's ability to learn patterns that transfer across different stocks and price scales.

---

## 5. Label Generation (Target Variable)

### 5.1 Why a Three-Class Formulation?

The system frames the prediction task as a **three-class classification** problem rather than binary (up/down) or regression (continuous return prediction). This is a deliberate design choice:

- **Binary classification** (up/down) forces the model to make a prediction on days where price movement is negligible, introducing label noise.
- **Regression** (predicting exact returns) is significantly harder due to the high noise-to-signal ratio in financial data; small magnitude errors can flip the sign of the predicted return.
- **Three-class classification** introduces a **neutral/hold** class that absorbs the noisy, low-magnitude movements, allowing the model to concentrate its discriminative power on identifying clear directional moves.

### 5.2 Fixed-Threshold Labels

The default labelling strategy uses a **forward-looking mean return** with a fixed percentage threshold:

1. Compute the mean closing price over the next `lookahead` days (default: 5 trading days).
2. Calculate the percentage change between today's close and that future mean.
3. Assign labels:
   - **Label 2 (Bullish)**: forward return ≥ `+thresh` (default: +1%)
   - **Label 1 (Bearish)**: forward return ≤ `-thresh` (default: -1%)
   - **Label 0 (Neutral)**: forward return between -1% and +1%

Using the forward **mean** rather than a single future day's price smooths out day-to-day noise and produces more stable labels.

### 5.3 Adaptive (Rolling-Percentile) Labels

A limitation of fixed thresholds is that they produce **class imbalance** across different volatility regimes. During high-volatility periods (e.g. market crises), almost all returns exceed ±1%, so the neutral class nearly disappears. During calm periods, most returns are small, inflating the neutral class.

The adaptive labelling strategy addresses this by computing **rolling percentile thresholds** over a trailing window (default: 60 trading days). The upper and lower boundaries of the neutral zone are set at the 67th and 33rd percentiles of recent forward returns, respectively. This ensures that approximately one-third of samples fall into each class regardless of the volatility regime, maintaining class balance across the training set.

### 5.4 Label Leakage Considerations

Because labels are constructed from **future prices**, care must be taken to avoid lookahead bias during model evaluation. The system addresses this through:

- **Chronological train/test splits** — the test set always follows the training set in time
- **Purged time-series cross-validation** — an explicit gap (`purge_gap`) is inserted between the training and test folds to prevent label leakage from the rolling forward-return window

---

## 6. XGBoost Model: Training Pipeline

### 6.1 Why XGBoost?

XGBoost (eXtreme Gradient Boosting) is chosen as the primary classifier for several reasons that align with the characteristics of financial feature data:

1. **Handles heterogeneous features natively**: unlike neural networks which require careful normalisation, XGBoost operates on decision trees that are invariant to feature scaling and can naturally handle a mix of bounded oscillators (RSI ∈ [0,100]) and unbounded indicators (OBV ∈ (-∞, +∞)).

2. **Built-in regularisation**: L1 (`reg_alpha`) and L2 (`reg_lambda`) regularisation, `max_depth` limits, `min_child_weight`, and subsampling (`subsample`, `colsample_bytree`) provide multiple levers to control overfitting — critical when the signal-to-noise ratio is low.

3. **Robustness to irrelevant features**: the tree-based feature selection mechanism (gain-based splits) naturally down-weights uninformative features rather than fitting to noise.

4. **Interpretability via SHAP**: TreeSHAP provides exact, polynomial-time Shapley value computation for tree ensembles, enabling feature importance analysis that is consistent and locally accurate.

5. **Computational efficiency**: XGBoost's histogram-based split finding and parallel tree construction scale well to the dataset sizes typical of multi-stock training (~50,000+ rows).

### 6.2 Default Hyperparameter Configuration

The baseline XGBoost model uses the following configuration:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `n_estimators` | 300 | Number of boosting rounds; enough to learn complex patterns while early stopping can terminate sooner if needed |
| `max_depth` | 6 | Maximum tree depth; controls model complexity. Deeper trees can capture higher-order feature interactions but risk overfitting |
| `learning_rate` | 0.05 | Step size shrinkage; smaller values require more trees but produce more robust ensembles |
| `subsample` | 0.9 | Row subsampling ratio per tree; mild stochastic regularisation |
| `colsample_bytree` | 0.8 | Feature subsampling ratio per tree; reduces correlation between trees and provides implicit feature selection |
| `objective` | `multi:softprob` | Multi-class classification with softmax probability output |
| `eval_metric` | `mlogloss` | Multi-class log loss; penalises confident wrong predictions |
| `random_state` | 42 | Reproducibility seed |

### 6.3 Model Training Configurations

The system trains and compares four model configurations to quantify the marginal value of each feature channel:

| Configuration | Features | Model | Purpose |
|--------------|----------|-------|---------|
| `technical_only_xgb` | Technical indicators only | XGBoost | Baseline — what can pure technical analysis achieve? |
| `sentiment_only_xgb` | Sentiment features only | XGBoost | Ablation — is sentiment alone predictive? |
| `hybrid_xgb` | Technical + Sentiment | XGBoost | The full hybrid model |
| `hybrid_rf` | Technical + Sentiment | Random Forest | Cross-model comparison; validates that XGBoost's boosting strategy outperforms bagging |

This **ablation study** design is important: by comparing the hybrid model against each individual channel, we can determine whether the sentiment features provide additive predictive value beyond what technical indicators alone capture.

### 6.4 Walk-Forward Cross-Validation

#### Why Walk-Forward (Not K-Fold)?

Standard k-fold cross-validation randomly shuffles data into folds, which violates the temporal ordering of financial time series. A model trained on 2024 data and tested on 2023 data would have access to future information, producing artificially inflated accuracy estimates. **Walk-forward validation** respects the causal structure of time series by ensuring the model is always trained on past data and evaluated on future data.

#### Implementation

The system uses an **expanding-window** walk-forward scheme with `n_splits` folds (default: 5). In each fold:

1. The training window expands from the start of the dataset to a progressively later cutoff point.
2. The test window covers the period immediately following the training cutoff.
3. An optional **purge gap** of `purge_gap` rows (default: 10) is inserted between training and test sets to eliminate label leakage from the forward-return window used in label construction.

```
Fold 1: [===== Train =====]---gap---[= Test =]
Fold 2: [=========== Train ===========]---gap---[= Test =]
Fold 3: [================= Train =================]---gap---[= Test =]
Fold 4: [======================== Train ========================]---gap---[= Test =]
Fold 5: [============================== Train ==============================]---gap---[= Test =]
```

Metrics (accuracy, F1-macro, precision, recall) are computed per fold and then averaged to produce the final cross-validated performance estimate.

### 6.5 Evaluation Metrics

The model is evaluated using multiple metrics to provide a comprehensive view of performance:

| Metric | Description | Why It Matters |
|--------|-------------|----------------|
| **Accuracy** | Fraction of correct predictions | Simple but misleading under class imbalance |
| **F1-Macro** | Unweighted average of per-class F1 scores | Primary metric; treats all three classes equally, robust to imbalance |
| **Precision (Macro)** | Average fraction of correct positive predictions per class | Measures how reliable the model's signals are |
| **Recall (Macro)** | Average fraction of actual positives correctly identified per class | Measures how many opportunities the model captures |
| **ROC-AUC** | Area under the one-vs-rest ROC curve per class | Evaluates the model's discriminative ability at all confidence thresholds |

**F1-Macro** is the primary metric because it penalises both false positives (bad trades) and false negatives (missed opportunities) equally across all three classes, including the often-underrepresented bearish class.

### 6.6 Feature Importance via SHAP

After training, **SHAP (SHapley Additive exPlanations)** values are computed using the `TreeExplainer` to quantify each feature's contribution to predictions. SHAP provides:

- **Global importance**: mean absolute SHAP value per feature, revealing which indicators drive the model's decisions overall.
- **Local explanations**: per-sample SHAP values showing why the model produced a specific signal on a given day.

This analysis is useful for validating that the model's learned patterns align with financial intuition (e.g., RSI divergence, volume spikes) and for identifying whether sentiment features contribute meaningfully to the hybrid model's predictions.

---

## 7. Universal (Cross-Stock) Model

### 7.1 Why a Universal Model?

Per-ticker models are limited by **small sample sizes** (a single stock over 2-3 years yields ~500-750 training samples) and cannot generalise to unseen stocks. A universal model, trained on pooled data from many stocks across multiple sectors, addresses these limitations:

- **Larger training set**: pooling 65+ tickers over 5+ years yields tens of thousands of training samples, reducing overfitting risk.
- **Cross-stock pattern learning**: market microstructure patterns (momentum, mean-reversion, volatility clustering) are shared across stocks; a universal model can learn these common patterns from diverse examples.
- **Generalisation to unseen stocks**: by training on scale-invariant features, the model can generate signals for stocks it has never seen during training — validated via held-out stock evaluation.

### 7.2 Sector-Aware Architecture

The stock universe is organised into **7 sectors**, each represented by 5–10 representative tickers:

| Sector | Example Tickers | Sector ID |
|--------|----------------|-----------|
| Technology | AAPL, MSFT, GOOGL, META, NVDA | 0 |
| Financials | JPM, BAC, GS, MS, WFC | 1 |
| Energy | CVX, XOM, COP, SLB, EOG | 2 |
| Healthcare | UNH, JNJ, PFE, ABBV, MRK | 3 |
| Consumer | AMZN, TSLA, HD, MCD, NKE | 4 |
| Industrials | CAT, BA, HON, UPS, GE | 5 |
| ETFs | VOO, SPY, QQQ, IWM, DIA | 6 |

Each row in the pooled dataset is tagged with a `sector_id`, enabling sector-level fine-tuning and evaluation.

### 7.3 Two-Phase Training Strategy

A core challenge is that **sentiment data coverage is temporally limited** — the Finnhub news API provides reliable historical news only for approximately the most recent 1–2 years, while price data is available for 5+ years. Training exclusively on the sentiment-available window would discard years of valuable technical pattern data.

The two-phase training strategy resolves this mismatch:

**Phase 1 — Technical Base Model (multi-year data):**
- Train on the full historical dataset (e.g. 2019–2023), with sentiment feature columns present in the schema but **zeroed out** for all dates before the sentiment availability cutoff.
- The model learns robust technical patterns from a large, diverse dataset. The zeroed sentiment columns teach the base model that "no sentiment information" is a valid input state.

**Phase 2 — Sentiment Fine-Tuning (recent data):**
- **Warm-start** from the Phase 1 booster by passing the trained model's booster object to XGBoost's `xgb_model` parameter.
- Fine-tune on the recent window where real sentiment features are available (e.g. 2025-04-01 onward).
- Use a reduced learning rate (0.01 vs 0.05) and limited additional boosting rounds (100) to adapt the model to utilise sentiment signal without catastrophically overwriting the technical patterns learned in Phase 1.

This warm-start approach is analogous to transfer learning in deep learning: the Phase 1 model provides a strong initialisation, and Phase 2 adjusts the decision boundaries to incorporate the additional sentiment signal.

### 7.4 Sector Fine-Tuning

After training the universal base model, optional **sector-specific fine-tuning** can be applied. Each sector's data is used to warm-start additional boosting rounds from the universal model, producing specialised models that capture sector-specific dynamics (e.g., energy stocks' sensitivity to oil prices, tech stocks' momentum patterns) while retaining the general market knowledge from the base model.

### 7.5 Temporal Train/Test Split

For the universal model, data is split **purely by date** (not by stock):

- **Train**: all rows up to `train_end` (e.g. 2023-12-31)
- **Validation**: rows between `train_end` and `val_end` (e.g. 2024-06-30) — used for early stopping
- **Test**: all rows after `val_end`

This ensures the model is evaluated on truly future data across all stocks simultaneously. **Early stopping** (patience of 30 rounds) monitors validation multi-class log loss to prevent overfitting.

---

## 8. Hyperparameter Optimisation with Optuna

### 8.1 Why Bayesian Optimisation?

Grid search or random search over XGBoost's 9+ hyperparameters is computationally expensive and sample-inefficient. **Optuna** uses a Tree-structured Parzen Estimator (TPE) — a Bayesian optimisation algorithm — that models the relationship between hyperparameters and the objective function, intelligently focusing the search on promising regions of the hyperparameter space.

### 8.2 Search Space

| Parameter | Range | Scale |
|-----------|-------|-------|
| `n_estimators` | 100 – 1,000 (step 50) | Linear |
| `max_depth` | 3 – 10 | Integer |
| `learning_rate` | 0.005 – 0.3 | Log-uniform |
| `subsample` | 0.5 – 1.0 | Uniform |
| `colsample_bytree` | 0.3 – 1.0 | Uniform |
| `min_child_weight` | 1 – 10 | Integer |
| `gamma` | 0.0 – 5.0 | Uniform |
| `reg_alpha` | 0.001 – 10.0 | Log-uniform |
| `reg_lambda` | 0.001 – 10.0 | Log-uniform |

### 8.3 Objective Function

Each Optuna trial:

1. Samples a hyperparameter configuration from the search space.
2. Trains an XGBoost model using **purged time-series cross-validation** (5 folds, 10-row purge gap).
3. Within each fold, holds out the last 10% of training data for **early stopping** (patience: 30 rounds), preventing individual trials from overfitting.
4. Reports the mean **F1-Macro** across all folds as the objective value to maximise.

The study runs for up to 50 trials (default) with a timeout of 600 seconds, returning the best parameter combination found.

---

## 9. Backtesting and Trading Simulation

### 9.1 Why Backtesting?

Prediction accuracy alone does not determine the practical value of a trading model. A model with 40% directional accuracy could still be profitable if its correct predictions correspond to larger price moves, or if it effectively avoids drawdowns. **Backtesting** evaluates the model in a simulated trading environment where signals are translated into buy/sell/hold actions and portfolio performance is tracked.

### 9.2 Trading Logic

The simulation implements a simple **long-only** strategy:

- **BUY signal (Label 2)**: if the model predicts Bullish and no position is held, invest all available cash.
- **SELL signal (Label 1)**: if the model predicts Bearish and a position is held, close the entire position.
- **HOLD signal (Label 0)**: take no action.
- **End of simulation**: all open positions are forcibly closed.

### 9.3 Performance Metrics

The backtesting module computes portfolio-level metrics that complement the classification metrics:

| Metric | Description |
|--------|-------------|
| **Overall Return** | Total percentage return over the simulation period |
| **Buy & Hold Return** | Return from simply buying on day 1 and holding — the baseline |
| **Excess Return** | Strategy return minus buy-and-hold return |
| **Maximum Drawdown** | Largest peak-to-trough decline in portfolio value — measures downside risk |
| **Sharpe Ratio** | Annualised risk-adjusted return: `mean(daily_returns) / std(daily_returns) × √252` |
| **Sortino Ratio** | Like Sharpe, but uses only downside deviation — penalises only harmful volatility |
| **Win Rate** | Fraction of round-trip trades (buy→sell) that were profitable |
| **Annualised Volatility** | Standard deviation of daily returns scaled to annual frequency |

### 9.4 Generalisability Testing

The system includes a **generalisation test** that runs the trained model on multiple test tickers, producing a comparison table. This validates whether the model has learned transferable market patterns or has overfit to the training tickers.

---

## 10. Current Limitations and Improvement Roadmap

The current model achieves approximately **~40% prediction accuracy**, which, while above the ~33% random baseline for a three-class problem, leaves significant room for improvement. The following analysis identifies the likely causes and proposes concrete improvements.

### 10.1 Diagnosing the Low Accuracy

#### Class Imbalance and Label Noise

With a fixed ±1% threshold, the class distribution is sensitive to the market regime during the training period. Bull markets produce a surplus of bullish labels; sideways markets inflate the neutral class. This imbalance causes the model to develop a bias toward the majority class.

**Improvement**: Adopt the **adaptive (rolling-percentile) labelling** strategy already implemented in the system (`generate_adaptive_label`). By using rolling 33rd/67th percentile thresholds over a 60-day window, class balance is maintained across volatility regimes, and the model is forced to learn relative rather than absolute return patterns.

#### Insufficient Training Data

Per-ticker models trained on 2-3 years of data have ~500 samples — far too few for a model with 40+ features to learn robust patterns without overfitting.

**Improvement**: Use the **universal cross-stock model** with pooled data from 65+ tickers across 5+ years, yielding 30,000–50,000 training samples. The scale-invariant relative features ensure that cross-stock pooling is meaningful.

#### Suboptimal Hyperparameters

The default hyperparameters (`max_depth=6`, `learning_rate=0.05`, `n_estimators=300`) were chosen heuristically and may not be optimal for the specific noise characteristics of financial data.

**Improvement**: Run the **Optuna hyperparameter search** with sufficient trials (100+). Key parameters to optimise include:
- `max_depth` (3–10): shallower trees (3–5) may generalise better on noisy financial data
- `learning_rate` (0.005–0.1): lower learning rates with more estimators typically improve out-of-sample performance
- `reg_alpha` and `reg_lambda`: stronger L1/L2 regularisation can suppress overfitting to noise
- `min_child_weight` (5–20): requiring more samples per leaf prevents the model from learning from tiny, noisy subsets

#### Lack of Early Stopping in Per-Ticker Training

The per-ticker training pipeline (`train_and_evaluate`) does not use early stopping — it trains for a fixed 300 rounds regardless of validation performance, likely overfitting in later rounds.

**Improvement**: Implement early stopping by holding out a validation set (e.g. the last 10% of the training partition) and monitoring `mlogloss`, halting training when validation loss stops improving.

### 10.2 Feature Engineering Improvements

#### Feature Selection / Dimensionality Reduction

With 40+ technical indicators, many of which are correlated (e.g., SMA(5), SMA(10), and SMA(20) are highly correlated), the model may be fitting to redundant or noisy features.

**Improvements**:
- **SHAP-based feature selection**: after an initial training run, remove features with near-zero mean absolute SHAP values. This data-driven pruning reduces noise dimensions while retaining the most predictive indicators.
- **Recursive Feature Elimination (RFE)**: iteratively remove the least important features and retrain, selecting the subset that maximises cross-validated F1.
- **Correlation filtering**: remove one of each pair of features with Pearson correlation > 0.95 to reduce multicollinearity.

#### Additional Feature Engineering

- **Cross-asset features**: include market-wide indicators (e.g. VIX, S&P 500 returns, sector ETF performance) as contextual features. Individual stocks do not move in isolation — market regime information can improve predictions.
- **Interaction features**: engineered features like `RSI × sent_mean` (momentum-sentiment interaction) could capture nonlinear relationships that even tree-based models may struggle to discover from raw features alone.
- **Higher-frequency sentiment aggregation**: instead of daily aggregation, use intraday sentiment windows (e.g. pre-market, post-market) to capture the timing of sentiment shifts.

### 10.3 Model Architecture Improvements

#### Class Weighting / Cost-Sensitive Learning

Even with balanced labels, the **cost of misclassification is asymmetric** in trading: a false bullish signal (buying before a decline) is more costly than a false neutral signal (missing a moderate gain). XGBoost supports custom `sample_weight` arrays that can penalise certain misclassifications more heavily.

**Implementation**: assign higher weights to bullish/bearish samples and lower weights to neutral samples during training, or use a custom multi-class loss function that penalises directional misclassifications (predicting bullish when the true label is bearish) more than magnitude misclassifications (predicting neutral when the true label is mildly bullish).

#### Ensemble Methods

Rather than relying on a single XGBoost model, combine predictions from multiple models:
- **Model averaging**: train an XGBoost, a LightGBM, and a Random Forest on the same features; average their predicted probabilities before making the final classification.
- **Stacking**: use the predicted probabilities from multiple base models as features for a meta-learner (e.g. logistic regression).
- **Temporal ensembling**: train multiple models on different historical windows and weight their predictions by recency.

#### Confidence-Based Filtering

The model outputs class probabilities via `predict_proba`. Currently, the argmax class is always used as the trading signal regardless of confidence. A **confidence threshold** can improve precision:

- Only act on signals where the predicted probability exceeds a threshold (e.g. 50% or 60%).
- Default to HOLD when the model is uncertain.
- This trades recall for precision, reducing the number of trades but improving win rate.

### 10.4 Training Procedure Improvements

#### Purged Walk-Forward Validation

The current walk-forward validation uses `purge_gap=0` by default. Given that labels use a 5-day forward-looking window, a purge gap of at least 5–10 rows should be used to prevent **information leakage** between training and test folds.

#### Expanding vs. Sliding Window

The current expanding-window approach gives equal weight to distant and recent history. Financial markets exhibit **regime changes** — patterns that worked in 2019 may not apply in 2025. A **sliding window** (fixed-size training set that moves forward) or **exponential decay weighting** (down-weighting older samples) could improve the model's adaptability to evolving market conditions.

#### Data Augmentation

Financial time series data is scarce by nature (one data point per trading day). Techniques to artificially increase training diversity include:
- **Noise injection**: adding small Gaussian noise to features during training (dropout-like regularisation for tabular data)
- **Synthetic minority oversampling (SMOTE)**: generating synthetic samples for underrepresented classes — though this must be applied carefully to avoid creating unrealistic feature combinations in time-series data

### 10.5 Alternative Labelling Strategies

#### Return Magnitude Bucketing
Instead of a binary threshold, use quantile-based bucketing (e.g. quintiles of forward returns) to create labels that better reflect the distribution of returns.

#### Risk-Adjusted Labels
Incorporate volatility into the label definition: a +1% return during a period where ATR is 3% is less significant than a +1% return when ATR is 0.5%. Normalising returns by realised volatility before thresholding produces more meaningful labels.

#### Shorter or Longer Lookahead
The default 5-day lookahead may not align with the optimal trading horizon. Experimenting with 1-day (day trading), 10-day, or 20-day (swing trading) horizons may reveal time horizons where technical and sentiment features are more predictive.

---

## 11. Summary

This hybrid trading system combines two complementary information channels — NLP-derived sentiment and quantitative technical indicators — into a unified machine learning framework. The XGBoost gradient-boosted tree classifier is well-suited to this tabular, heterogeneous feature space, and the modular pipeline architecture supports rigorous evaluation through walk-forward validation, ablation studies, and portfolio-level backtesting.

The current ~40% accuracy represents a starting point that can be meaningfully improved through a combination of:

1. **Better labels** — adaptive thresholds, risk-adjusted labelling
2. **More data** — universal cross-stock model training with 65+ tickers
3. **Stronger regularisation** — Optuna-tuned hyperparameters, early stopping, feature selection
4. **Smarter trading** — confidence-based signal filtering, asymmetric cost functions

These improvements target the fundamental challenges of financial ML: low signal-to-noise ratio, non-stationarity, and regime dependence — and represent a principled roadmap for advancing the system's predictive and economic performance.
