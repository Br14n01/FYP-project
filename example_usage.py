from src.technical.trading_signals_model import TradingSignalModel

# Initialize model for Apple stock
model = TradingSignalModel(ticker='AAPL', period='2y')

# Train the model
print("Training XGBoost Trading Signal Model...")
model.train()

# Save the trained model
model.save_model()

# Get trading signal for latest data
print("\n" + "="*50)
print("Latest Trading Signal")
print("="*50)
signal_result = model.predict_signal()
print(f"Signal: {signal_result['signal']}")
print(f"Confidence: {signal_result['confidence']:.2%}")
print(f"Probabilities:")
for action, prob in signal_result['probabilities'].items():
    print(f"  {action}: {prob:.2%}")

# Example: Test with multiple stocks
print("\n" + "="*50)
print("Testing Multiple Stocks")
print("="*50)
for ticker in ['MSFT']:
    model_ticker = TradingSignalModel(ticker=ticker, period='1y')
    model_ticker.train()
    signal = model_ticker.predict_signal()
    print(f"\n{ticker}: {signal['signal']} (Confidence: {signal['confidence']:.2%})")
