import numpy as np
import pandas as pd
import yfinance as yf
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import warnings
warnings.filterwarnings('ignore')

class TradingSignalModel:
    def __init__(self, ticker='AAPL', period='1y'):
        self.ticker = ticker
        self.period = period
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        
    def fetch_data(self):
        """Fetch OHLCV data from yfinance"""
        df = yf.download(self.ticker, period=self.period, progress=False)
        # Flatten MultiIndex columns if present if isinstance(df.columns, pd.MultiIndex): 
        if isinstance(df.columns, pd.MultiIndex): 
            df.columns = df.columns.get_level_values(0)
        return df
    
    def create_features(self, df):
        """Create technical indicator features from OHLCV data"""
        df = df.copy()
        
        # Price-based features
        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Diff'] = df['MACD'] - df['MACD_Signal']
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Momentum indicators
        df['Momentum'] = df['Close'] - df['Close'].shift(10)
        df['Rate_of_Change'] = ((df['Close'] - df['Close'].shift(12)) / df['Close'].shift(12)) * 100
        
        # Volume features
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        # df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        # Price features
        df['High_Low_Ratio'] = df['High'] / df['Low']
        df['Close_Open_Ratio'] = df['Close'] / df['Open']
        df['Price_Range'] = (df['High'] - df['Low']) / df['Close']
        
        return df
    
    def create_labels(self, df, lookahead=5):
        """Create trading signal labels based on future price movement"""
        df = df.copy()
        
        # Calculate future returns
        df['Future_Return'] = df['Close'].shift(-lookahead) / df['Close'] - 1
        
        # Define thresholds
        buy_threshold = 0.02  # 2% gain
        sell_threshold = -0.02  # 2% loss
        
        # Create labels: 0=Sell, 1=Hold, 2=Buy
        df['Signal'] = 1  # Default to Hold
        df.loc[df['Future_Return'] > buy_threshold, 'Signal'] = 2  # Buy
        df.loc[df['Future_Return'] < sell_threshold, 'Signal'] = 0  # Sell
        
        return df
    
    def prepare_data(self):
        """Prepare data for model training"""
        df = self.fetch_data()
        df = self.create_features(df)
        df = self.create_labels(df)
        
        # Drop NaN values
        df = df.dropna()
        
        # Define feature columns
        self.feature_columns = [col for col in df.columns 
                               if col not in ['Signal', 'Future_Return', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
        
        X = df[self.feature_columns]
        y = df['Signal']
        
        return X, y, df
    
    def train(self):
        """Train XGBoost model"""
        X, y, df = self.prepare_data()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train XGBoost model
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            objective='multi:softprob',
            device='cuda',
            num_class=3
        )
        
        self.model.fit(X_train_scaled, y_train, verbose=False)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        print(f"\n{'='*50}")
        print(f"Model Performance for {self.ticker}")
        print(f"{'='*50}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Sell', 'Hold', 'Buy']))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        return self.model, self.scaler
    
    def predict_signal(self, recent_data=None):
        """Predict trading signal for latest data"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        if recent_data is None:
            df = self.fetch_data()
            df = self.create_features(df)
            df = df.fillna(method='ffill').dropna().tail(1)
        else:
            df = recent_data
          
        # Only use feature columns that exist in the current data
        available_features = [col for col in self.feature_columns if col in df.columns]
        X = df[available_features]
        X_scaled = self.scaler.transform(X)
        
        signal = self.model.predict(X_scaled)[0]
        probabilities = self.model.predict_proba(X_scaled)[0]
        
        signal_map = {0: 'Sell', 1: 'Hold', 2: 'Buy'}
        
        return {
            'signal': signal_map[signal],
            'probabilities': {
                'Sell': probabilities[0],
                'Hold': probabilities[1],
                'Buy': probabilities[2]
            },
            'confidence': max(probabilities)
        }
    
    def save_model(self, model_path='trading_model.pkl', scaler_path='scaler.pkl'):
        """Save model and scaler"""
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"Model saved to {model_path}")
        print(f"Scaler saved to {scaler_path}")
    
    def load_model(self, model_path='trading_model.pkl', scaler_path='scaler.pkl'):
        """Load model and scaler"""
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        print("Model and scaler loaded successfully")
