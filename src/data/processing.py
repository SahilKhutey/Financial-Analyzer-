"""
Data Processing Utilities.
Handles image normalization for Vision models and feature engineering for TS models.
"""
import pandas as pd
import numpy as np
from PIL import Image
import io

class ImageProcessor:
    """
    Prepares images for the Vision Engine.
    """
    
    @staticmethod
    def preprocess_image(image: Image.Image, target_size=(224, 224)) -> np.ndarray:
        """
        Resize and normalize image for EfficientNet.
        """
        # Resize
        if image.size != target_size:
            image = image.resize(target_size)
            
        # Convert to RGB (in case of RGBA/Grayscale)
        image = image.convert('RGB')
        
        # To Array and Normalize (0-1)
        arr = np.array(image) / 255.0
        
        # Add Batch Dimension (1, H, W, C) for model input
        return np.expand_dims(arr, axis=0)

class FeatureEngineer:
    """
    Prepares numerical data for the Time-Series Engine.
    """
    
    @staticmethod
    def compute_technical_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add features: Log Returns, Volatility, Volume Delta, RSI, MACD, ATR.
        """
        data = df.copy()
        
        # 1. Log Returns & Volatility
        data['log_returns'] = np.log(data['Close'] / data['Close'].shift(1))
        data['volatility'] = data['log_returns'].rolling(window=20).std()
        
        # 2. Volume Delta
        data['volume_delta'] = data['Volume'].pct_change()
        
        # 3. RSI (14)
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))
        
        # 4. MACD (12, 26, 9)
        ema12 = data['Close'].ewm(span=12, adjust=False).mean()
        ema26 = data['Close'].ewm(span=26, adjust=False).mean()
        data['macd_line'] = ema12 - ema26
        data['macd_signal'] = data['macd_line'].ewm(span=9, adjust=False).mean()
        data['macd_hist'] = data['macd_line'] - data['macd_signal']
        
        # 5. ATR (14) - For Regime/Risk
        high_low = data['High'] - data['Low']
        high_close = (data['High'] - data['Close'].shift()).abs()
        low_close = (data['Low'] - data['Close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        data['atr'] = tr.rolling(window=14).mean()
        
        # Drop initialization NaNs
        data.dropna(inplace=True)
        
        return data
