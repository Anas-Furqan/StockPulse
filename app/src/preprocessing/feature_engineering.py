import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple

class FeatureEngineer:
    """
    A class to engineer features from stock data for anomaly detection.
    """
    
    def __init__(self):
        """
        Initialize the FeatureEngineer.
        """
        pass
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features from the input DataFrame.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        # Make a copy to avoid modifying the original
        df_featured = df.copy()
        
        # Add price-based features
        df_featured = self._add_price_features(df_featured)
        
        # Add volume-based features
        df_featured = self._add_volume_features(df_featured)
        
        # Add volatility features
        df_featured = self._add_volatility_features(df_featured)
        
        # Add technical indicators
        df_featured = self._add_technical_indicators(df_featured)
        
        # Drop rows with NaN values created during feature engineering
        df_featured = df_featured.dropna()
        
        return df_featured
    
    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add price-based features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with price features added
        """
        # Group by Symbol if available
        if 'Symbol' in df.columns:
            result = pd.DataFrame()
            for symbol, group in df.groupby('Symbol'):
                result = pd.concat([result, self._calculate_price_features(group)])
            return result
        else:
            return self._calculate_price_features(df)
    
    def _calculate_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate price-based features for a single symbol.
        
        Args:
            df: Input DataFrame for a single symbol
            
        Returns:
            DataFrame with price features added
        """
        # Make a copy
        df_price = df.copy()
        
        # Daily returns
        df_price['Daily_Return'] = df_price['Close'].pct_change()
        
        # Log returns
        df_price['Log_Return'] = np.log(df_price['Close'] / df_price['Close'].shift(1))
        
        # Moving averages
        df_price['MA5'] = df_price['Close'].rolling(window=5).mean()
        df_price['MA10'] = df_price['Close'].rolling(window=10).mean()
        df_price['MA20'] = df_price['Close'].rolling(window=20).mean()
        
        # Price momentum
        df_price['Price_Momentum_1d'] = df_price['Close'] - df_price['Close'].shift(1)
        df_price['Price_Momentum_5d'] = df_price['Close'] - df_price['Close'].shift(5)
        
        # Price gap
        df_price['Price_Gap'] = df_price['Open'] - df_price['Close'].shift(1)
        
        # High-Low range
        df_price['HL_Range'] = df_price['High'] - df_price['Low']
        df_price['HL_Range_Pct'] = df_price['HL_Range'] / df_price['Close']
        
        return df_price
    
    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add volume-based features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with volume features added
        """
        # Check if Volume column exists
        if 'Volume' not in df.columns:
            return df
        
        # Group by Symbol if available
        if 'Symbol' in df.columns:
            result = pd.DataFrame()
            for symbol, group in df.groupby('Symbol'):
                result = pd.concat([result, self._calculate_volume_features(group)])
            return result
        else:
            return self._calculate_volume_features(df)
    
    def _calculate_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate volume-based features for a single symbol.
        
        Args:
            df: Input DataFrame for a single symbol
            
        Returns:
            DataFrame with volume features added
        """
        # Make a copy
        df_vol = df.copy()
        
        # Volume change
        df_vol['Volume_Change'] = df_vol['Volume'].pct_change()
        
        # Volume moving averages
        df_vol['Volume_MA5'] = df_vol['Volume'].rolling(window=5).mean()
        df_vol['Volume_MA10'] = df_vol['Volume'].rolling(window=10).mean()
        
        # Volume relative to moving average
        df_vol['Volume_Ratio_MA5'] = df_vol['Volume'] / df_vol['Volume_MA5']
        
        # Price-volume correlation
        df_vol['Price_Volume_Corr'] = df_vol['Close'].rolling(window=5).corr(df_vol['Volume'])
        
        return df_vol
    
    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add volatility-based features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with volatility features added
        """
        # Group by Symbol if available
        if 'Symbol' in df.columns:
            result = pd.DataFrame()
            for symbol, group in df.groupby('Symbol'):
                result = pd.concat([result, self._calculate_volatility_features(group)])
            return result
        else:
            return self._calculate_volatility_features(df)
    
    def _calculate_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate volatility-based features for a single symbol.
        
        Args:
            df: Input DataFrame for a single symbol
            
        Returns:
            DataFrame with volatility features added
        """
        # Make a copy
        df_vol = df.copy()
        
        # Daily volatility (standard deviation of returns)
        if 'Daily_Return' in df_vol.columns:
            df_vol['Volatility_5d'] = df_vol['Daily_Return'].rolling(window=5).std()
            df_vol['Volatility_10d'] = df_vol['Daily_Return'].rolling(window=10).std()
        
        # Average True Range (ATR)
        df_vol['TR'] = np.maximum(
            df_vol['High'] - df_vol['Low'],
            np.maximum(
                abs(df_vol['High'] - df_vol['Close'].shift(1)),
                abs(df_vol['Low'] - df_vol['Close'].shift(1))
            )
        )
        df_vol['ATR_5d'] = df_vol['TR'].rolling(window=5).mean()
        
        return df_vol
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with technical indicators added
        """
        # Group by Symbol if available
        if 'Symbol' in df.columns:
            result = pd.DataFrame()
            for symbol, group in df.groupby('Symbol'):
                result = pd.concat([result, self._calculate_technical_indicators(group)])
            return result
        else:
            return self._calculate_technical_indicators(df)
    
    def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators for a single symbol.
        
        Args:
            df: Input DataFrame for a single symbol
            
        Returns:
            DataFrame with technical indicators added
        """
        # Make a copy
        df_tech = df.copy()
        
        # Relative Strength Index (RSI)
        delta = df_tech['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df_tech['RSI'] = 100 - (100 / (1 + rs))
        
        # Moving Average Convergence Divergence (MACD)
        ema12 = df_tech['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df_tech['Close'].ewm(span=26, adjust=False).mean()
        df_tech['MACD'] = ema12 - ema26
        df_tech['MACD_Signal'] = df_tech['MACD'].ewm(span=9, adjust=False).mean()
        df_tech['MACD_Hist'] = df_tech['MACD'] - df_tech['MACD_Signal']
        
        # Bollinger Bands
        df_tech['BB_Middle'] = df_tech['Close'].rolling(window=20).mean()
        df_tech['BB_Std'] = df_tech['Close'].rolling(window=20).std()
        df_tech['BB_Upper'] = df_tech['BB_Middle'] + (df_tech['BB_Std'] * 2)
        df_tech['BB_Lower'] = df_tech['BB_Middle'] - (df_tech['BB_Std'] * 2)
        df_tech['BB_Width'] = (df_tech['BB_Upper'] - df_tech['BB_Lower']) / df_tech['BB_Middle']
        
        return df_tech