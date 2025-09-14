import os
import pandas as pd
import yfinance as yf
from alpha_vantage.timeseries import TimeSeries
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple

class DataFetcher:
    """
    A class to fetch stock data from various sources including Yahoo Finance,
    Alpha Vantage, and user-uploaded CSV files.
    """
    
    def __init__(self):
        """
        Initialize the DataFetcher with API keys from environment variables.
        """
        self.alpha_vantage_key = os.getenv("ALPHA_VANTAGE_KEY")
        
    def fetch_from_yahoo(self, tickers: List[str], start_date: str, end_date: str, interval: str = "1d") -> Dict[str, pd.DataFrame]:
        """
        Fetch stock data from Yahoo Finance.
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            interval: Data interval (1d, 1h, 5m, etc.)
            
        Returns:
            Dictionary mapping ticker symbols to DataFrames
        """
        result = {}
        for ticker in tickers:
            try:
                data = yf.download(
                    ticker,
                    start=start_date,
                    end=end_date,
                    interval=interval,
                    progress=False
                )
                
                
                data.reset_index(inplace=True)
                
                
                data['Symbol'] = ticker
                
                
                data.columns = [col if col != 'Adj Close' else 'Adj_Close' for col in data.columns]
                
                result[ticker] = data
            except Exception as e:
                print(f"Error fetching data for {ticker}: {e}")
        
        return result
    
    def fetch_from_alpha_vantage(self, tickers: List[str], output_size: str = "full", interval: str = "daily") -> Dict[str, pd.DataFrame]:
        """
        Fetch stock data from Alpha Vantage.
        
        Args:
            tickers: List of ticker symbols
            output_size: 'compact' (latest 100 data points) or 'full' (up to 20 years of data)
            interval: Data interval ('daily', 'weekly', 'monthly', 'intraday')
            
        Returns:
            Dictionary mapping ticker symbols to DataFrames
        """
        if not self.alpha_vantage_key:
            raise ValueError("Alpha Vantage API key not found in environment variables")
        
        ts = TimeSeries(key=self.alpha_vantage_key, output_format='pandas')
        result = {}
        
        for ticker in tickers:
            try:
                if interval == 'daily':
                    data, meta_data = ts.get_daily(symbol=ticker, outputsize=output_size)
                elif interval == 'weekly':
                    data, meta_data = ts.get_weekly(symbol=ticker)
                elif interval == 'monthly':
                    data, meta_data = ts.get_monthly(symbol=ticker)
                elif interval == 'intraday':
                    data, meta_data = ts.get_intraday(symbol=ticker, interval='60min', outputsize=output_size)
                else:
                    raise ValueError(f"Unsupported interval: {interval}")
                
                
                data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                
                
                data['Adj_Close'] = data['Close']
                
                
                data.reset_index(inplace=True)
                data.rename(columns={'index': 'Date'}, inplace=True)
                
                
                data['Symbol'] = ticker
                
                result[ticker] = data
            except Exception as e:
                print(f"Error fetching data for {ticker} from Alpha Vantage: {e}")
        
        return result
    
    def process_uploaded_csv(self, file_path: str, column_mapping: Dict[str, str] = None) -> pd.DataFrame:
        """
        Process an uploaded CSV file and map columns to the canonical format.
        
        Args:
            file_path: Path to the uploaded CSV file
            column_mapping: Dictionary mapping user columns to canonical columns
            
        Returns:
            DataFrame with standardized columns
        """
        
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file_path)
        else:
            raise ValueError("Unsupported file format. Please upload a CSV or Excel file.")
        
        
        if column_mapping is None:
            column_mapping = self._infer_column_mapping(df)
        
        
        df_mapped = self._apply_column_mapping(df, column_mapping)
        
        return df_mapped
    
    def _infer_column_mapping(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Infer column mapping from DataFrame columns.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary mapping DataFrame columns to canonical columns
        """
        
        canonical_columns = {
            'Date': ['date', 'timestamp', 'time', 'datetime'],
            'Open': ['open', 'opening', 'open_price'],
            'High': ['high', 'highest', 'high_price'],
            'Low': ['low', 'lowest', 'low_price'],
            'Close': ['close', 'closing', 'close_price'],
            'Adj_Close': ['adj_close', 'adjclose', 'adjusted_close', 'adj close', 'adjusted close'],
            'Volume': ['volume', 'vol', 'trade_volume'],
            'Symbol': ['symbol', 'ticker', 'stock', 'asset']
        }
        
        
        column_mapping = {}
        
        
        df_columns_lower = {col.lower(): col for col in df.columns}
        
        
        for canonical_col, synonyms in canonical_columns.items():
            
            if canonical_col.lower() in df_columns_lower:
                column_mapping[canonical_col] = df_columns_lower[canonical_col.lower()]
                continue
            
            
            for synonym in synonyms:
                if synonym in df_columns_lower:
                    column_mapping[canonical_col] = df_columns_lower[synonym]
                    break
        
        return column_mapping
    
    def _apply_column_mapping(self, df: pd.DataFrame, column_mapping: Dict[str, str]) -> pd.DataFrame:
        """
        Apply column mapping to DataFrame.
        
        Args:
            df: Input DataFrame
            column_mapping: Dictionary mapping canonical columns to DataFrame columns
            
        Returns:
            DataFrame with standardized columns
        """
        
        df_mapped = pd.DataFrame()
        
        
        for canonical_col, df_col in column_mapping.items():
            if df_col in df.columns:
                df_mapped[canonical_col] = df[df_col]
        
        
        required_columns = ['Date', 'Open', 'High', 'Low', 'Close']
        missing_columns = [col for col in required_columns if col not in df_mapped.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
        
        
        if 'Adj_Close' not in df_mapped.columns and 'Close' in df_mapped.columns:
            df_mapped['Adj_Close'] = df_mapped['Close']
        
        if 'Volume' not in df_mapped.columns:
            df_mapped['Volume'] = float('nan')
        
        if 'Symbol' not in df_mapped.columns:
            df_mapped['Symbol'] = 'UNKNOWN'
        
        
        df_mapped['Date'] = pd.to_datetime(df_mapped['Date'])
        
        
        df_mapped.sort_values('Date', inplace=True)
        
        return df_mapped