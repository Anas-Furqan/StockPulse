import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_sample_data(ticker='SAMPLE', days=365, start_date=None, volatility=0.02):
    """Generate sample stock data for testing purposes."""
    if start_date is None:
        start_date = datetime.now() - timedelta(days=days)
    
    # Generate dates
    dates = [start_date + timedelta(days=i) for i in range(days)]
    dates = [d for d in dates if d.weekday() < 5]  # Only business days
    
    # Initial price
    price = 100.0
    
    # Generate price data with random walk
    opens = [price]
    highs = [price * (1 + np.random.uniform(0, volatility))]
    lows = [price * (1 - np.random.uniform(0, volatility))]
    closes = [price * (1 + np.random.normal(0, volatility))]
    
    for i in range(1, len(dates)):
        # Previous close becomes reference for today
        prev_close = closes[i-1]
        
        # Random walk with drift
        daily_return = np.random.normal(0.0002, volatility)  # Small positive drift
        price = prev_close * (1 + daily_return)
        
        # Generate OHLC data
        open_price = price * (1 + np.random.normal(0, 0.002))
        high_price = max(open_price, price) * (1 + np.random.uniform(0, volatility))
        low_price = min(open_price, price) * (1 - np.random.uniform(0, volatility))
        close_price = price
        
        opens.append(open_price)
        highs.append(high_price)
        lows.append(low_price)
        closes.append(close_price)
    
    # Generate volume data
    base_volume = 1000000
    volumes = [int(base_volume * (1 + np.random.normal(0, 0.3))) for _ in range(len(dates))]
    volumes = [max(v, 100000) for v in volumes]  # Ensure minimum volume
    
    # Create DataFrame
    df = pd.DataFrame({
        'Date': dates,
        'Open': opens,
        'High': highs,
        'Low': lows,
        'Close': closes,
        'Adj_Close': closes,  # Simplified, assuming no adjustments
        'Volume': volumes,
        'Symbol': ticker
    })
    
    return df

def save_sample_data():
    """Generate and save sample data for multiple tickers."""
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    
    # Create data directory if it doesn't exist
    data_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Generate data for each ticker
    for ticker in tickers:
        df = generate_sample_data(ticker=ticker, days=365)
        
        # Save as CSV
        file_path = os.path.join(data_dir, f"{ticker}_daily.csv")
        df.to_csv(file_path, index=False)
        print(f"Sample data for {ticker} saved to {file_path}")

if __name__ == "__main__":
    save_sample_data()