import unittest
import pandas as pd
import os
import sys
from unittest.mock import patch, MagicMock

# Add the app directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.src.data.fetcher import DataFetcher

class TestDataFetcher(unittest.TestCase):
    
    def setUp(self):
        self.fetcher = DataFetcher()
    
    @patch('app.src.data.fetcher.yf.download')
    def test_fetch_from_yahoo(self, mock_download):
        # Mock the yfinance download function
        mock_data = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [105, 106, 107],
            'Low': [95, 96, 97],
            'Close': [102, 103, 104],
            'Adj Close': [102, 103, 104],
            'Volume': [1000, 1100, 1200]
        })
        mock_download.return_value = mock_data
        
        # Test fetching data
        result = self.fetcher.fetch_from_yahoo(tickers=['AAPL'], period='1mo')
        
        # Assertions
        self.assertIsInstance(result, dict)
        self.assertIn('AAPL', result)
        self.assertIsInstance(result['AAPL'], pd.DataFrame)
        self.assertEqual(len(result['AAPL']), 3)
        
        # Verify the mock was called with expected parameters
        mock_download.assert_called_once()
    
    @patch('app.src.data.fetcher.requests.get')
    def test_fetch_from_alpha_vantage(self, mock_get):
        # Mock the response from Alpha Vantage API
        mock_response = MagicMock()
        mock_response.json.return_value = {
            'Time Series (Daily)': {
                '2023-01-01': {
                    '1. open': '100',
                    '2. high': '105',
                    '3. low': '95',
                    '4. close': '102',
                    '5. volume': '1000'
                },
                '2023-01-02': {
                    '1. open': '101',
                    '2. high': '106',
                    '3. low': '96',
                    '4. close': '103',
                    '5. volume': '1100'
                }
            }
        }
        mock_get.return_value = mock_response
        
        # Test fetching data
        result = self.fetcher.fetch_from_alpha_vantage(tickers=['AAPL'], output_size='compact')
        
        # Assertions
        self.assertIsInstance(result, dict)
        self.assertIn('AAPL', result)
        self.assertIsInstance(result['AAPL'], pd.DataFrame)
        
        # Verify the mock was called
        mock_get.assert_called()

if __name__ == '__main__':
    unittest.main()