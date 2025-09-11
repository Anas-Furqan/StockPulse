# StockPulse - Anomaly Tracker

A comprehensive tool for detecting anomalies in stock market data using machine learning techniques.

## Features

- Data fetching from Yahoo Finance and Alpha Vantage
- CSV/Excel file upload support
- Exploratory Data Analysis (EDA) with interactive visualizations
- Multiple anomaly detection algorithms:
  - Isolation Forest
  - Local Outlier Factor (LOF)
  - Autoencoder Neural Network
- Model parameter tuning and comparison
- Export functionality for data and results
- Containerized deployment with Docker

## Installation

### Local Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/StockPulse.git
   cd StockPulse
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file based on `.env.example` and add your API keys.

5. Run the application:
   ```bash
   streamlit run app/main.py
   ```

### Docker Installation

1. Build and run using Docker Compose:
   ```bash
   docker-compose up --build
   ```

2. Access the application at http://localhost:8501

## Usage

1. **Overview Page**: Select a data source (Yahoo Finance, Alpha Vantage, or upload your own CSV) and load the data.
2. **EDA Page**: Explore the loaded data with visualizations and statistics.
3. **Anomalies Page**: Run anomaly detection algorithms and view the results.
4. **Model Tuning Page**: Adjust parameters for the anomaly detection models.
5. **Export Page**: Download the cleaned data and anomaly reports.

## Testing

To run the tests:

```bash
# Run all tests
python -m pytest tests/

# Run tests with coverage report
python -m pytest tests/ --cov=app

# Run a specific test file
python -m pytest tests/test_models.py
```

To generate sample data for testing:

```bash
python data/sample_stock_data.py
```

## Project Structure
StockPulse/
├── app/
│   ├── api/
│   ├── main.py
│   ├── src/
│   │   ├── data/
│   │   ├── evaluation/
│   │   ├── models/
│   │   ├── preprocessing/
│   │   └── reports/
│   └── ui/
│       ├── components.py
│       └── streamlit_pages.py
├── data/
├── exports/
├── tests/
├── .env.example
├── Dockerfile
├── docker-compose.yml
└── requirements.txt

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit your changes: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin feature-name`
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.