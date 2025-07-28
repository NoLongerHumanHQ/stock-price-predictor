# Stock Price Predictor

A machine learning application that predicts stock prices using advanced algorithms and provides real-time analysis through an interactive web interface.

## Features

- **Real-time Stock Data Fetching**: Get up-to-date stock information from Yahoo Finance API
- **Multiple ML Models**: Choose from Linear Regression, Random Forest, and Gradient Boosting algorithms
- **Interactive Web Interface**: User-friendly Streamlit-based dashboard
- **Future Price Predictions**: Generate forecasts based on historical data patterns
- **Beautiful Visualizations**: Comprehensive charts and graphs for data analysis
- **Customizable Time Periods**: Select different timeframes for analysis

## Technology Stack

- **Python**: Core programming language
- **Streamlit**: Web application framework
- **Scikit-learn**: Machine learning algorithms
- **Yahoo Finance API**: Real-time stock data source
- **Matplotlib/Seaborn**: Data visualization libraries

## Installation

1. Clone the repository:
```bash
git clone https://github.com/NoLongerHumanHQ/stock-price-predictor.git
cd stock-price-predictor
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Streamlit application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the provided local URL (typically `http://localhost:8501`)

3. Using the application:
   - Enter a stock symbol (e.g., AAPL, GOOGL, TSLA, MSFT)
   - Select your desired time period for analysis
   - Click "Analyze Stock" to generate predictions
   - View the results including predictions, visualizations, and analysis

## Machine Learning Models

The application implements three different machine learning algorithms:

### Linear Regression
- Simple and interpretable model
- Good for identifying linear trends
- Fast training and prediction times

### Random Forest
- Ensemble method using multiple decision trees
- Handles non-linear relationships well
- Robust against overfitting

### Gradient Boosting
- Sequential ensemble method
- High predictive accuracy
- Excellent for complex pattern recognition

## Project Structure

```
stock-price-predictor/
├── app.py                 # Main Streamlit application
├── models/                # Machine learning models
├── data/                  # Data processing utilities
├── utils/                 # Helper functions
├── requirements.txt       # Python dependencies
└── README.md             # Project documentation
```

## Data Sources

The application fetches real-time stock data using the Yahoo Finance API, which provides:
- Historical stock prices
- Volume data
- Market indicators
- Company information

## Disclaimer

This application is for educational and research purposes only. Stock price predictions are inherently uncertain and should not be used as the sole basis for investment decisions. Always consult with financial professionals and conduct thorough research before making investment choices.

**Important**: Past performance does not guarantee future results. Stock markets are volatile and unpredictable.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

If you encounter any issues or have questions:
- Open an issue on GitHub
- Check the documentation
- Review existing issues for similar problems

## Acknowledgments

- Yahoo Finance for providing free stock data API
- Scikit-learn community for excellent machine learning tools
- Streamlit team for the intuitive web framework
