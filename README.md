# 📈 AI Stock Price Predictor

An intelligent machine learning application that predicts stock prices using advanced algorithms and provides real-time market analysis.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://aistockpricepredictor.streamlit.app/)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 🚀 Live Demo

**[Try the app live here!](https://aistockpricepredictor.streamlit.app/)**

## ✨ Features

- **Real-time Stock Data**: Fetches live stock market data using Yahoo Finance API
- **Multiple ML Models**: Implements various machine learning algorithms:
  - Linear Regression
  - Random Forest
  - Gradient Boosting
- **Interactive Web Interface**: User-friendly Streamlit dashboard
- **Future Price Predictions**: Generates forecasts for stock price movements
- **Beautiful Visualizations**: Interactive charts and graphs for data analysis
- **Multiple Time Periods**: Analyze stocks across different timeframes
- **Stock Symbol Support**: Works with major stock symbols (AAPL, GOOGL, TSLA, etc.)

## 🛠️ Technology Stack

- **Python**: Core programming language
- **Streamlit**: Web application framework
- **Scikit-learn**: Machine learning library
- **Yahoo Finance API**: Real-time stock data
- **Matplotlib/Seaborn**: Data visualization
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing

## 📋 Prerequisites

Before running this application, make sure you have Python 3.8+ installed on your system.

## 🔧 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/NoLongerHumanHQ/stock-price-predictor.git
   cd stock-price-predictor
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Usage

### Running Locally

1. **Start the Streamlit app**
   ```bash
   streamlit run app.py
   ```

2. **Open your browser** and navigate to `http://localhost:8501`

### Using the Application

1. **Enter a Stock Symbol**: Input a valid stock ticker (e.g., AAPL, GOOGL, TSLA, MSFT)
2. **Select Time Period**: Choose your preferred analysis timeframe
3. **Click "Analyze Stock"**: Start the prediction process
4. **View Results**: Examine predictions, charts, and analysis

## 📊 Supported Stock Symbols

The app supports major stock exchanges including:
- **NASDAQ**: AAPL, GOOGL, MSFT, TSLA, AMZN
- **NYSE**: JPM, JNJ, V, PG, UNH
- **And many more!**

## 🤖 Machine Learning Models

### Linear Regression
- Simple and interpretable model
- Good for identifying linear trends
- Fast training and prediction

### Random Forest
- Ensemble method using multiple decision trees
- Handles non-linear relationships
- Robust against overfitting

### Gradient Boosting
- Advanced ensemble technique
- High accuracy for complex patterns
- Sequential error correction

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

**Important**: This application is for educational and informational purposes only. Stock market predictions are inherently uncertain and should not be used as the sole basis for investment decisions. Always consult with a qualified financial advisor before making investment choices.

## 👨‍💻 Author

**NoLongerHumanHQ**
- GitHub: [@NoLongerHumanHQ](https://github.com/NoLongerHumanHQ)

## 🙏 Acknowledgments

- Yahoo Finance for providing stock data API
- Streamlit team for the amazing web framework
- Scikit-learn contributors for machine learning tools
- The open-source community for inspiration and support

**⭐ If you found this project helpful, please give it a star!**
