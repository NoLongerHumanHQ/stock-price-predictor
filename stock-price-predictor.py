"""
Stock Price Predictor - AI/ML
A comprehensive machine learning application for predicting stock prices
Author: Veeru Patel
"""

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import warnings
import logging
import joblib
import os

warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StockPredictor:
    """
    A comprehensive stock price prediction system using machine learning.
    
    Features:
    - Multiple ML models (Linear Regression, Random Forest, Gradient Boosting)
    - Advanced feature engineering
    - Model persistence
    - Performance visualization
    - Next-day price predictions
    """
    
    def __init__(self, symbol='AAPL', period='2y'):
        """
        Initialize the Stock Predictor
        
        Args:
            symbol (str): Stock ticker symbol (default: AAPL)
            period (str): Historical data period (default: 2y)
        """
        self.symbol = symbol.upper()
        self.period = period
        self.data = None
        self.features = None
        self.target = None
        self.models = {}
        self.best_model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.results = {}
        
        # Create models directory
        os.makedirs('models', exist_ok=True)
        
    def fetch_data(self):
        """Download and validate stock data from Yahoo Finance"""
        logger.info(f"📈 Fetching data for {self.symbol}...")
        
        try:
            stock = yf.Ticker(self.symbol)
            info = stock.info
            
            # Validate ticker
            if not info or 'symbol' not in info:
                raise ValueError(f"Invalid ticker symbol: {self.symbol}")
            
            self.data = stock.history(period=self.period)
            
            if self.data.empty:
                raise ValueError(f"No data available for {self.symbol}")
            
            # Add company info
            self.company_name = info.get('longName', self.symbol)
            self.sector = info.get('sector', 'Unknown')
            
            logger.info(f"✅ Successfully downloaded {len(self.data)} days of data for {self.company_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error fetching data: {e}")
            return False
    
    def create_features(self):
        """Create comprehensive features from raw stock data"""
        logger.info("🔧 Creating features...")
        
        df = self.data.copy()
        
        # Basic price features
        df['Price_Change'] = df['Close'].pct_change()
        df['High_Low_Pct'] = (df['High'] - df['Low']) / df['Close']
        df['Price_Range'] = df['High'] - df['Low']
        df['Open_Close_Pct'] = (df['Close'] - df['Open']) / df['Open']
        
        # Moving averages (multiple timeframes)
        for window in [5, 10, 20, 50]:
            df[f'MA_{window}'] = df['Close'].rolling(window=window).mean()
            df[f'Price_vs_MA{window}'] = df['Close'] / df[f'MA_{window}']
        
        # Exponential Moving Averages
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        
        # Volatility measures
        for window in [5, 10, 20]:
            df[f'Volatility_{window}'] = df['Close'].rolling(window=window).std()
            df[f'Volatility_Pct_{window}'] = df[f'Volatility_{window}'] / df['Close']
        
        # Volume features
        df['Volume_MA'] = df['Volume'].rolling(window=10).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
        df['Price_Volume'] = df['Close'] * df['Volume']
        
        # Technical indicators
        # RSI (Relative Strength Index)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        bb_window = 20
        bb_std = df['Close'].rolling(window=bb_window).std()
        df['BB_Middle'] = df['Close'].rolling(window=bb_window).mean()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # Lagged features
        for lag in [1, 2, 3, 5]:
            df[f'Close_Lag{lag}'] = df['Close'].shift(lag)
            df[f'Volume_Lag{lag}'] = df['Volume'].shift(lag)
            df[f'Price_Change_Lag{lag}'] = df['Price_Change'].shift(lag)
        
        # Day of week and month effects
        df['DayOfWeek'] = df.index.dayofweek
        df['Month'] = df.index.month
        df['Quarter'] = df.index.quarter
        
        # Target variable (next day's closing price)
        df['Target'] = df['Close'].shift(-1)
        
        # Select feature columns
        self.feature_columns = [
            'Open', 'High', 'Low', 'Volume',
            'Price_Change', 'High_Low_Pct', 'Price_Range', 'Open_Close_Pct',
            'MA_5', 'MA_10', 'MA_20', 'MA_50',
            'Price_vs_MA5', 'Price_vs_MA10', 'Price_vs_MA20', 'Price_vs_MA50',
            'EMA_12', 'EMA_26', 'MACD',
            'Volatility_5', 'Volatility_10', 'Volatility_20',
            'Volume_Ratio', 'Price_Volume',
            'RSI', 'BB_Position',
            'Close_Lag1', 'Close_Lag2', 'Close_Lag3', 'Close_Lag5',
            'Volume_Lag1', 'Price_Change_Lag1',
            'DayOfWeek', 'Month', 'Quarter'
        ]
        
        # Clean data (remove rows with NaN values)
        df_clean = df[self.feature_columns + ['Target']].dropna()
        
        self.features = df_clean[self.feature_columns]
        self.target = df_clean['Target']
        
        logger.info(f"✅ Created {len(self.feature_columns)} features with {len(df_clean)} samples")
        
    def train_models(self):
        """Train multiple machine learning models with hyperparameter tuning"""
        logger.info("🤖 Training multiple models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.features, self.target, test_size=0.2, random_state=42, shuffle=False
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Define models with hyperparameters
        model_configs = {
            'linear': {
                'model': LinearRegression(),
                'params': {}
            },
            'random_forest': {
                'model': RandomForestRegressor(random_state=42),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5]
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingRegressor(random_state=42),
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.05, 0.1],
                    'max_depth': [3, 5]
                }
            }
        }
        
        best_score = -np.inf
        
        for name, config in model_configs.items():
            logger.info(f"Training {name}...")
            
            if config['params']:
                # Use GridSearchCV for hyperparameter tuning
                grid_search = GridSearchCV(
                    config['model'], config['params'], 
                    cv=3, scoring='r2', n_jobs=-1
                )
                grid_search.fit(X_train_scaled, y_train)
                model = grid_search.best_estimator_
            else:
                model = config['model']
                model.fit(X_train_scaled, y_train)
            
            # Make predictions
            train_pred = model.predict(X_train_scaled)
            test_pred = model.predict(X_test_scaled)
            
            # Calculate metrics
            metrics = {
                'train_mae': mean_absolute_error(y_train, train_pred),
                'test_mae': mean_absolute_error(y_test, test_pred),
                'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
                'test_rmse': np.sqrt(mean_squared_error(y_test, test_pred)),
                'train_r2': r2_score(y_train, train_pred),
                'test_r2': r2_score(y_test, test_pred)
            }
            
            self.models[name] = {
                'model': model,
                'metrics': metrics,
                'predictions': {'train': train_pred, 'test': test_pred}
            }
            
            # Track best model
            if metrics['test_r2'] > best_score:
                best_score = metrics['test_r2']
                self.best_model = name
        
        # Store training data for visualization
        self.train_data = {
            'X_train': X_train, 'X_test': X_test,
            'y_train': y_train, 'y_test': y_test,
            'X_train_scaled': X_train_scaled, 'X_test_scaled': X_test_scaled
        }
        
        self._print_model_comparison()
        
    def _print_model_comparison(self):
        """Print comparison of all models"""
        logger.info("📊 Model Performance Comparison:")
        print("\n" + "="*80)
        print(f"{'Model':<20} {'Test MAE':<12} {'Test RMSE':<12} {'Test R²':<10} {'Status'}")
        print("="*80)
        
        for name, results in self.models.items():
            metrics = results['metrics']
            status = "⭐ BEST" if name == self.best_model else ""
            print(f"{name:<20} ${metrics['test_mae']:<11.2f} ${metrics['test_rmse']:<11.2f} {metrics['test_r2']:<9.3f} {status}")
        print("="*80)
    
    def save_model(self, model_name=None):
        """Save the best model and scaler"""
        if model_name is None:
            model_name = self.best_model
            
        model_path = f"models/{self.symbol}_{model_name}_model.joblib"
        scaler_path = f"models/{self.symbol}_scaler.joblib"
        
        joblib.dump(self.models[model_name]['model'], model_path)
        joblib.dump(self.scaler, scaler_path)
        
        logger.info(f"💾 Model saved: {model_path}")
        return model_path
    
    def load_model(self, model_path, scaler_path):
        """Load a saved model and scaler"""
        self.models['loaded'] = {'model': joblib.load(model_path)}
        self.scaler = joblib.load(scaler_path)
        self.best_model = 'loaded'
        logger.info(f"📂 Model loaded: {model_path}")
    
    def plot_comprehensive_analysis(self):
        """Create comprehensive visualizations"""
        logger.info("📈 Creating comprehensive visualizations...")
        
        plt.style.use('seaborn-v0_8')
        fig = plt.figure(figsize=(20, 16))
        
        # Get best model data
        best_results = self.models[self.best_model]
        test_pred = best_results['predictions']['test']
        y_test = self.train_data['y_test']
        
        # 1. Actual vs Predicted
        ax1 = plt.subplot(3, 3, 1)
        plt.scatter(y_test, test_pred, alpha=0.6, color='blue')
        min_val, max_val = y_test.min(), y_test.max()
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        plt.xlabel('Actual Price ($)')
        plt.ylabel('Predicted Price ($)')
        plt.title(f'Actual vs Predicted ({self.best_model.title()})')
        plt.grid(True, alpha=0.3)
        
        # Add R² score
        r2 = best_results['metrics']['test_r2']
        plt.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax1.transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        # 2. Time series comparison
        ax2 = plt.subplot(3, 3, 2)
        test_dates = self.train_data['X_test'].index
        plt.plot(test_dates, y_test.values, label='Actual', linewidth=2, color='blue')
        plt.plot(test_dates, test_pred, label='Predicted', linewidth=2, color='red')
        plt.xlabel('Date')
        plt.ylabel('Stock Price ($)')
        plt.title('Price Prediction Timeline')
        plt.legend()
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # 3. Feature importance (if available)
        ax3 = plt.subplot(3, 3, 3)
        if hasattr(self.models[self.best_model]['model'], 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.models[self.best_model]['model'].feature_importances_
            }).sort_values('importance', ascending=True).tail(15)
            
            plt.barh(range(len(importance_df)), importance_df['importance'])
            plt.yticks(range(len(importance_df)), importance_df['feature'])
            plt.xlabel('Feature Importance')
            plt.title('Top 15 Feature Importances')
            plt.grid(True, alpha=0.3)
        else:
            plt.text(0.5, 0.5, 'Feature importance\nnot available for\nthis model type', 
                    ha='center', va='center', transform=ax3.transAxes)
            plt.title('Feature Importance')
        
        # 4. Prediction errors
        ax4 = plt.subplot(3, 3, 4)
        errors = y_test - test_pred
        plt.hist(errors, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(x=0, color='red', linestyle='--', alpha=0.8)
        plt.xlabel('Prediction Error ($)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Prediction Errors')
        plt.grid(True, alpha=0.3)
        
        # Add error statistics
        mae = np.mean(np.abs(errors))
        plt.text(0.05, 0.95, f'MAE = ${mae:.2f}', transform=ax4.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
        
        # 5. Model comparison
        ax5 = plt.subplot(3, 3, 5)
        model_names = list(self.models.keys())
        r2_scores = [self.models[name]['metrics']['test_r2'] for name in model_names]
        colors = ['gold' if name == self.best_model else 'lightblue' for name in model_names]
        
        bars = plt.bar(model_names, r2_scores, color=colors)
        plt.ylabel('R² Score')
        plt.title('Model Performance Comparison')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Add values on bars
        for bar, score in zip(bars, r2_scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')
        
        # 6. Stock price history
        ax6 = plt.subplot(3, 3, 6)
        plt.plot(self.data.index, self.data['Close'], linewidth=1.5, color='purple')
        plt.xlabel('Date')
        plt.ylabel('Stock Price ($)')
        plt.title(f'{self.symbol} Price History ({self.period})')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # 7. Volume analysis
        ax7 = plt.subplot(3, 3, 7)
        plt.plot(self.data.index, self.data['Volume'], linewidth=1, color='orange')
        plt.xlabel('Date')
        plt.ylabel('Volume')
        plt.title('Volume History')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # 8. Volatility analysis
        ax8 = plt.subplot(3, 3, 8)
        volatility = self.data['Close'].rolling(window=20).std()
        plt.plot(self.data.index, volatility, linewidth=1.5, color='red')
        plt.xlabel('Date')
        plt.ylabel('20-Day Volatility')
        plt.title('Price Volatility')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # 9. Summary statistics
        ax9 = plt.subplot(3, 3, 9)
        ax9.axis('off')
        
        summary_text = f"""
        📊 ANALYSIS SUMMARY
        
        Company: {getattr(self, 'company_name', self.symbol)}
        Sector: {getattr(self, 'sector', 'Unknown')}
        
        📈 Current Price: ${self.data['Close'].iloc[-1]:.2f}
        📊 Data Points: {len(self.data)}
        🎯 Best Model: {self.best_model.title()}
        
        🔍 Performance Metrics:
        • R² Score: {best_results['metrics']['test_r2']:.3f}
        • MAE: ${best_results['metrics']['test_mae']:.2f}
        • RMSE: ${best_results['metrics']['test_rmse']:.2f}
        
        ⚠️  Disclaimer: For educational use only.
        Not financial advice.
        """
        
        ax9.text(0.1, 0.9, summary_text, transform=ax9.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", 
                facecolor="lightgray", alpha=0.8))
        
        plt.suptitle(f'{self.symbol} Stock Price Prediction Analysis', fontsize=20, y=0.98)
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        plt.show()
    
    def predict_next_days(self, days=5):
        """Predict stock prices for the next N days"""
        logger.info(f"🔮 Predicting next {days} days for {self.symbol}...")
        
        predictions = []
        current_features = self.features.iloc[-1:].copy()
        
        for day in range(days):
            # Scale features
            features_scaled = self.scaler.transform(current_features.values)
            
            # Make prediction
            prediction = self.models[self.best_model]['model'].predict(features_scaled)[0]
            predictions.append(prediction)
            
            # Update features for next prediction (simplified approach)
            # In practice, you'd want more sophisticated feature updating
            next_features = current_features.copy()
            next_features.iloc[0, next_features.columns.get_loc('Close_Lag1')] = prediction
            current_features = next_features
        
        # Display predictions
        current_price = self.data['Close'].iloc[-1]
        print(f"\n📈 Multi-day Predictions for {self.symbol}:")
        print(f"Current Price: ${current_price:.2f}")
        print("-" * 50)
        
        for i, pred in enumerate(predictions, 1):
            change = pred - current_price
            change_pct = (change / current_price) * 100
            print(f"Day +{i}: ${pred:.2f} (Change: ${change:+.2f}, {change_pct:+.1f}%)")
        
        return predictions
    
    def run_complete_analysis(self):
        """Execute the complete stock prediction pipeline"""
        logger.info(f"🚀 Starting Complete Analysis for {self.symbol}")
        print("=" * 80)
        
        try:
            # Step 1: Fetch data
            if not self.fetch_data():
                return False
            
            # Step 2: Create features
            self.create_features()
            
            # Step 3: Train models
            self.train_models()
            
            # Step 4: Save best model
            self.save_model()
            
            # Step 5: Create visualizations
            self.plot_comprehensive_analysis()
            
            # Step 6: Make predictions
            self.predict_next_days(5)
            
            print("=" * 80)
            print("✅ Complete Analysis Finished Successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Analysis failed: {e}")
            return False

def compare_stocks(symbols=['AAPL', 'GOOGL', 'TSLA', 'MSFT'], period='1y'):
    """Compare prediction accuracy across multiple stocks"""
    logger.info("📊 Comparing Multiple Stocks...")
    
    results = []
    for symbol in symbols:
        try:
            predictor = StockPredictor(symbol=symbol, period=period)
            if predictor.fetch_data():
                predictor.create_features()
                predictor.train_models()
                
                best_metrics = predictor.models[predictor.best_model]['metrics']
                results.append({
                    'Symbol': symbol,
                    'Best_Model': predictor.best_model.title(),
                    'R²': best_metrics['test_r2'],
                    'MAE': best_metrics['test_mae'],
                    'RMSE': best_metrics['test_rmse'],
                    'Rating': 'Excellent' if best_metrics['test_r2'] > 0.8 else 
                             'Good' if best_metrics['test_r2'] > 0.6 else 
                             'Fair' if best_metrics['test_r2'] > 0.4 else 'Poor'
                })
        except Exception as e:
            logger.error(f"Failed to analyze {symbol}: {e}")
    
    if results:
        comparison_df = pd.DataFrame(results).sort_values('R²', ascending=False)
        print("\n📊 STOCK COMPARISON RESULTS")
        print("=" * 80)
        print(comparison_df.to_string(index=False, float_format='%.3f'))
        return comparison_df
    
    return None

# Main execution
if __name__ == "__main__":
    print("Stock Price Predictor")
    print("Portfolio Project - Machine Learning for Finance")
    print("=" * 60)
    
    # Example 1: Single stock analysis
    print("\n1️⃣ Running Single Stock Analysis...")
    predictor = StockPredictor(symbol='AAPL', period='2y')
    success = predictor.run_complete_analysis()
    
    if success:
        print("\n✅ Single stock analysis completed!")
        
        # Example 2: Multi-stock comparison
        print("\n2️⃣ Running Multi-Stock Comparison...")
        comparison_results = compare_stocks(['AAPL', 'GOOGL', 'TSLA', 'MSFT'])
        
        print("\n🎉 All analyses completed successfully!")
        print("\n💡 To use with different stocks:")
        print("predictor = StockPredictor(symbol='YOUR_SYMBOL', period='1y')")
        print("predictor.run_complete_analysis()")
    else:
        print("❌ Analysis failed to complete")

def create_project_summary():
    """Generate project summary"""
    summary = """
    🚀 STOCK PRICE PREDICTOR
    
    📋 PROJECT OVERVIEW:
    A comprehensive machine learning system for stock price prediction with
    professional-grade features and visualizations.
    
    🛠️ TECHNICAL STACK:
    • Python, pandas, numpy, scikit-learn
    • Yahoo Finance API integration
    • Multiple ML algorithms (Linear Regression, Random Forest, Gradient Boosting)
    • Advanced feature engineering & hyperparameter tuning
    • Model persistence & deployment-ready code
    
    🎯 KEY FEATURES:
    • Automated data collection & validation
    • 30+ engineered features (technical indicators, moving averages, etc.)
    • Multi-model training with automatic best model selection
    • Comprehensive performance analysis & visualization
    • Multi-day price predictions
    • Model saving/loading for production use
    • Cross-stock comparison capabilities
    
    📊 SKILLS DEMONSTRATED:
    • Data Science & Machine Learning
    • Financial Market Analysis
    • Feature Engineering & Selection
    • Model Evaluation & Validation
    • Data Visualization & Interpretation
    • Code Architecture & Documentation
    • Error Handling & Logging
    
    ⚠️ DISCLAIMER: Educational purposes only. Not financial advice.
    """
    print(summary)

# Uncomment to show project summary
# create_project_summary()
