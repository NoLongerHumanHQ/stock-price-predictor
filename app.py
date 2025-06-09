"""
Stock Price Predictor Web Application
Streamlit interface for the ML stock prediction system
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Stock Price Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.25rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class StreamlitStockPredictor:
    """Streamlit-optimized version of the stock predictor"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = None
        self.features = None
        self.target = None
        
    @st.cache_data
    def fetch_stock_data(_self, symbol, period):
        """Fetch stock data with caching"""
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period=period)
            info = stock.info
            return data, info, None
        except Exception as e:
            return None, None, str(e)
    
    def create_features(self, data):
        """Create features for prediction"""
        df = data.copy()
        
        # Basic features
        df['Price_Change'] = df['Close'].pct_change()
        df['High_Low_Pct'] = (df['High'] - df['Low']) / df['Close']
        df['Volume_MA'] = df['Volume'].rolling(window=10).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
        
        # Moving averages
        for window in [5, 10, 20]:
            df[f'MA_{window}'] = df['Close'].rolling(window=window).mean()
            df[f'Price_vs_MA{window}'] = df['Close'] / df[f'MA_{window}']
        
        # Volatility
        df['Volatility_10'] = df['Close'].rolling(window=10).std()
        
        # Lagged features
        df['Close_Lag1'] = df['Close'].shift(1)
        df['Close_Lag2'] = df['Close'].shift(2)
        df['Volume_Lag1'] = df['Volume'].shift(1)
        
        # Technical indicators
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Target
        df['Target'] = df['Close'].shift(-1)
        
        # Feature columns
        feature_cols = [
            'Open', 'High', 'Low', 'Volume', 'Price_Change', 'High_Low_Pct',
            'MA_5', 'MA_10', 'MA_20', 'Price_vs_MA5', 'Price_vs_MA10', 'Price_vs_MA20',
            'Volatility_10', 'Volume_Ratio', 'Close_Lag1', 'Close_Lag2', 'Volume_Lag1', 'RSI'
        ]
        
        df_clean = df[feature_cols + ['Target']].dropna()
        return df_clean[feature_cols], df_clean['Target']
    
    def train_model(self, features, target, model_type='random_forest'):
        """Train the selected model"""
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=0.2, random_state=42, shuffle=False
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        if model_type == 'random_forest':
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            self.model = LinearRegression()
        
        self.model.fit(X_train_scaled, y_train)
        
        train_pred = self.model.predict(X_train_scaled)
        test_pred = self.model.predict(X_test_scaled)
        
        metrics = {
            'train_mae': mean_absolute_error(y_train, train_pred),
            'test_mae': mean_absolute_error(y_test, test_pred),
            'train_r2': r2_score(y_train, train_pred),
            'test_r2': r2_score(y_test, test_pred)
        }
        
        return metrics, X_test.index, y_test, test_pred
    
    def predict_next_price(self, features):
        """Predict next day's price"""
        if self.model is None:
            return None
        
        latest_features = features.iloc[-1:].values
        latest_scaled = self.scaler.transform(latest_features)
        prediction = self.model.predict(latest_scaled)[0]
        return prediction

def main():
    # Header
    st.markdown('<h1 class="main-header">📈 Stock Price Predictor</h1>', unsafe_allow_html=True)
    st.markdown("**Powered by Machine Learning | Built for Educational Purposes**")
    
    # Sidebar
    st.sidebar.header("🎛️ Configuration")
    
    # Stock selection
    symbol = st.sidebar.text_input("Stock Symbol", value="AAPL", help="Enter a valid stock ticker (e.g., AAPL, GOOGL, TSLA)")
    period = st.sidebar.selectbox("Data Period", ["1y", "2y", "5y", "max"], index=1)
    model_type = st.sidebar.selectbox("Model Type", ["random_forest", "linear_regression"])
    
    # Initialize predictor
    predictor = StreamlitStockPredictor()
    
    if st.sidebar.button("🚀 Run Analysis", type="primary"):
        with st.spinner(f"Analyzing {symbol.upper()}..."):
            # Fetch data
            data, info, error = predictor.fetch_stock_data(symbol.upper(), period)
            
            if error:
                st.error(f"❌ Error fetching data: {error}")
                return
            
            if data is None or data.empty:
                st.error(f"❌ No data found for {symbol.upper()}")
                return
            
            # Store in session state
            st.session_state['data'] = data
            st.session_state['info'] = info
            st.session_state['symbol'] = symbol.upper()
            
            # Create features
            features, target = predictor.create_features(data)
            
            # Train model
            metrics, test_dates, y_test, test_pred = predictor.train_model(features, target, model_type)
            
            # Store results
            st.session_state['predictor'] = predictor
            st.session_state['features'] = features
            st.session_state['metrics'] = metrics
            st.session_state['test_results'] = (test_dates, y_test, test_pred)
            
            st.success("✅ Analysis completed successfully!")
    
    # Display results if available
    if 'data' in st.session_state:
        data = st.session_state['data']
        info = st.session_state['info']
        symbol = st.session_state['symbol']
        
        # Company info
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Company", info.get('longName', symbol)[:20] + "..." if len(info.get('longName', symbol)) > 20 else info.get('longName', symbol))
        
        with col2:
            current_price = data['Close'].iloc[-1]
            prev_price = data['Close'].iloc[-2]
            change = current_price - prev_price
            st.metric("Current Price", f"${current_price:.2f}", f"${change:.2f}")
        
        with col3:
            st.metric("Data Points", len(data))
        
        with col4:
            st.metric("Sector", info.get('sector', 'Unknown'))
        
        # Price chart
        st.subheader("📊 Price History")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Close'],
            mode='lines',
            name='Close Price',
            line=dict(color='#1f77b4', width=2)
        ))
        fig.update_layout(
            title=f"{symbol} Stock Price History",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Model results
        if 'metrics' in st.session_state:
            metrics = st.session_state['metrics']
            test_dates, y_test, test_pred = st.session_state['test_results']
            
            st.subheader("🤖 Model Performance")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Test R² Score", f"{metrics['test_r2']:.3f}")
            with col2:
                st.metric("Test MAE", f"${metrics['test_mae']:.2f}")
            with col3:
                rating = "Excellent" if metrics['test_r2'] > 0.8 else "Good" if metrics['test_r2'] > 0.6 else "Fair" if metrics['test_r2'] > 0.4 else "Poor"
                st.metric("Model Rating", rating)
            with col4:
                st.metric("Model Type", model_type.replace('_', ' ').title())
            
            # Prediction vs Actual chart
            st.subheader("🎯 Predictions vs Actual")
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=test_dates,
                y=y_test.values,
                mode='lines',
                name='Actual',
                line=dict(color='blue', width=2)
            ))
            fig2.add_trace(go.Scatter(
                x=test_dates,
                y=test_pred,
                mode='lines',
                name='Predicted',
                line=dict(color='red', width=2, dash='dash')
            ))
            fig2.update_layout(
                title="Actual vs Predicted Prices",
                xaxis_title="Date",
                yaxis_title="Price ($)",
                hovermode='x unified'
            )
            st.plotly_chart(fig2, use_container_width=True)
            
            # Next day prediction
            predictor = st.session_state['predictor']
            features = st.session_state['features']
            next_price = predictor.predict_next_price(features)
            
            if next_price:
                st.subheader("🔮 Next Day Prediction")
                current_price = data['Close'].iloc[-1]
                change = next_price - current_price
                change_pct = (change / current_price) * 100
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Current Price", f"${current_price:.2f}")
                with col2:
                    st.metric("Predicted Price", f"${next_price:.2f}", f"${change:.2f}")
                with col3:
                    st.metric("Expected Change", f"{change_pct:.1f}%")
                
                # Confidence indicator
                confidence = "High" if metrics['test_r2'] > 0.7 else "Medium" if metrics['test_r2'] > 0.5 else "Low"
                st.info(f"💡 **Prediction Confidence:** {confidence} (based on R² score of {metrics['test_r2']:.3f})")
        
        # Volume analysis
        st.subheader("📈 Volume Analysis")
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=data.index,
            y=data['Volume'],
            mode='lines',
            name='Volume',
            line=dict(color='orange', width=1)
        ))
        fig3.update_layout(
            title=f"{symbol} Trading Volume",
            xaxis_title="Date",
            yaxis_title="Volume",
            hovermode='x unified'
        )
        st.plotly_chart(fig3, use_container_width=True)
    
    # Disclaimer
    st.markdown("---")
    st.markdown("""
    **⚠️ Important Disclaimer:**
    This tool is for educational purposes only and should not be used as the sole basis for investment decisions. 
    Stock market investments carry inherent risks, and past performance does not guarantee future results. 
    Always consult with qualified financial advisors before making investment decisions.
    """)
    
    # Footer
    st.markdown("---")
    st.markdown("**Built with ❤️ using Streamlit and Machine Learning**")

if __name__ == "__main__":
    main()