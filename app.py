import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import os
import warnings
warnings.filterwarnings('ignore')

# Try to import tensorflow/keras with error handling
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    KERAS_AVAILABLE = True
except ImportError as e:
    st.error(f"TensorFlow/Keras not available: {e}")
    KERAS_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Stock Trend Predictor",
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
    .stButton > button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_stock_data(ticker, period="5y"):
    """Load stock data with error handling"""
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period=period)
        if data.empty:
            st.error(f"No data found for ticker {ticker}")
            return None
        return data
    except Exception as e:
        st.error(f"Error loading data for {ticker}: {str(e)}")
        return None

def calculate_moving_averages(data):
    """Calculate moving averages"""
    data = data.copy()
    data['MA_100'] = data['Close'].rolling(window=100).mean()
    data['MA_200'] = data['Close'].rolling(window=200).mean()
    return data

def create_price_chart(data, ticker):
    """Create interactive price chart"""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(f'{ticker} Stock Price', 'Volume'),
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3]
    )
    
    # Price chart
    fig.add_trace(
        go.Scatter(x=data.index, y=data['Close'], name='Close Price', line=dict(color='blue')),
        row=1, col=1
    )
    
    if 'MA_100' in data.columns and not data['MA_100'].isna().all():
        fig.add_trace(
            go.Scatter(x=data.index, y=data['MA_100'], name='MA 100', line=dict(color='orange')),
            row=1, col=1
        )
    
    if 'MA_200' in data.columns and not data['MA_200'].isna().all():
        fig.add_trace(
            go.Scatter(x=data.index, y=data['MA_200'], name='MA 200', line=dict(color='red')),
            row=1, col=1
        )
    
    # Volume chart
    fig.add_trace(
        go.Bar(x=data.index, y=data['Volume'], name='Volume', marker_color='lightblue'),
        row=2, col=1
    )
    
    fig.update_layout(
        title=f'{ticker} Stock Analysis',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        height=600,
        showlegend=True
    )
    
    return fig

def prepare_lstm_data(data, lookback=100):
    """Prepare data for LSTM model"""
    if len(data) < lookback:
        st.error(f"Not enough data. Need at least {lookback} days.")
        return None, None, None
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
    
    # Create sequences
    x_data, y_data = [], []
    for i in range(lookback, len(scaled_data)):
        x_data.append(scaled_data[i-lookback:i, 0])
        y_data.append(scaled_data[i, 0])
    
    return np.array(x_data), np.array(y_data), scaler

def create_sample_model():
    """Create a simple LSTM model for demonstration"""
    if not KERAS_AVAILABLE:
        return None
    
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(100, 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=True),
        Dropout(0.2),
        LSTM(50, return_sequences=True),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def main():
    # Header
    st.markdown('<h1 class="main-header">📈 Stock Trend Predictor</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("Configuration")
    
    # Stock selection
    default_stocks = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'NVDA', 'META']
    
    ticker = st.sidebar.text_input("Enter Stock Ticker:", value="AAPL").upper()
    
    # Quick select buttons
    st.sidebar.write("Quick Select:")
    cols = st.sidebar.columns(2)
    for i, stock in enumerate(default_stocks):
        if cols[i % 2].button(stock, key=f"stock_{stock}"):
            ticker = stock
            st.rerun()
    
    # Time period selection
    period = st.sidebar.selectbox(
        "Select Time Period:",
        ["1y", "2y", "5y", "10y", "max"],
        index=2
    )
    
    # Load data
    if ticker:
        with st.spinner(f"Loading data for {ticker}..."):
            data = load_stock_data(ticker, period)
        
        if data is not None and not data.empty:
            # Calculate moving averages
            data = calculate_moving_averages(data)
            
            # Display basic info
            col1, col2, col3, col4 = st.columns(4)
            
            current_price = data['Close'].iloc[-1]
            prev_price = data['Close'].iloc[-2] if len(data) > 1 else current_price
            change = current_price - prev_price
            
            with col1:
                st.metric("Current Price", f"${current_price:.2f}")
            with col2:
                st.metric("Daily Change", f"${change:.2f}", f"{change:.2f}")
            with col3:
                st.metric("Volume", f"{data['Volume'].iloc[-1]:,.0f}")
            with col4:
                st.metric("52W High", f"${data['High'].max():.2f}")
            
            # Price chart
            st.plotly_chart(create_price_chart(data, ticker), use_container_width=True)
            
            # Moving averages analysis
            st.subheader("📊 Moving Averages Analysis")
            
            if len(data) >= 200:
                ma_100 = data['MA_100'].iloc[-1]
                ma_200 = data['MA_200'].iloc[-1]
                
                if not pd.isna(ma_100) and not pd.isna(ma_200):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**100-day MA:** ${ma_100:.2f}")
                        if current_price > ma_100:
                            st.success("Price is above 100-day MA (Bullish)")
                        else:
                            st.warning("Price is below 100-day MA (Bearish)")
                    
                    with col2:
                        st.write(f"**200-day MA:** ${ma_200:.2f}")
                        if current_price > ma_200:
                            st.success("Price is above 200-day MA (Bullish)")
                        else:
                            st.warning("Price is below 200-day MA (Bearish)")
                else:
                    st.info("Not enough data for moving averages analysis")
            else:
                st.info("Need at least 200 days of data for moving averages analysis")
            
            # LSTM Prediction Section
            st.subheader("🤖 LSTM Price Prediction")
            
            if KERAS_AVAILABLE:
                # Check for existing model
                model_path = "keras_model.h5"
                
                if os.path.exists(model_path):
                    try:
                        with st.spinner("Loading LSTM model..."):
                            model = load_model(model_path)
                        
                        # Prepare data for prediction
                        x_data, y_data, scaler = prepare_lstm_data(data)
                        
                        if x_data is not None and len(x_data) > 0:
                            # Make prediction
                            last_100_days = x_data[-1].reshape(1, 100, 1)
                            prediction = model.predict(last_100_days, verbose=0)
                            predicted_price = scaler.inverse_transform(prediction)[0][0]
                            
                            price_change = predicted_price - current_price
                            price_change_pct = (price_change / current_price) * 100
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Current Price", f"${current_price:.2f}")
                            with col2:
                                st.metric("Predicted Price", f"${predicted_price:.2f}")
                            with col3:
                                st.metric("Expected Change", f"${price_change:.2f}", f"{price_change_pct:.1f}%")
                            
                            # Model performance (if we have enough data)
                            if len(x_data) > 100:
                                test_size = min(100, len(x_data) // 4)
                                x_test = x_data[-test_size:]
                                y_test = y_data[-test_size:]
                                
                                predictions = model.predict(x_test, verbose=0)
                                
                                # Calculate metrics
                                mse = mean_squared_error(y_test, predictions)
                                mae = mean_absolute_error(y_test, predictions)
                                rmse = np.sqrt(mse)
                                
                                st.write("**Model Performance Metrics:**")
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.write(f"RMSE: {rmse:.4f}")
                                with col2:
                                    st.write(f"MAE: {mae:.4f}")
                                with col3:
                                    st.write(f"MSE: {mse:.4f}")
                    
                    except Exception as e:
                        st.error(f"Error loading model: {str(e)}")
                        st.info("You can create a sample model using the button below.")
                
                else:
                    st.warning("No trained model found. Please train a model first.")
                    
                    if st.button("Create Sample Model"):
                        with st.spinner("Creating sample model..."):
                            try:
                                model = create_sample_model()
                                if model:
                                    # Train on current data
                                    x_data, y_data, scaler = prepare_lstm_data(data)
                                    if x_data is not None:
                                        x_train = x_data.reshape(x_data.shape[0], x_data.shape[1], 1)
                                        model.fit(x_train, y_data, epochs=1, batch_size=32, verbose=0)
                                        model.save(model_path)
                                        st.success("Sample model created and saved!")
                                        st.rerun()
                            except Exception as e:
                                st.error(f"Error creating model: {str(e)}")
            
            else:
                st.error("TensorFlow/Keras is required for LSTM predictions. Please install it.")
            
            # Data table
            with st.expander("📋 View Raw Data"):
                st.dataframe(data.tail(10))
            
            # Disclaimer
            st.markdown("---")
            st.warning("⚠️ **Disclaimer:** This tool is for educational purposes only. Stock predictions should not be used for actual trading decisions. Always consult with financial professionals before making investment decisions.")

if __name__ == "__main__":
    main()
