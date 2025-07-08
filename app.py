import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
import warnings
warnings.filterwarnings('ignore')

# Configure matplotlib for streamlit
plt.style.use('default')

# App configuration
st.set_page_config(
    page_title="Stock Trend Predictor",
    page_icon="📈",
    layout="wide"
)

# Title and description
st.title('📈 Stock Trend Predictor')
st.markdown("---")

# Sidebar for user inputs
st.sidebar.header("Configuration")
user_input = st.sidebar.text_input('Enter Stock Ticker', 'AAPL').upper()

# Date range selection
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime('2010-01-01'))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime('2019-12-31'))

# Convert dates to strings
start = start_date.strftime('%Y-%m-%d')
end = end_date.strftime('%Y-%m-%d')

# Data loading with error handling
@st.cache_data
def load_stock_data(ticker, start_date, end_date):
    try:
        data = yf.download(ticker, start_date, end_date)
        if data.empty:
            st.error(f"No data found for ticker {ticker}")
            return None
        
        # Handle MultiIndex columns if present
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)
        
        return data
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Load data
with st.spinner(f'Loading data for {user_input}...'):
    df = load_stock_data(user_input, start, end)

if df is not None and not df.empty:
    # Display basic information
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Days", len(df))
    with col2:
        st.metric("Start Price", f"${df['Close'].iloc[0]:.2f}")
    with col3:
        st.metric("End Price", f"${df['Close'].iloc[-1]:.2f}")
    
    # Data description
    st.subheader(f'📊 Data Summary for {user_input} ({start} to {end})')
    st.write(df.describe())
    
    # Visualizations
    st.subheader("📈 Stock Price Visualizations")
    
    # Basic closing price chart
    st.write("**Closing Price Over Time**")
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(df.index, df['Close'], color='blue', linewidth=1.5)
    ax1.set_title(f'{user_input} Closing Price')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Closing Price ($)')
    ax1.grid(True, alpha=0.3)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    plt.tight_layout()
    st.pyplot(fig1)
    plt.close(fig1)
    
    # Moving averages chart
    st.write("**Closing Price with 100-Day Moving Average**")
    ma100 = df['Close'].rolling(100).mean()
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    ax2.plot(df.index, df['Close'], label='Close Price', color='blue', alpha=0.7)
    ax2.plot(df.index, ma100, label='100-Day MA', color='red', linewidth=2)
    ax2.set_title(f'{user_input} Price with 100-Day Moving Average')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Price ($)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)
    
    # Multiple moving averages
    st.write("**Closing Price with 100-Day & 200-Day Moving Averages**")
    ma100 = df['Close'].rolling(100).mean()
    ma200 = df['Close'].rolling(200).mean()
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    ax3.plot(df.index, df['Close'], label='Close Price', color='blue', alpha=0.7)
    ax3.plot(df.index, ma100, label='100-Day MA', color='red', linewidth=2)
    ax3.plot(df.index, ma200, label='200-Day MA', color='green', linewidth=2)
    ax3.set_title(f'{user_input} Price with Moving Averages')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Price ($)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close(fig3)
    
    # Model prediction section
    st.subheader("🤖 Stock Price Prediction")
    
    # Check if model exists
    model_path = 'keras_model.h5'
    if os.path.exists(model_path):
        try:
            # Import tensorflow here to avoid issues if not available
            try:
                import tensorflow as tf
                from tensorflow.keras.models import load_model
            except ImportError:
                try:
                    from keras.models import load_model
                except ImportError:
                    st.error("TensorFlow/Keras not available. Please install TensorFlow to use prediction features.")
                    st.stop()
            
            # Load the model
            with st.spinner('Loading prediction model...'):
                model = load_model(model_path)
            
            # Prepare data for prediction
            data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
            data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])
            
            # Scale the data
            scaler = MinMaxScaler(feature_range=(0, 1))
            data_training_array = scaler.fit_transform(data_training)
            
            # Prepare test data
            past_100_days = data_training.tail(100)
            final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
            input_data = scaler.transform(final_df)
            
            # Create test sequences
            x_test = []
            y_test = []
            
            for i in range(100, input_data.shape[0]):
                x_test.append(input_data[i-100:i])
                y_test.append(input_data[i, 0])
            
            if len(x_test) == 0:
                st.warning("Not enough data for prediction. Please select a longer date range.")
            else:
                x_test = np.array(x_test)
                y_test = np.array(y_test)
                
                # Make predictions
                with st.spinner('Generating predictions...'):
                    y_predicted = model.predict(x_test, verbose=0)
                
                # Inverse transform predictions
                y_predicted = scaler.inverse_transform(y_predicted)
                y_test = scaler.inverse_transform(y_test.reshape(-1, 1))
                
                # Calculate metrics
                mse = mean_squared_error(y_test, y_predicted)
                mae = mean_absolute_error(y_test, y_predicted)
                rmse = np.sqrt(mse)
                
                # Display metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("RMSE", f"{rmse:.2f}")
                with col2:
                    st.metric("MAE", f"{mae:.2f}")
                with col3:
                    st.metric("MSE", f"{mse:.2f}")
                
                # Plot predictions vs actual
                st.write("**Predictions vs Actual Prices**")
                fig4, ax4 = plt.subplots(figsize=(12, 6))
                
                # Create date index for test data
                test_start_idx = int(len(df) * 0.70)
                test_dates = df.index[test_start_idx + 100:]  # +100 because we need 100 days for prediction
                
                if len(test_dates) == len(y_test.flatten()):
                    ax4.plot(test_dates, y_test.flatten(), 'b-', label='Actual Price', linewidth=2)
                    ax4.plot(test_dates, y_predicted.flatten(), 'r-', label='Predicted Price', linewidth=2)
                    ax4.set_title(f'{user_input} Stock Price Prediction')
                    ax4.set_xlabel('Date')
                    ax4.set_ylabel('Price ($)')
                    ax4.legend()
                    ax4.grid(True, alpha=0.3)
                    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig4)
                    plt.close(fig4)
                    
                    # Prediction accuracy
                    accuracy = 100 - (np.mean(np.abs((y_test - y_predicted) / y_test)) * 100)
                    st.success(f"Model Accuracy: {accuracy:.2f}%")
                else:
                    st.error("Date alignment issue in prediction visualization.")
            
        except Exception as e:
            st.error(f"Error loading or running the model: {str(e)}")
            st.info("Please ensure the keras_model.h5 file is present and properly trained.")
    else:
        st.warning("⚠️ Model file 'keras_model.h5' not found!")
        st.info("Please run the training notebook (main.ipynb) first to generate the model file.")
        
        # Show sample prediction structure without model
        st.write("**Sample Data Structure for Prediction:**")
        data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
        data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("Training Data Shape:", data_training.shape)
            st.write("Training Data Sample:")
            st.write(data_training.head())
        
        with col2:
            st.write("Testing Data Shape:", data_testing.shape)
            st.write("Testing Data Sample:")
            st.write(data_testing.head())

else:
    st.error("Failed to load stock data. Please check the ticker symbol and try again.")
    st.info("Make sure you enter a valid stock ticker symbol (e.g., AAPL, GOOGL, MSFT)")

# Footer
st.markdown("---")
st.markdown("**Note:** This is a demonstration app. Stock predictions are for educational purposes only and should not be used for actual trading decisions.")
