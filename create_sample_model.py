import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import os
import warnings
warnings.filterwarnings('ignore')

# Check TensorFlow
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    print(f"TensorFlow version: {tf.__version__}")
    TF_AVAILABLE = True
except ImportError as e:
    print(f"TensorFlow not available: {e}")
    TF_AVAILABLE = False

def download_stock_data(ticker="AAPL", period="5y"):
    try:
        print(f"\nDownloading data for {ticker} ({period})...")
        stock = yf.Ticker(ticker)
        data = stock.history(period=period)
        if data.empty:
            print("No data returned.")
            return None
        print(f"Data downloaded: {len(data)} rows")
        return data
    except Exception as e:
        print(f"Error downloading data: {e}")
        return None

def prepare_data_for_lstm(data, lookback=100):
    if data is None or data.empty:
        print("No data to prepare")
        return None, None, None, None
    if len(data) < lookback + 50:
        print(f"Not enough data: {len(data)} rows (min {lookback+50})")
        return None, None, None, None

    close_prices = data['Close'].fillna(method='ffill').fillna(method='bfill').values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_prices.reshape(-1, 1))

    x_data, y_data = [], []
    for i in range(lookback, len(scaled_data)):
        x_data.append(scaled_data[i-lookback:i, 0])
        y_data.append(scaled_data[i, 0])

    x_data = np.array(x_data).reshape(-1, lookback, 1)
    y_data = np.array(y_data)

    print(f"Prepared sequences: X={x_data.shape}, Y={y_data.shape}")
    return x_data, y_data, scaler, close_prices

def create_lstm_model(lookback=100):
    if not TF_AVAILABLE:
        print("TensorFlow is not available.")
        return None
    try:
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(lookback, 1)),
            Dropout(0.2),
            LSTM(50, return_sequences=True),
            Dropout(0.2),
            LSTM(50, return_sequences=True),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
        print("Model built successfully.")
        return model
    except Exception as e:
        print(f"Error creating model: {e}")
        return None

def train_model(model, x_data, y_data):
    try:
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        history = model.fit(x_data, y_data, epochs=50, batch_size=32, validation_split=0.2,
                            callbacks=[early_stop], verbose=1)
        print("Training completed.")
        return history
    except Exception as e:
        print(f"Error during training: {e}")
        return None

def evaluate_model(model, x_data, y_data, scaler):
    try:
        predictions = model.predict(x_data)
        loss = model.evaluate(x_data, y_data, verbose=0)
        y_actual = scaler.inverse_transform(y_data.reshape(-1, 1))
        y_pred = scaler.inverse_transform(predictions)

        mse = np.mean((y_actual - y_pred) ** 2)
        mae = np.mean(np.abs(y_actual - y_pred))
        rmse = np.sqrt(mse)

        print(f"Evaluation — Loss: {loss[0]:.6f}, MAE: {loss[1]:.6f}")
        print(f"Metrics — RMSE: ${rmse:.2f}, MAE: ${mae:.2f}, MSE: ${mse:.2f}")
        return {'rmse': rmse, 'mae': mae, 'mse': mse, 'predictions': y_pred, 'actual': y_actual}
    except Exception as e:
        print(f"Evaluation failed: {e}")
        return None

def save_model(model, filename="keras_model.h5"):
    try:
        model.save(filename)
        size = os.path.getsize(filename) / (1024 * 1024)
        print(f"Model saved as '{filename}' ({size:.2f} MB)")
        return True
    except Exception as e:
        print(f"Model save failed: {e}")
        return False

#  Main execution function (call this in a Jupyter cell)
def run_lstm_pipeline(ticker="AAPL", period="5y", lookback=100):
    print("=" * 60)
    print(f"📈 Running LSTM model for {ticker} ({period})")
    print("=" * 60)

    if not TF_AVAILABLE:
        print("❌ TensorFlow is not installed.")
        return

    data = download_stock_data(ticker, period)
    if data is None: return

    x_data, y_data, scaler, _ = prepare_data_for_lstm(data, lookback)
    if x_data is None: return

    model = create_lstm_model(lookback)
    if model is None: return

    model.summary()

    history = train_model(model, x_data, y_data)
    if history is None: return

    _ = evaluate_model(model, x_data, y_data, scaler)

    if not save_model(model):
        return

    print("Done! Model is ready and saved.")

return model
