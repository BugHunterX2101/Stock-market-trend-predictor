# Stock Market Trend Predictor

A Streamlit web application for stock price analysis and prediction using LSTM neural networks.

## Features

- Real-time stock data fetching using Yahoo Finance
- Interactive stock price visualizations
- Moving averages analysis (100-day and 200-day)
- LSTM-based stock price prediction
- Model performance metrics (RMSE, MAE, MSE)
- User-friendly web interface

## Installation

1. Clone this repository
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Streamlit App

```bash
streamlit run app.py
```

### Training a New Model

1. Open and run the `main.ipynb` notebook to train a new LSTM model
2. The trained model will be saved as `keras_model.h5`
3. Run the Streamlit app to use the trained model for predictions

### Quick Model Creation

If you want to quickly create a sample model for testing:

```bash
python create_sample_model.py
```

## File Structure

- `app.py` - Main Streamlit application
- `main.ipynb` - Jupyter notebook for model training
- `requirements.txt` - Python dependencies
- `keras_model.h5` - Trained LSTM model (generated after training)

## Model Architecture

The LSTM model uses:
- 4 LSTM layers with 50 units each
- Dropout layers (0.2) for regularization
- 100-day lookback window for predictions
- Adam optimizer with mean squared error loss

## Note

Stock predictions are for educational purposes only and should not be used for actual trading decisions.
