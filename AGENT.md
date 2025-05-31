# Stock Market Predictor Agent Guide

## Commands
- **Run Streamlit app**: `streamlit run app.py`  
- **Install dependencies**: `pip install -r requirements.txt`
- **Test model**: Run cells in `main.ipynb` or `python app.py`
- **No formal tests**: This is a ML prototype without test suite

## Code Style
- **Imports**: Standard library first, then third-party (numpy, pandas, matplotlib, keras, streamlit, yfinance, sklearn)
- **Variables**: Use snake_case (e.g., `data_training`, `y_predicted`, `user_input`)
- **Constants**: Use lowercase for data ranges (e.g., `start = '2010-01-01'`, `end = '2019-12-31'`)
- **Comments**: Inline comments with `#` for brief explanations
- **Plotting**: Use matplotlib with streamlit integration (`st.pyplot(fig)`)
- **Data processing**: Use pandas DataFrames, numpy arrays for ML operations
- **Model loading**: Use keras `load_model()` for saved models
- **Scaling**: MinMaxScaler from sklearn for data normalization
- **ML arrays**: Convert to numpy arrays before model operations (`np.array()`)

## Project Structure
- `app.py`: Main Streamlit web application
- `main.ipynb`: Jupyter notebook for model development and testing  
- `keras_model.h5`: Trained LSTM model file
- `requirements.txt`: Python dependencies
