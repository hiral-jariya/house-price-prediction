# House Price Prediction & Real Estate Analytics

A Streamlit machine-learning application that predicts house prices using Linear Regression and provides dataset exploration, charts, model evaluation, and session-based prediction history.

## Features
- Dashboard with dataset and price statistics
- Dataset exploration and correlation heatmap
- House price prediction using trained Linear Regression
- Model evaluation (R², MAE, RMSE) and actual-vs-predicted chart
- Prediction history with CSV export (current browser session)

## Setup
1. Put `Housing.csv` in this project folder. The CSV must contain a `price` column.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Train the model:
   ```bash
   python train_model.py
   ```
4. Start the app:
   ```bash
   python -m streamlit run app.py
   ```

The app displays prices in the same units as the source dataset. It is an educational estimate, not a professional property valuation.
