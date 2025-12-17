import pandas as pd
import yfinance as yf
import streamlit as st
from statsmodels.tsa.ar_model import AutoReg
import os


@st.cache_data(ttl=86400)
def fetch_sp_tickers():
    # D'abord essaie le CSV local
    current_file = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
    csv_path = os.path.join(project_root, "assets", "data", "sp500_tickers.csv")
    
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            sp_tickers = df.set_index("Symbol")["Security"].to_dict()
            st.success(f"✅ {len(sp_tickers)} tickers chargés depuis le CSV local")
            return sp_tickers
        except Exception as e:
            st.warning(f"Erreur avec le CSV local : {e}. Tentative via Wikipedia...")
    
    # Fallback : Wikipedia (toujours à jour)
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        tables = pd.read_html(url)
        df = tables[0]
        ticker_col = "Symbol" if "Symbol" in df.columns else "Ticker symbol"
        df[ticker_col] = df[ticker_col].str.replace('.', '-')
        sp_tickers = df.set_index(ticker_col)["Security"].to_dict()
        st.success(f"✅ {len(sp_tickers)} tickers S&P 500 chargés depuis Wikipedia (à jour !)")
        return sp_tickers
    except Exception as e:
        st.error(f"Impossible de charger les tickers : {e}")
        return {}

def fetch_stock_history(stock_ticker, period="max", interval="1d"):
    """
    Fetch historical stock data from Yahoo Finance.
    Args:
        stock_ticker (str): The stock ticker symbol.
        period (str): The time period for the data.
        interval (str): The interval for the data.
    Returns:
        pd.DataFrame: A DataFrame containing stock data with columns ['Open', 'High', 'Low', 'Close'].
    """
    try:
        stock_data = yf.Ticker(stock_ticker).history(period=period, interval=interval)
        if stock_data.empty:
            raise ValueError(f"No data found for ticker {stock_ticker}.")
        return stock_data[['Open', 'High', 'Low', 'Close']]
    except Exception as e:
        raise Exception(f"Error fetching stock data for {stock_ticker}: {e}")


def generate_stock_prediction(stock_ticker, forecast_days=30):
    """
    Generate stock price predictions using AutoReg model.
    Args:
        stock_ticker (str): The stock ticker symbol.
        forecast_days (int): The number of days to forecast.
    Returns:
        tuple: Training data, test data, predictions, and forecast values.
    """
    try:
        # Fetch the last 2 years of historical stock data
        stock_data = fetch_stock_history(stock_ticker, period="2y")

        # Prepare the close prices data
        close_prices = stock_data['Close'].asfreq('D', method='ffill')

        # Ensure there's enough data for the model
        if len(close_prices) < 250:  # Minimum data required for lags
            raise ValueError("Not enough historical data available for this stock to generate predictions.")

        # Split the data into train and test sets
        train_data = close_prices.iloc[:int(0.9 * len(close_prices))]
        test_data = close_prices.iloc[int(0.9 * len(close_prices)):]

        # Fit the AutoReg model
        model = AutoReg(train_data, lags=min(250, len(train_data) - 1)).fit()

        # Predict on the test data
        predictions = model.predict(start=test_data.index[0], end=test_data.index[-1], dynamic=True)

        # Predict future values
        forecast_index = pd.date_range(start=test_data.index[-1] + pd.Timedelta(days=1), periods=forecast_days, freq='D')
        forecast = model.predict(start=len(close_prices), end=len(close_prices) + forecast_days - 1)
        forecast = pd.Series(forecast, index=forecast_index)

        return train_data, test_data, predictions, forecast

    except ValueError as ve:
        raise ValueError(ve)  # Raise user-friendly warnings
    except Exception as e:
        raise Exception(f"Error generating prediction: {e}")

