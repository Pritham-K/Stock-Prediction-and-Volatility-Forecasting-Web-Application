import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from arch import arch_model
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
import numpy as np

# Function to set background image using CSS
def set_bg_hack_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background: url("https://static.vecteezy.com/system/resources/previews/006/852/804/original/abstract-blue-background-simple-design-for-your-website-free-vector.jpg");
             background-size: cover;
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

# Function to load data from Yahoo Finance
@st.cache_data
def load_data(ticker, start, end):
    data = yf.download(ticker, start, end)['Close']
    return data

# Function to plot raw data (closing prices over time)
def plot_raw_data(data, stock_name):
    fig = plt.figure(figsize=(12, 6))
    plt.plot(data)
    plt.title(f'{stock_name} Closing Price Over Time', fontsize=20)
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    st.pyplot(fig)

# Function to plot returns
def plot_returns(returns, stock_name):
    fig = plt.figure(figsize=(16, 4))
    plt.plot(returns)
    plt.title(f'{stock_name} Returns', fontsize=20)
    plt.xlabel('Date')
    plt.ylabel('Percentage Returns')
    st.pyplot(fig)

# Function to plot PACF and ACF graphs
def plot_pacf_acf(data):
    lag_acf = range(2, 50)
    lag_pacf = range(2, 40)
    height = 4
    width = 12
    f, ax = plt.subplots(nrows=2, ncols=1, figsize=(width, 2 * height))
    fig1 = plot_acf(data, lags=lag_acf, ax=ax[0])
    fig2 = plot_pacf(data, lags=lag_pacf, ax=ax[1], method='ols')
    plt.tight_layout()
    st.pyplot(f)

# Function to fit ARCH model and display summary
@st.cache_resource
def fit_arch_model(train, p):
    model = arch_model(train, p=p, q=p)
    model_fit = model.fit(disp='off')
    return model_fit

# Function to forecast volatility and plot predictions
def forecast_volatility(model_fit, returns):
    horizon = 7
    pred = model_fit.forecast(horizon=horizon)
    future_dates = [returns.index[-1] + timedelta(days=i) for i in range(1, horizon + 1)]
    pred = pd.Series(np.sqrt(pred.variance.values[-1, :]), index=future_dates)

    fig = plt.figure(figsize=(10, 4))
    plt.plot(pred)
    plt.title('Volatility Prediction - Next 7 Days', fontsize=20)
    plt.xlabel('Date')
    plt.ylabel('Volatility')
    st.pyplot(fig)

# Main Streamlit application
def main():
    st.title("Stock Prediction App")
    set_bg_hack_url()

    # Date selection for data loading
    start_date = st.date_input("Start Date", datetime(2022, 1, 1))
    end_date = datetime.today().date()

    stock_name = st.text_input("Enter Stock Ticker (e.g., AAPL)", 'AAPL')

    # Load data
    data_loader_state = st.text("Loading data...")
    data = load_data(stock_name, start_date, end_date)
    data_loader_state.text("Loading data...done!")

    st.subheader('Raw Data')
    st.write(data.tail())

    # Plot closing prices
    st.subheader("Closing Price vs Time Chart")
    plot_raw_data(data, stock_name)

    # Calculate returns
    returns = 100 * data.pct_change().dropna()

    # Plot returns
    st.subheader("Returns")
    plot_returns(returns, stock_name)

    # Plot PACF and ACF
    st.subheader("ACF and PACF Plots")
    plot_pacf_acf(data)

    # Model selection
    st.subheader("Model Selection and Summary")
    model_order = st.selectbox("Select Model Order (p=q)", ['1', '2', '3'])
    model_order = int(model_order)

    model_fit = fit_arch_model(returns, model_order)
    st.write(model_fit.summary())

    # Forecast volatility and plot predictions
    st.subheader("Volatility Forecasting")
    forecast_volatility(model_fit, returns)

if _name_ == '_main_':
    main()
