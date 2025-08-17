import os
import time
import json
import random
import requests
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------
# Page Configuration
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Stock Predictor Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------------------------------
# Constants & API Keys
# --------------------------------------------------------------------------
ALPHA_KEY = (
    st.secrets.get("ALPHA_VANTAGE_API_KEY", "").strip() or
    os.getenv("ALPHA_VANTAGE_API_KEY", "").strip() or
    "U6C4TOUUYCXNM53B"  # Fallback demo key
)
AV_URL = "https://www.alphavantage.co/query"
PERIOD_DAYS = {"1mo": 30, "3mo": 90, "6mo": 180, "1y": 365, "2y": 730, "5y": 1825}
PRESET_TICKERS = {
    "Apple Inc.": "AAPL",
    "Microsoft Corp.": "MSFT",
    "Amazon.com Inc.": "AMZN",
    "NVIDIA Corp.": "NVDA",
    "Alphabet Inc. (Google)": "GOOGL",
    "Tesla Inc.": "TSLA",
    "Meta Platforms Inc.": "META",
    "UnitedHealth Group": "UNH",
    "JPMorgan Chase & Co.": "JPM",
    "Visa Inc.": "V"
}

# --------------------------------------------------------------------------
# Custom CSS Styling to mimic Investing.com
# --------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap');

    body, .stApp {
        font-family: 'Roboto', sans-serif;
        background-color: #FFFFFF;
    }
    .main-container {
        padding: 1rem 2rem;
        border-radius: 10px;
        max-width: 1200px;
        margin: auto;
    }
    .stApp > header {
        background-color: transparent;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: 700;
        color: #333;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #E6E6E6;
        padding-bottom: 0.5rem;
    }
    .stock-table {
        width: 100%;
        border-collapse: collapse;
    }
    .stock-table th, .stock-table td {
        padding: 0.75rem 0.5rem;
        text-align: left;
        border-bottom: 1px solid #E6E6E6;
    }
    .stock-table th {
        font-weight: 500;
        color: #666;
    }
    .stock-table td {
        font-weight: 500;
        color: #222;
    }
    .stock-table a {
        color: #0059B3;
        font-weight: 500;
        text-decoration: none;
    }
    .positive { color: #008000; }
    .negative { color: #D90000; }
    .greet-badge {
      position: fixed; top: 15px; right: 25px; z-index: 1000;
      background: #FFFFFF; border: 1px solid #DDD;
      padding: 8px 16px; border-radius: 8px; font-weight: 500;
      box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Helper Functions (Data Fetching & Model Training)
# --------------------------------------------------------------------------
def normalise(sym: str) -> str:
    sym = sym.strip().upper()
    return sym.replace("-", ".") if "-" in sym and "." not in sym else sym

@st.cache_data(ttl=300)
def fetch_history(sym: str, period: str) -> pd.DataFrame:
    try:
        params = {
            "function": "TIME_SERIES_DAILY", "symbol": normalise(sym),
            "outputsize": "full", "apikey": ALPHA_KEY,
        }
        r = requests.get(AV_URL, params=params, timeout=25).json()
        ts = r["Time Series (Daily)"]
        df = pd.DataFrame(ts).T.astype(float).rename(columns={
            "1. open": "Open", "2. high": "High", "3. low": "Low",
            "4. close": "Close", "5. volume": "Volume"
        })
        df.index = pd.to_datetime(df.index)
        df = df.sort_index().reset_index().rename(columns={"index": "Date"})
        cutoff_date = datetime.now() - timedelta(days=PERIOD_DAYS[period])
        return df[df['Date'] >= cutoff_date]
    except Exception:
        # Fallback to sample data
        days = PERIOD_DAYS[period]
        dates = pd.date_range(end=datetime.now(), periods=days, freq="B")
        base = random.uniform(50, 300)
        prices = base * np.cumprod(1 + np.random.normal(0, 0.02, len(dates)))
        df = pd.DataFrame({"Date": dates, "Close": prices})
        for col in ["Open", "High", "Low"]: df[col] = df["Close"]
        df["Volume"] = np.random.randint(100_000, 400_000, len(dates))
        st.warning(f"Could not fetch live data for {sym}. Displaying sample data instead.")
        return df

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) < 55: return pd.DataFrame()
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    df["RSI"] = 100 - 100 / (1 + gain / loss)
    for lag in [1, 2, 3, 5]: df[f"Lag{lag}"] = df["Close"].shift(lag)
    return df.dropna()

def train_model(df: pd.DataFrame):
    features = ["Open", "High", "Low", "Volume", "MA20", "MA50", "RSI", "Lag1", "Lag2", "Lag3", "Lag5"]
    if len(df) < 80: return None
    X, y = df[features], df["Close"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, shuffle=False)
    scaler = StandardScaler().fit(Xtr)
    mdl = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42).fit(scaler.transform(Xtr), ytr)
    metrics = dict(
        R2=r2_score(yte, mdl.predict(scaler.transform(Xte))),
        RMSE=np.sqrt(mean_squared_error(yte, mdl.predict(scaler.transform(Xte)))),
    )
    return mdl, scaler, metrics, features

# --------------------------------------------------------------------------
# User Greeting Logic
# --------------------------------------------------------------------------
if "username" not in st.session_state:
    username = "Guest"
    try:
        # This works on Streamlit Community Cloud & versions >= 1.14
        user_obj = st.experimental_user
        name = getattr(user_obj, "name", None)
        email = getattr(user_obj, "email", None)
        if name:
            username = name
        elif email:
            username = email.split("@")[0]
    except (AttributeError, Exception):
        # Fallback for local development or older versions
        pass
    st.session_state.username = username.title()

st.markdown(f'<div class="greet-badge">Welcome, {st.session_state.username}</div>', unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Sidebar Controls
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("Prediction Parameters")
    
    selected_stock_name = st.selectbox(
        "Select a Stock",
        options=list(PRESET_TICKERS.keys()),
        index=0 # Default to the first stock in the list
    )
    ticker = PRESET_TICKERS[selected_stock_name]
    
    period = st.selectbox("History Window", list(PERIOD_DAYS.keys()), index=3)
    run = st.button("🔮 Predict Stock Price")

# --------------------------------------------------------------------------
# Main App UI
# --------------------------------------------------------------------------
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# --- Static Indices Table ---
st.markdown('<div class="section-header">United States Indices</div>', unsafe_allow_html=True)
indices_html = """
<table class="stock-table">
  <tr><th>Name</th><th>Last</th><th>Chg.</th><th>Chg. %</th></tr>
  <tr><td><a href="#">Dow Jones</a></td><td>39,411.21</td><td class="negative">-120.62</td><td class="negative">-0.31%</td></tr>
  <tr><td><a href="#">S&P 500</a></td><td>5,464.62</td><td class="positive">+12.42</td><td class="positive">+0.23%</td></tr>
  <tr><td><a href="#">Nasdaq</a></td><td>17,689.36</td><td class="positive">+152.11</td><td class="positive">+0.87%</td></tr>
  <tr><td><a href="#">NYSE Composite</a></td><td>18,202.7</td><td class="negative">-5.70</td><td class="negative">-0.03%</td></tr>
</table>
"""
st.markdown(indices_html, unsafe_allow_html=True)

# --- Prediction Tool Section ---
st.markdown(f'<div class="section-header">Stock Predictor Pro: {selected_stock_name} ({ticker})</div>', unsafe_allow_html=True)

if run:
    with st.spinner(f"Fetching data and training model for {ticker}..."):
        raw = fetch_history(ticker, period)
        proc = add_indicators(raw.copy())

        if proc.empty:
            st.error("Not enough data to compute indicators. Please choose a longer history window (e.g., '1y').")
        else:
            model_data = train_model(proc)
            if model_data is None:
                st.error("Not enough data to train the model. A minimum of ~80 data points is required after processing.")
            else:
                model, scaler, metrics, feats = model_data
                latest_pred = float(model.predict(scaler.transform(proc[feats].iloc[[-1]]))[0])
                current_price = float(proc["Close"].iloc[-1])
                pct_change = (latest_pred - current_price) / current_price * 100

                st.subheader("Prediction Results")
                col1, col2 = st.columns(2)
                col1.metric("Current Price", f"${current_price:.2f}")
                col2.metric("Predicted Next Close", f"${latest_pred:.2f}", f"{pct_change:+.2f}%")
                
                st.write(f"**Model Performance:** R² {metrics['R2']:.3f} · RMSE ${metrics['RMSE']:.2f}")

                st.subheader("Price History & Moving Averages")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=proc["Date"], y=proc["Close"], name="Close Price", line=dict(color='#0059B3', width=2)))
                fig.add_trace(go.Scatter(x=proc["Date"], y=proc["MA20"], name="20-Day MA", line=dict(color='orange', dash='dash')))
                fig.add_trace(go.Scatter(x=proc["Date"], y=proc["MA50"], name="50-Day MA", line=dict(color='purple', dash='dot')))
                fig.update_layout(template="plotly_white", height=400, margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Select a stock and click 'Predict Stock Price' in the sidebar to begin.")

st.markdown('</div>', unsafe_allow_html=True)
