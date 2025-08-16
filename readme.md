# US StockAI Predictor Pro

US StockAI Predictor Pro is a Streamlit web app for US market analysis and AI-powered next-day price prediction. It fetches daily stock data from Alpha Vantage, engineers technical indicators, trains a Random Forest model, visualizes insights with Plotly, and provides an interactive, US-themed UI.

- Live-ready: Deploy in minutes on Streamlit Community Cloud
- Batteries included: Technical indicators, model metrics, feature importance, CSV export
- Robust: Falls back to realistic sample data if live APIs are unavailable

## Features

- US-focused UI and ticker presets (AAPL, MSFT, NVDA, AMZN, TSLA, etc.)
- Data source: Alpha Vantage (daily OHLCV)
- Optional yfinance availability check (for status only)
- Technical indicators: 20/50-day MA, RSI, Volume MA, daily price change
- Machine learning:
  - RandomForestRegressor with feature scaling
  - Train/test split with R², RMSE, MAE metrics
  - Feature importance visualization
- Prediction: Next-day price estimate with signal (bullish/neutral/bearish)
- Visualizations: Price + MAs, Volume, RSI with thresholds
- Data export: Download CSV, view last 50 rows
- Caching: st.cache_data for faster reloads
- Graceful fallback: Auto-generates realistic sample data when API fails

## Tech Stack

- Python, Streamlit
- scikit-learn, pandas, numpy
- plotly, requests
- Alpha Vantage API
- yfinance (optional; status check only)

## Repository Structure

Suggested layout:
- app.py (main Streamlit app; contains all code)
- requirements.txt
- runtime.txt (optional, pin Python version)
- .streamlit/secrets.toml (local development only; do NOT commit)
- README.md

## Requirements

Create a requirements.txt like this:

```
streamlit==1.36.0
pandas
numpy
plotly
scikit-learn
requests
yfinance
```

Optional: Pin Python version with runtime.txt

```
python-3.11
```

## Setup (Local Development)

1) Clone and create a virtual environment:

```
git clone <your-repo-url>
cd <your-repo>
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

2) Configure secrets (Alpha Vantage API key). For local development, create .streamlit/secrets.toml:

```
# .streamlit/secrets.toml
ALPHA_VANTAGE_API_KEY = "your_real_alpha_vantage_key"
```

The app also supports environment variables as a fallback (useful on some hosts):

- Add to your shell or host: ALPHA_VANTAGE_API_KEY=your_real_key
- In code, it already tries st.secrets.get(..., "fallback"), but you can extend it to check os.getenv if you also deploy on hosts that use env vars (e.g., Hugging Face/Render).

3) Run the app:

```
streamlit run app.py
```

Open the local URL shown in the terminal.

## Usage

- Sidebar:
  - Choose market: US Stocks or Custom Ticker
  - Select a preset ticker or type one (e.g., AAPL, BRK-B)
  - Choose period: 1mo to 5y
  - Click “Predict Stock Price”
- Tabs:
  - Stock Analysis: Key stats, RSI, volatility, 52W high/low
  - Predictions: Model metrics + next-day prediction and signal
  - Charts: Close + MAs, Volume, RSI with overbought/oversold
  - Model Performance: Training/testing metrics and feature importance
  - Data Table: Last 50 rows, CSV download, basic stats

Note: The “Days to Predict” slider is currently a preview control; the app predicts the next day price.

## Deployment (Free Options)

### 1) Streamlit Community Cloud (recommended)

- Push your repo to GitHub
- Go to https://share.streamlit.io and sign in with GitHub
- Create new app: select repo, branch, and main file (app.py)
- Set Python version in Advanced or include runtime.txt
- Add secrets: App → Settings → Secrets

Paste in TOML format:

ALPHA_VANTAGE_API_KEY = "your_real_key_here"

- Click Deploy and wait for build to finish

Notes:
- Free Alpha Vantage: ~5 requests/min and 500/day (check your plan)
- The app includes a 1s sleep; consider increasing if traffic grows
- Do not commit secrets.toml to your repo

### 2) Hugging Face Spaces (Streamlit)

- Create a Space (SDK: Streamlit)
- Upload app.py, requirements.txt (and runtime.txt if needed)
- Add a repository secret: ALPHA_VANTAGE_API_KEY
- If you want environment variable support in code, you can modify the line that reads the key to also check os.getenv

Example pattern:
```
import os
ALPHA_VANTAGE_API_KEY = st.secrets.get("ALPHA_VANTAGE_API_KEY") or os.getenv("ALPHA_VANTAGE_API_KEY", "YOUR_FALLBACK_OR_EMPTY")
```

### 3) Render (Web Service)

- Connect your GitHub repo on Render and create a Web Service
- Build command:
```
pip install -r requirements.txt
```
- Start command:
```
streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```
- Add environment variable ALPHA_VANTAGE_API_KEY in Render settings
- Choose free instance type and deploy

## Configuration and Secrets

The app reads the Alpha Vantage API key like this:
- st.secrets.get("ALPHA_VANTAGE_API_KEY", "RSMSPEMM4AQHT6QL")
- Replace the fallback with a blank string in production, or set your real key via Streamlit Cloud Secrets
- For hosts that prefer environment variables, add an os.getenv fallback

## Data Source and Rate Limits

- Primary: Alpha Vantage TIME_SERIES_DAILY endpoint
- Rate limits on free tier are strict; you may see “rate limit exceeded” messages
- The app uses a 1-second delay per request and caches results for 300 seconds
- If live data fails, the app generates realistic sample data so users can continue exploring features

## Model Details

- RandomForestRegressor (n_estimators=200, max_depth=12)
- StandardScaler for feature scaling
- Train/test split (time-ordered, no shuffle)
- Indicators: MA_20, MA_50, RSI, Volume_MA, Price_Change, lag features Close_Lag_{1,2,3,5}
- Metrics: R², RMSE, MAE (training and testing)
- Feature importance plotted via Plotly

Note: Next-day predictions are point estimates and should not be considered financial advice.

## Troubleshooting

- ModuleNotFoundError: Ensure your requirements.txt is complete and re-deploy
- Blank/empty data: Check your ticker symbol and Alpha Vantage API key
- Rate limit warnings: Wait and retry, or upgrade API plan
- Build failures on cloud:
  - Pin Python version with runtime.txt
  - Confirm your app’s main file path matches the deploy settings
- Secrets not found:
  - Streamlit Cloud: Use the Secrets UI (TOML format)
  - Hugging Face/Render: Use environment variables and add os.getenv fallback

## Extending the App

- Add more indicators (MACD, Bollinger Bands)
- Try other models (XGBoost, LightGBM, LSTM)
- Implement multi-day forecasting
- Add earnings/calendar overlays
- Persist predictions or logs via a database

## Disclaimer

This application is for educational and research purposes only. Stock predictions are uncertain and should not be used as the sole basis for investment decisions. Do your own research and consult a qualified financial advisor.

## License

Choose a license that fits your needs (e.g., MIT). If you’re unsure, MIT is a common permissive choice.

## Acknowledgments

- Streamlit for rapid UI
- Alpha Vantage for market data
- Plotly for interactive charts
- scikit-learn for ML utilities

