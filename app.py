import os, time, requests, warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------ #
# Streamlit page setup
# ------------------------------------------------------------------ #
st.set_page_config(
    page_title="US StockAI Predictor Pro",
    page_icon="🇺🇸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------ #
# --- CSS ---
# ------------------------------------------------------------------ #
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
        .main-header {
            font-size: 3.5rem; font-weight: 800;
            background: linear-gradient(90deg,#0B5394,#D62828);
            -webkit-background-clip:text; -webkit-text-fill-color:transparent;
            background-clip:text; text-align:center; margin-bottom:.5rem;
            font-family:'Inter',sans-serif; letter-spacing:.5px;
        }
        .subtitle {text-align:center;font-size:1.2rem;color:#4a4a4a;margin-bottom:1rem;font-weight:400;}
        .welcome-banner {
            text-align:center;margin:.75rem auto 2rem auto;padding:.75rem 1rem;
            font-weight:700;color:#0B5394;border:1px solid rgba(11,83,148,.25);
            border-radius:10px;
            background:linear-gradient(90deg,rgba(11,83,148,.08),rgba(214,40,40,.08));
            max-width:900px;
        }
        .api-status {padding:1rem;border-radius:8px;margin:1rem 0;}
        .api-working {background:#e8f5e9;color:#1b5e20;border:1px solid #c8e6c9;}
        .api-failed {background:#ffebee;color:#b71c1c;border:1px solid #ffcdd2;}
        .us-badge {
            background:linear-gradient(90deg,rgba(11,83,148,.15),rgba(214,40,40,.15));
            padding:.8rem 1rem;border:1px solid rgba(11,83,148,.25);
            border-left:5px solid #0B5394;border-radius:8px;margin-bottom:1rem;
        }
        .stButton>button {
            background:linear-gradient(45deg,#0B5394,#D62828);color:#fff;border:none;
            padding:.75rem 1.5rem;border-radius:8px;font-weight:600;font-size:1rem;
            transition:all .25s ease;width:100%;
        }
        .stButton>button:hover {
            background:linear-gradient(45deg,#083b6a,#a41f1f);
            transform:translateY(-1px);box-shadow:0 6px 16px rgba(0,0,0,.15);
        }
        .info-card {
            background:#f7fbff;border:1px solid #d0e6ff;border-left:5px solid #0B5394;
            padding:1rem 1.25rem;border-radius:8px;margin-bottom:1rem;
        }
        .warning-card {
            background:#fff9e6;padding:1.25rem;border-radius:8px;
            border:1px solid #ffe8a1;margin-top:1.5rem;border-left:5px solid #ffca28;
        }
        #MainMenu{visibility:hidden;} footer{visibility:hidden;} header{visibility:hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------ #
# --- yfinance availability (only for API-status panel) -------------
# ------------------------------------------------------------------ #
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    st.warning("⚠️ yfinance not installed. Only Alpha Vantage will be used.")

# ------------------------------------------------------------------ #
# --- Alpha Vantage config -----------------------------------------
# ------------------------------------------------------------------ #
ALPHA_VANTAGE_API_KEY = (
    st.secrets.get("ALPHA_VANTAGE_API_KEY", "").strip()
    or os.getenv("ALPHA_VANTAGE_API_KEY", "").strip()
    or "U6C4TOUUYCXNM53B"   # fallback for testing; replace in production
)
AV_BASE_URL = "https://www.alphavantage.co/query"

# ------------------------------------------------------------------ #
RELIABLE_TICKERS_US = {
    "AAPL":"Apple Inc.","GOOGL":"Alphabet Inc.","MSFT":"Microsoft Corporation",
    "TSLA":"Tesla Inc.","AMZN":"Amazon.com Inc.","NVDA":"NVIDIA Corporation",
    "META":"Meta Platforms Inc.","NFLX":"Netflix Inc.","JPM":"JPMorgan Chase & Co.",
    "V":"Visa Inc.","BRK-B":"Berkshire Hathaway Inc. Class B","UNH":"UnitedHealth Group",
    "XOM":"Exxon Mobil","PG":"Procter & Gamble","HD":"Home Depot"
}

# ------------------------------------------------------------------ #
def normalize_symbol_for_alpha_vantage(ticker:str)->str:
    """Convert 'BRK-B' ➜ 'BRK.B' etc. for Alpha Vantage."""
    t=ticker.strip().upper()
    return t.replace("-",".") if "-" in t and "." not in t else t

# ------------------------- API status helper ---------------------- #
def test_api_connections():
    status={"yfinance":{"available":YFINANCE_AVAILABLE,"working":False,"message":""},
            "alpha_vantage":{"available":True,"working":False,"message":""}}
    if YFINANCE_AVAILABLE:
        try:
            yf.Ticker("AAPL").history(period="5d")
            status["yfinance"]["working"]=True
            status["yfinance"]["message"]="✅ yfinance is working"
        except Exception as e:
            status["yfinance"]["message"]=f"❌ yfinance error: {e}"
    else:
        status["yfinance"]["message"]="❌ yfinance not installed"

    try:
        params={"function":"TIME_SERIES_DAILY","symbol":"AAPL",
                "apikey":ALPHA_VANTAGE_API_KEY,"outputsize":"compact"}
        r=requests.get(AV_BASE_URL,params=params,timeout=15).json()
        if "Time Series (Daily)" in r:
            status["alpha_vantage"]["working"]=True
            status["alpha_vantage"]["message"]="✅ Alpha Vantage is working"
        elif "Error Message" in r:
            status["alpha_vantage"]["message"]=f"❌ {r['Error Message']}"
        else:
            status["alpha_vantage"]["message"]="⚠️ Rate-limited?"
    except Exception as e:
        status["alpha_vantage"]["message"]=f"❌ AV error: {e}"
    return status

# --------------------- helper: days for period -------------------- #
PERIOD_DAYS={"1mo":30,"3mo":90,"6mo":180,"1y":365,"2y":730,"5y":1825}

def get_period_days(period:str)->int:
    return PERIOD_DAYS.get(period,365)

# ------------------------- Fetch data ----------------------------- #
@st.cache_data(ttl=300)
def fetch_stock_data(ticker:str,period:str):
    try:
        av_symbol=normalize_symbol_for_alpha_vantage(ticker)
        time.sleep(1)   # respect free-tier rate limit
        params={"function":"TIME_SERIES_DAILY","symbol":av_symbol,
                "apikey":ALPHA_VANTAGE_API_KEY,"outputsize":"full"}
        data=requests.get(AV_BASE_URL,params=params,timeout=30).json()
        if "Time Series (Daily)" not in data:
            raise ValueError(data.get("Error Message","Unknown AV error"))
        df=pd.DataFrame.from_dict(data["Time Series (Daily)"],orient="index")
        df=df.rename(columns={"1. open":"Open","2. high":"High","3. low":"Low",
                              "4. close":"Close","5. volume":"Volume"}).astype(float)
        df.index=pd.to_datetime(df.index)
        df=df.sort_index().reset_index().rename(columns={"index":"Date"})
        cutoff=datetime.now()-timedelta(days=get_period_days(period))
        df=df[df["Date"]>=cutoff]
        df.attrs={"source":"alpha_vantage"}
        return df
    except Exception as e:
        st.warning(f"Alpha Vantage error ➜ using sample data ({e})")
        return generate_sample_data(ticker,period)

# ----------------------- Sample data (fallback) ------------------- #
def generate_sample_data(ticker,period):
    days=get_period_days(period)
    base=100
    np.random.seed(hash(ticker)%2**32)
    dates=pd.date_range(end=datetime.now(),periods=days,freq="B")
    returns=np.random.normal(0.08/252,0.02,days)
    prices=[base]
    for r in returns[1:]:
        prices.append(prices[-1]*(1+r))
    df=pd.DataFrame({"Date":dates,"Close":prices})
    df["Open"]=df["High"]=df["Low"]=df["Close"]
    df["Volume"]=np.random.randint(100000,300000,size=len(df))
    df.attrs={"source":"sample"}
    return df

# ------------------------- Indicators etc. ------------------------ #
def add_indicators(df):
    df["MA20"]=df["Close"].rolling(20).mean()
    df["MA50"]=df["Close"].rolling(50).mean()
    delta=df["Close"].diff()
    gain=(delta.clip(lower=0)).rolling(14).mean()
    loss=(-delta.clip(upper=0)).rolling(14).mean()
    rs=gain/loss
    df["RSI"]=100-100/(1+rs)
    for i in [1,2,3,5]:
        df[f"Close_Lag{i}"]=df["Close"].shift(i)
    return df.dropna()

# ------------------------- ML helpers ----------------------------- #
def prepare_Xy(df):
    features=[c for c in ["Open","High","Low","Volume",
                          "MA20","MA50","RSI",
                          "Close_Lag1","Close_Lag2","Close_Lag3","Close_Lag5"]
              if c in df.columns]
    X=df[features]; y=df["Close"]
    return X,y,features

def train_rf(df):
    X,y,features=prepare_Xy(df)
    Xtrain,Xtest,ytrain,ytest=train_test_split(X,y,test_size=.2,shuffle=False)
    scaler=StandardScaler().fit(Xtrain)
    Xtr=scaler.transform(Xtrain); Xte=scaler.transform(Xtest)
    model=RandomForestRegressor(n_estimators=200,max_depth=12,n_jobs=-1).fit(Xtr,ytrain)
    metrics={"R2":r2_score(ytest,model.predict(Xte)),
             "RMSE":np.sqrt(mean_squared_error(ytest,model.predict(Xte)))}
    importances=pd.Series(model.feature_importances_,index=features)\
                 .sort_values(ascending=False)
    return model,scaler,metrics,importances

# ------------------------------------------------------------------ #
# ----------------------------  UI  -------------------------------- #
# ------------------------------------------------------------------ #
st.markdown('<h1 class="main-header">US StockAI Predictor Pro 🇺🇸</h1>',unsafe_allow_html=True)
st.markdown('<p class="subtitle">US Market analysis & AI-powered prediction</p>',unsafe_allow_html=True)
st.markdown('<div class="welcome-banner">WELCOME to the STOCKS world</div>',unsafe_allow_html=True)

with st.sidebar:
    st.markdown("#### 🇺🇸 Platform Status")
    st.markdown('<div class="us-badge">US Market Mode Enabled</div>',unsafe_allow_html=True)

    choice=st.selectbox("Pick a stock list",["US Presets","Custom"])
    if choice=="US Presets":
        ticker=st.selectbox("Symbol",list(RELIABLE_TICKERS_US))
    else:
        ticker=st.text_input("Enter ticker (e.g. AAPL, BRK-B)","AAPL")
    period=st.selectbox("Period",["1mo","3mo","6mo","1y","2y","5y"],index=3)
    if st.button("🚀 Predict"):
        run=True
    else:
        run=False

# ------------------------------------------------------------------ #
if run:
    data=fetch_stock_data(ticker,period)
    data=add_indicators(data)
    src=data.attrs["source"]
    st.write(f"Data source: **{src}** · Rows: {len(data)}")
    model,scaler,metrics,imp=train_rf(data)
    pred= model.predict(scaler.transform(prepare_Xy(data)[0].iloc[[-1]]))[0]
    current=data["Close"].iloc[-1]
    delta=pred-current; pct=delta/current*100
    st.metric("Current",f"${current:.2f}")
    st.metric("Next-day pred",f"${pred:.2f}",f"{pct:+.2f}%")
    st.write("Model R²",f"{metrics['R2']:.3f}  |  RMSE {metrics['RMSE']:.2f}")
    st.bar_chart(imp.head(10))

# --------------- optional API status panel ------------------------ #
with st.expander("API status"):
    if st.button("Test now"):
        st.json(test_api_connections())
# ------------------------------------------------------------------ #
