import os, time, json, random, requests, warnings
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

# ──────────────────────────────────────────────────────────────────────────────
#  Streamlit page config
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="US StockAI Predictor Pro",
                   page_icon="🇺🇸", layout="wide",
                   initial_sidebar_state="expanded")

# ──────────────────────────────────────────────────────────────────────────────
#  GLOBAL CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────
ALPHA_VANTAGE_API_KEY = (
    st.secrets.get("ALPHA_VANTAGE_API_KEY", "").strip()
    or os.getenv("ALPHA_VANTAGE_API_KEY", "").strip()
    or "U6C4TOUUYCXNM53B"     # fallback for demo
)
AV_BASE_URL = "https://www.alphavantage.co/query"

RELIABLE_TICKERS_US = {
    "AAPL": "Apple Inc.", "GOOGL": "Alphabet Inc.", "MSFT": "Microsoft Corp.",
    "TSLA": "Tesla Inc.", "AMZN": "Amazon.com Inc.", "NVDA": "NVIDIA Corp.",
    "META": "Meta Platforms Inc.", "NFLX": "Netflix Inc.",
    "JPM": "JPMorgan Chase & Co.", "V": "Visa Inc.",
    "BRK-B": "Berkshire Hathaway Class B", "UNH": "UnitedHealth Group",
    "XOM": "Exxon Mobil", "PG": "Procter & Gamble", "HD": "Home Depot"
}
PERIOD_DAYS = {"1mo":30,"3mo":90,"6mo":180,"1y":365,"2y":730,"5y":1825}

# ──────────────────────────────────────────────────────────────────────────────
#  Styling + Animated watermark (CSS + JS canvas)
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');

body {font-family:'Inter',sans-serif;}
.main-header{font-size:3.5rem;font-weight:800;background:linear-gradient(90deg,#0B5394,#D62828);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;text-align:center;margin-bottom:.3rem;}
.subtitle{text-align:center;font-size:1.20rem;color:#555;margin-bottom:.5rem;font-weight:400;}
.welcome-banner{text-align:center;margin:10px auto 25px;padding:8px 14px;font-weight:700;color:#0B5394;border:1px solid rgba(11,83,148,.25);border-radius:10px;background:linear-gradient(90deg,rgba(11,83,148,.08),rgba(214,40,40,.08));max-width:900px;}
.us-badge{background:linear-gradient(90deg,rgba(11,83,148,.15),rgba(214,40,40,.15));padding:.6rem 1rem;border:1px solid rgba(11,83,148,.25);border-left:5px solid #0B5394;border-radius:8px;margin-bottom:1rem;}
.stButton>button{background:linear-gradient(45deg,#0B5394,#D62828)!important;color:#fff;border:none;padding:10px 16px;border-radius:8px;font-weight:600;font-size:15px;transition:.25s;width:100%;}
.stButton>button:hover{background:linear-gradient(45deg,#083b6a,#a41f1f)!important;transform:translateY(-1px);box-shadow:0 6px 16px rgba(0,0,0,.15);}
.info-card{background:#f7fbff;border:1px solid #d0e6ff;border-left:5px solid #0B5394;padding:1rem 1.2rem;border-radius:8px;margin-bottom:1rem;}
.warning-card{background:#fff9e6;padding:1rem 1.2rem;border-radius:8px;border:1px solid #ffe8a1;margin-top:1.5rem;border-left:5px solid #ffca28;font-size:.9rem;}
#MainMenu, footer, header{visibility:hidden;}

/*  ── WATERMARK CANVAS  ─────────────────────────────────────────── */
#stock-watermark{
   position:fixed; inset:0; z-index:-1; pointer-events:none; opacity:0.08;
   background:#ffffff;
}
</style>

<canvas id="stock-watermark"></canvas>

<script>
const cvs = document.getElementById('stock-watermark');
const ctx = cvs.getContext('2d');
function resize(){cvs.width=window.innerWidth; cvs.height=window.innerHeight;}
window.addEventListener('resize',resize); resize();

// random US tickers for animation
const tickers = %s;
let quotes = tickers.map(t => ({sym:t,px:Math.random()*300+50}));
function draw(){
  ctx.clearRect(0,0,cvs.width,cvs.height);
  ctx.font = 'bold 26px Inter';
  ctx.fillStyle = '#0B5394';
  const speed = 0.5;
  quotes.forEach(q=>{
      q.x = (q.x===undefined)? Math.random()*cvs.width : q.x-speed;
      if(q.x < -120){ q.x = cvs.width+Math.random()*200; q.px = (Math.random()*300+50).toFixed(2);}
      ctx.fillText(`${q.sym}  ${q.px}`, q.x, q.y|| (q.y=Math.random()*cvs.height));
  });
  requestAnimationFrame(draw);
}
draw();

// every 8 s pretend prices changed
setInterval(()=>quotes.forEach(q=>q.px=(+q.px+(Math.random()-0.5)*2).toFixed(2)),8000);
</script>
""" % json.dumps(list(RELIABLE_TICKERS_US.keys())),
unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
#  Utilities
# ──────────────────────────────────────────────────────────────────────────────
def normalize_symbol(symbol:str)->str:
    symbol = symbol.strip().upper()
    return symbol.replace("-", ".") if "-" in symbol and "." not in symbol else symbol

def fetch_data(symbol:str,period:str)->pd.DataFrame:
    """Return OHLCV DataFrame with at least 60 rows or sample fallback."""
    try:
        av_symbol = normalize_symbol(symbol)
        time.sleep(1)   # free-tier polite delay
        params = {"function":"TIME_SERIES_DAILY","symbol":av_symbol,
                  "outputsize":"full","apikey":ALPHA_VANTAGE_API_KEY}
        r = requests.get(AV_BASE_URL, params=params, timeout=25).json()
        if "Time Series (Daily)" not in r:  raise ValueError("Alpha Vantage error")
        df = pd.DataFrame(r["Time Series (Daily)"]).T.rename(columns={
            "1. open":"Open","2. high":"High","3. low":"Low","4. close":"Close","5. volume":"Volume"
        }).astype(float)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index().reset_index().rename(columns={"index":"Date"})
        cutoff = datetime.now() - timedelta(days=PERIOD_DAYS[period])
        df = df[df["Date"] >= cutoff]
        df.attrs["source"] = "alpha"
        return df
    except Exception as e:
        st.warning(f"Live feed failed ({e}) — using sample data.")
        return sample_data(symbol, period)

def sample_data(symbol:str,period:str)->pd.DataFrame:
    days = PERIOD_DAYS[period]
    rng = pd.date_range(end=datetime.now(), periods=days, freq="B")
    base = random.uniform(50,300)
    prices = np.maximum(base*np.cumprod(1+np.random.normal(0,0.02,len(rng))),1)
    df = pd.DataFrame({"Date":rng,"Close":prices})
    df["Open"]=df["High"]=df["Low"]=df["Close"]
    df["Volume"] = np.random.randint(100_000,500_000,size=len(rng))
    df.attrs["source"] = "sample"
    return df

def add_indicators(df:pd.DataFrame)->pd.DataFrame:
    if len(df) < 55:   # not enough for MA50
        return pd.DataFrame()    # will be handled later
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()
    # RSI
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    df["RSI"] = 100 - 100/(1+rs)
    for lag in [1,2,3,5]:
        df[f"CloseLag{lag}"] = df["Close"].shift(lag)
    return df.dropna()

def train_model(df:pd.DataFrame):
    feature_cols = ["Open","High","Low","Volume","MA20","MA50","RSI",
                    "CloseLag1","CloseLag2","CloseLag3","CloseLag5"]
    X = df[feature_cols]; y = df["Close"]
    if len(df) < 80:         # not enough rows for split
        return None,None,None,None
    X_train,X_test,y_train,y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    scaler = StandardScaler().fit(X_train)
    model = RandomForestRegressor(n_estimators=200,max_depth=12,n_jobs=-1,random_state=42)
    model.fit(scaler.transform(X_train), y_train)
    y_pred = model.predict(scaler.transform(X_test))
    metrics = {"R2":r2_score(y_test,y_pred),
               "RMSE":np.sqrt(mean_squared_error(y_test,y_pred)),
               "MAE":mean_absolute_error(y_test,y_pred)}
    importance = pd.Series(model.feature_importances_,index=feature_cols)\
                  .sort_values(ascending=False)
    return model, scaler, metrics, importance

# ──────────────────────────────────────────────────────────────────────────────
#  UI  (sidebar controls)
# ──────────────────────────────────────────────────────────────────────────────
st.markdown('<h1 class="main-header">US StockAI Predictor Pro</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-powered next-day US-stock prediction</p>', unsafe_allow_html=True)
st.markdown('<div class="welcome-banner">WELCOME to the STOCKS world</div>',unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### Configuration")
    st.markdown('<div class="us-badge">US Market Mode Enabled</div>',unsafe_allow_html=True)
    method = st.radio("Choose symbol source", ["Preset list","Manual"])
    if method=="Preset list":
        symbol = st.selectbox("Pick symbol", list(RELIABLE_TICKERS_US))
    else:
        symbol = st.text_input("Enter ticker (US)", "AAPL")
    period = st.selectbox("History window", list(PERIOD_DAYS.keys()), index=3)
    if st.button("🔍 Analyze & Predict"):
        run=True
    else:
        run=False

# ──────────────────────────────────────────────────────────────────────────────
if run:
    df_raw = fetch_data(symbol,period)
    df = add_indicators(df_raw.copy())

    if df.empty:
        st.error("Not enough historical rows to compute indicators. "
                 "Choose a longer period (≥ 6 mo).")
        st.stop()

    src = df_raw.attrs["source"]
    st.success(f"Loaded {len(df)} rows · source: {src}")

    model,scaler,metrics,importance = train_model(df)
    if model is None:
        st.error("Insufficient rows to train model (need ≥ 80).")
        st.stop()

    latest_X = scaler.transform(df[importance.index].iloc[[-1]])
    pred = float(model.predict(latest_X)[0])
    current = float(df["Close"].iloc[-1])
    delta = pred-current; pct = delta/current*100

    c1,c2 = st.columns(2)
    with c1:
        st.metric("Current", f"${current:.2f}")
    with c2:
        st.metric("Next-day prediction", f"${pred:.2f}", f"{pct:+.2f}%")

    st.write("Model R²", f"{metrics['R2']:.3f}   |   RMSE {metrics['RMSE']:.2f}")

    # price chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Date"],y=df["Close"],mode="lines",name="Close"))
    fig.add_trace(go.Scatter(x=df["Date"],y=df["MA20"],mode="lines",name="MA20"))
    fig.add_trace(go.Scatter(x=df["Date"],y=df["MA50"],mode="lines",name="MA50"))
    fig.update_layout(height=400,template="plotly_white")
    st.plotly_chart(fig,use_container_width=True)

    st.subheader("Feature importance")
    st.bar_chart(importance.head(8))

    st.subheader("Raw data (last 40)")
    st.dataframe(df.tail(40),use_container_width=True)

else:
    st.info("Select a symbol & click **Analyze & Predict**")

# ──────────────────────────────────────────────────────────────────────────────
