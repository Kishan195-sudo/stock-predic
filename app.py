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

# ------------------------------------------------------------------------------
# Streamlit page configuration
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="US StockAI Predictor Pro",
    page_icon="🇺🇸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------------------
# Configuration / constants
# ------------------------------------------------------------------------------
ALPHA_VANTAGE_API_KEY = (
    st.secrets.get("ALPHA_VANTAGE_API_KEY", "").strip()
    or os.getenv("ALPHA_VANTAGE_API_KEY", "").strip()
    or "U6C4TOUUYCXNM53B"          # fallback for quick demo
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
PERIOD_DAYS = {"1mo": 30, "3mo": 90, "6mo": 180, "1y": 365, "2y": 730, "5y": 1825}

# ------------------------------------------------------------------------------
# Optional yfinance availability (for API-status panel only)
# ------------------------------------------------------------------------------
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False


# ------------------------------------------------------------------------------
# Styling + animated watermark (f-string with doubled braces)
# ------------------------------------------------------------------------------
tickers_json = json.dumps(list(RELIABLE_TICKERS_US.keys()))
st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');

body {{font-family:'Inter',sans-serif;}}
.main-header {{
  font-size:3.5rem;font-weight:800;
  background:linear-gradient(90deg,#0B5394,#D62828);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
  background-clip:text;text-align:center;margin-bottom:.3rem;
}}
.subtitle {{text-align:center;font-size:1.20rem;color:#555;margin-bottom:.5rem;font-weight:400;}}
.welcome-banner {{
  text-align:center;margin:10px auto 25px;padding:8px 14px;font-weight:700;color:#0B5394;
  border:1px solid rgba(11,83,148,.25);border-radius:10px;
  background:linear-gradient(90deg,rgba(11,83,148,.08),rgba(214,40,40,.08));max-width:900px;
}}
.us-badge {{
  background:linear-gradient(90deg,rgba(11,83,148,.15),rgba(214,40,40,.15));
  padding:.6rem 1rem;border:1px solid rgba(11,83,148,.25);
  border-left:5px solid #0B5394;border-radius:8px;margin-bottom:1rem;
}}
.stButton>button {{
  background:linear-gradient(45deg,#0B5394,#D62828)!important;color:#fff;border:none;
  padding:10px 16px;border-radius:8px;font-weight:600;font-size:15px;transition:.25s;width:100%%;
}}
.stButton>button:hover {{
  background:linear-gradient(45deg,#083b6a,#a41f1f)!important;transform:translateY(-1px);
  box-shadow:0 6px 16px rgba(0,0,0,.15);
}}
.info-card {{
  background:#f7fbff;border:1px solid #d0e6ff;border-left:5px solid #0B5394;
  padding:1rem 1.2rem;border-radius:8px;margin-bottom:1rem;
}}
.warning-card {{
  background:#fff9e6;padding:1rem 1.2rem;border-radius:8px;
  border:1px solid #ffe8a1;margin-top:1.5rem;border-left:5px solid #ffca28;font-size:.9rem;
}}
#MainMenu, footer, header {{visibility:hidden;}}

/* Watermark canvas */
#stock-watermark {{
   position:fixed;inset:0;pointer-events:none;z-index:-1;opacity:0.08;
}}
</style>

<canvas id="stock-watermark"></canvas>

<script>
const cvs = document.getElementById('stock-watermark');
const ctx  = cvs.getContext('2d');
function resize(){{cvs.width=window.innerWidth; cvs.height=window.innerHeight;}}
window.addEventListener('resize',resize); resize();

const tickers = {tickers_json};
let quotes = tickers.map(t=>{{ return {{sym:t,px:(Math.random()*300+50).toFixed(2),x:Math.random()*window.innerWidth,
                                         y:Math.random()*window.innerHeight}} }});
function draw(){{
  ctx.clearRect(0,0,cvs.width,cvs.height);
  ctx.font='bold 26px Inter'; ctx.fillStyle='#0B5394';
  quotes.forEach(q=>{{
     q.x -= 0.6;
     if(q.x < -150) {{ q.x = cvs.width + Math.random()*200;
                      q.y = Math.random()*cvs.height;
                      q.px = (parseFloat(q.px)+(Math.random()-0.5)*3).toFixed(2); }}
     ctx.fillText(q.sym + '  ' + q.px, q.x, q.y);
  }});
  requestAnimationFrame(draw);
}}
draw();
</script>
""",
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------------------
def normalize_symbol(sym:str)->str:
    sym = sym.strip().upper()
    return sym.replace("-", ".") if "-" in sym and "." not in sym else sym

@st.cache_data(ttl=300)
def fetch_data(sym:str,period:str)->pd.DataFrame:
    try:
        av_symbol = normalize_symbol(sym)
        time.sleep(1)
        params = {"function":"TIME_SERIES_DAILY","symbol":av_symbol,
                  "outputsize":"full","apikey":ALPHA_VANTAGE_API_KEY}
        raw = requests.get(AV_BASE_URL, params=params, timeout=25).json()
        if "Time Series (Daily)" not in raw:
            raise ValueError("Alpha-Vantage returned no data")
        df = (
            pd.DataFrame(raw["Time Series (Daily)"])
            .T.rename(columns={
                "1. open":"Open","2. high":"High","3. low":"Low",
                "4. close":"Close","5. volume":"Volume"})
            .astype(float)
        )
        df.index = pd.to_datetime(df.index)
        df = df.sort_index().reset_index().rename(columns={"index":"Date"})
        df = df[df["Date"] >= datetime.now()-timedelta(days=PERIOD_DAYS[period])]
        df.attrs["src"]="alpha"
        return df
    except Exception as e:
        st.warning(f"Live data failed ({e}) – using sample")
        return sample_data(sym,period)

def sample_data(sym:str,period:str)->pd.DataFrame:
    days = PERIOD_DAYS[period]
    dates = pd.date_range(end=datetime.now(), periods=days, freq="B")
    base  = random.uniform(50, 300)
    prices = base*np.cumprod(1+np.random.normal(0,0.02,len(dates)))
    df = pd.DataFrame({"Date":dates,"Close":prices})
    df["Open"]=df["High"]=df["Low"]=df["Close"]
    df["Volume"]=np.random.randint(100_000,400_000,len(dates))
    df.attrs["src"]="sample"
    return df

def add_indicators(df):
    if len(df) < 55: return pd.DataFrame()
    df["MA20"]=df["Close"].rolling(20).mean()
    df["MA50"]=df["Close"].rolling(50).mean()
    delta=df["Close"].diff()
    gain=delta.clip(lower=0).rolling(14).mean()
    loss=(-delta.clip(upper=0)).rolling(14).mean()
    df["RSI"]=100-100/(1+gain/loss)
    for lag in [1,2,3,5]:
        df[f"CloseLag{lag}"]=df["Close"].shift(lag)
    return df.dropna()

def train_rf(df):
    feats=["Open","High","Low","Volume","MA20","MA50","RSI",
           "CloseLag1","CloseLag2","CloseLag3","CloseLag5"]
    X=df[feats]; y=df["Close"]
    if len(X) < 80: return None,None,None,None
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,shuffle=False)
    scaler=StandardScaler().fit(Xtr)
    model=RandomForestRegressor(n_estimators=200,max_depth=12,n_jobs=-1,random_state=42)
    model.fit(scaler.transform(Xtr),ytr)
    metrics=dict(R2=r2_score(yte,model.predict(scaler.transform(Xte))),
                 RMSE=np.sqrt(mean_squared_error(yte,model.predict(scaler.transform(Xte)))),
                 MAE=mean_absolute_error(yte,model.predict(scaler.transform(Xte))))
    importance=pd.Series(model.feature_importances_,index=feats).sort_values(ascending=False)
    return model,scaler,metrics,importance

# ------------------------------------------------------------------------------
# Sidebar controls
# ------------------------------------------------------------------------------
with st.sidebar:
    st.markdown("#### Configure")
    choice=st.radio("Ticker input",["Presets","Manual"])
    if choice=="Presets":
        ticker=st.selectbox("Symbol", list(RELIABLE_TICKERS_US))
    else:
        ticker=st.text_input("Enter symbol","AAPL")
    period=st.selectbox("History window", list(PERIOD_DAYS), index=3)
    run_btn = st.button("🚀 Run")

# ------------------------------------------------------------------------------
# Main app
# ------------------------------------------------------------------------------
st.markdown('<h1 class="main-header">US StockAI Predictor Pro</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-powered next-day prediction</p>', unsafe_allow_html=True)
st.markdown('<div class="welcome-banner">WELCOME to the STOCKS world</div>', unsafe_allow_html=True)

if run_btn:
    raw = fetch_data(ticker,period)
    proc = add_indicators(raw.copy())
    if proc.empty:
        st.error("Not enough rows for indicators/model. Pick a longer period.")
        st.stop()

    model,scaler,metrics,imp = train_rf(proc)
    if model is None:
        st.error("Need at least ~80 rows to train. Pick longer period.")
        st.stop()

    latest_feat = scaler.transform(proc[imp.index].iloc[[-1]])
    pred = float(model.predict(latest_feat)[0])
    current = float(proc["Close"].iloc[-1])
    pct = (pred-current)/current*100

    c1,c2 = st.columns(2)
    c1.metric("Current", f"${current:.2f}")
    c2.metric("Predicted next close", f"${pred:.2f}", f"{pct:+.2f}%")

    st.write(f"Model R² {metrics['R2']:.3f}   ·   RMSE {metrics['RMSE']:.2f}")

    # chart
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=proc["Date"],y=proc["Close"],name="Close",mode="lines"))
    fig.add_trace(go.Scatter(x=proc["Date"],y=proc["MA20"],name="MA20",mode="lines"))
    fig.add_trace(go.Scatter(x=proc["Date"],y=proc["MA50"],name="MA50",mode="lines"))
    fig.update_layout(template="plotly_white",height=380)
    st.plotly_chart(fig,use_container_width=True)

    st.subheader("Feature importance")
    st.bar_chart(imp.head(8))

    st.subheader("Recent data")
    st.dataframe(proc.tail(40),use_container_width=True)
else:
    st.info("Choose settings in the sidebar and click **Run** to start.")
