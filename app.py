import os, time, json, random, requests, warnings
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

# ─────────────────────────  page config  ───────────────────────── #
st.set_page_config(page_title="US StockAI Predictor Pro",
                   page_icon="📈", layout="wide",
                   initial_sidebar_state="expanded")

# ───────────────────────  constants / keys  ────────────────────── #
ALPHA_KEY = (
    st.secrets.get("ALPHA_VANTAGE_API_KEY", "").strip()
    or os.getenv("ALPHA_VANTAGE_API_KEY", "").strip()
    or "U6C4TOUUYCXNM53B"          # demo key: replace in production
)
AV_URL = "https://www.alphavantage.co/query"
PERIOD_DAYS = {"1mo": 30, "3mo": 90, "6mo": 180, "1y": 365, "2y": 730, "5y": 1825}

# ────────────────────────  global styling  ─────────────────────── #
ticker_string = json.dumps(["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "NVDA"])
st.markdown(
    """
    <style>
    /* colourful gradient background */
    .stApp {
      background: radial-gradient(circle at top left, #dbeafe, #fef3c7 40%, #fde68a 75%, #fcd34d);
      background-attachment: fixed;
    }
    /* translucent overlay to keep text readable */
    .stApp::before {
      content:"";
      position:fixed; inset:0; z-index:-1; background:rgba(255,255,255,0.85);
    }
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;500;700&display=swap');
    html, body, [class*="css"] {font-family:'Inter',sans-serif;}
    .main-hdr {
      font-size:3.2rem; font-weight:700; text-align:center; margin-bottom:.2rem;
      background:linear-gradient(90deg,#0b5394,#d62828);
      -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    }
    .sub-hdr {text-align:center;font-size:1.1rem;color:#444;margin-bottom:.6rem;}
    /* greeting badge */
    .greet-badge {
      position:fixed; top:12px; right:18px; z-index:1000;
      background:rgba(11,83,148,.12); border:1px solid rgba(11,83,148,.25);
      padding:6px 14px; border-radius:9px; font-weight:600;
    }
    /* ticker canvas */
    #tickerCanvas {position:fixed; inset:0; pointer-events:none; z-index:-1; opacity:.06;}
    </style>

    <div id="greeting" class="greet-badge">Welcome&nbsp;Guest</div>
    <canvas id="tickerCanvas"></canvas>

    <script>
    const cvs = document.getElementById('tickerCanvas');
    const ctx  = cvs.getContext('2d');
    function fit(){cvs.width=innerWidth; cvs.height=innerHeight;}
    addEventListener('resize',fit); fit();
    const tickers=""" + ticker_string + """;
    let quotes=tickers.map(t=>({sym:t, px:(Math.random()*300+50).toFixed(2),
                                  x:Math.random()*innerWidth,
                                  y:Math.random()*innerHeight}));
    function loop(){
      ctx.clearRect(0,0,cvs.width,cvs.height);
      ctx.font='bold 26px Inter'; ctx.fillStyle='#0b5394';
      quotes.forEach(q=>{
        q.x -= 0.5;
        if(q.x < -160){
           q.x = cvs.width + Math.random()*200;
           q.y = Math.random()*cvs.height;
           q.px = (+q.px + (Math.random()-0.5)*2).toFixed(2);
        }
        ctx.fillText(q.sym + ' ' + q.px, q.x, q.y);
      });
      requestAnimationFrame(loop);
    } loop();
    </script>
    """,
    unsafe_allow_html=True,
)

# ────────────────────────  helper functions  ───────────────────── #
def normalise(sym: str) -> str:
    """BRK-B  -> BRK.B for Alpha-Vantage."""
    sym = sym.strip().upper()
    return sym.replace("-", ".") if "-" in sym and "." not in sym else sym

@st.cache_data(ttl=300)
def fetch_history(sym: str, period: str) -> pd.DataFrame:
    try:
        r = requests.get(
            AV_URL,
            params={
                "function": "TIME_SERIES_DAILY",
                "symbol": normalise(sym),
                "outputsize": "full",
                "apikey": ALPHA_KEY,
            },
            timeout=25,
        ).json()
        ts = r["Time Series (Daily)"]
        df = (
            pd.DataFrame(ts)
            .T.astype(float)
            .rename(
                columns={
                    "1. open": "Open",
                    "2. high": "High",
                    "3. low": "Low",
                    "4. close": "Close",
                    "5. volume": "Volume",
                }
            )
        )
        df.index = pd.to_datetime(df.index)
        df = (
            df.sort_index()
            .reset_index()
            .rename(columns={"index": "Date"})
            .query("Date >= @datetime.now() - @timedelta(days=PERIOD_DAYS[period])")
        )
        return df
    except Exception:
        # sample fallback
        days = PERIOD_DAYS[period]
        dates = pd.date_range(end=datetime.now(), periods=days, freq="B")
        base = random.uniform(50, 300)
        prices = base * np.cumprod(1 + np.random.normal(0, 0.02, len(dates)))
        df = pd.DataFrame({"Date": dates, "Close": prices})
        for col in ["Open", "High", "Low"]:
            df[col] = df["Close"]
        df["Volume"] = np.random.randint(100_000, 400_000, len(dates))
        return df

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) < 55:
        return pd.DataFrame()
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    df["RSI"] = 100 - 100 / (1 + gain / loss)
    for lag in [1, 2, 3, 5]:
        df[f"Lag{lag}"] = df["Close"].shift(lag)
    return df.dropna()

def train_model(df: pd.DataFrame):
    features = [
        "Open", "High", "Low", "Volume", "MA20", "MA50", "RSI",
        "Lag1", "Lag2", "Lag3", "Lag5",
    ]
    if len(df) < 80:
        return None
    X, y = df[features], df["Close"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, shuffle=False)
    scaler = StandardScaler().fit(Xtr)
    mdl = RandomForestRegressor(
        n_estimators=200, max_depth=12, random_state=42
    ).fit(scaler.transform(Xtr), ytr)
    metrics = dict(
        R2=r2_score(yte, mdl.predict(scaler.transform(Xte))),
        RMSE=np.sqrt(mean_squared_error(yte, mdl.predict(scaler.transform(Xte)))),
    )
    return mdl, scaler, metrics, features

# ───────────────────────  user name handling  ───────────────────── #
if "username" not in st.session_state:
    username = None
    try:
        user_obj = st.experimental_user  # available on Streamlit Cloud
    except AttributeError:
        user_obj = None

    if user_obj:
        username = getattr(user_obj, "name", None)
        if not username:
            email = getattr(user_obj, "email", None)
            if email:
                username = email.split("@")[0]
    if not username:
        with st.sidebar:
            username = st.text_input("Enter your name", "Guest")
    st.session_state.username = username.title()

st.markdown(
    "<script>document.getElementById('greeting').innerText = "
    f"'Welcome {st.session_state.username}';</script>",
    unsafe_allow_html=True,
)

# ─────────────────────────  sidebar  ────────────────────────────── #
with st.sidebar:
    st.header("Parameters")
    ticker = st.text_input("Ticker (e.g. AAPL, MSFT, BRK-B)", "AAPL")
    period = st.selectbox("History window", list(PERIOD_DAYS), index=3)
    run = st.button("🔮 Predict")

# ─────────────────────────  main UI  ────────────────────────────── #
st.markdown('<div class="main-hdr">US StockAI Predictor Pro</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-hdr">AI-powered next-day stock prediction</div>', unsafe_allow_html=True)

if run:
    raw = fetch_history(ticker, period)
    proc = add_indicators(raw.copy())

    if proc.empty:
        st.error("Not enough rows to compute indicators. Choose a longer history window.")
        st.stop()

    model_data = train_model(proc)
    if model_data is None:
        st.error("Need at least 80 rows to train the model. Choose a longer window.")
        st.stop()

    model, scaler, metrics, feats = model_data
    latest_pred = float(model.predict(scaler.transform(proc[feats].iloc[[-1]]))[0])
    current_price = float(proc["Close"].iloc[-1])
    pct = (latest_pred - current_price) / current_price * 100

    st.metric("Current", f"${current_price:.2f}")
    st.metric("Predicted next close", f"${latest_pred:.2f}", f"{pct:+.2f}%")
    st.write(f"Model R² {metrics['R2']:.3f} · RMSE {metrics['RMSE']:.2f}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=proc["Date"], y=proc["Close"], name="Close", mode="lines"))
    fig.add_trace(go.Scatter(x=proc["Date"], y=proc["MA20"], name="MA20"))
    fig.add_trace(go.Scatter(x=proc["Date"], y=proc["MA50"], name="MA50"))
    fig.update_layout(template="plotly_white", height=380)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Recent data")
    st.dataframe(proc.tail(40), use_container_width=True)
else:
    st.info("Set parameters in the sidebar and press **Predict**.")
