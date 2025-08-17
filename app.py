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

# ──────────────────────────────────────────────────────────── #
# Streamlit page config
# ──────────────────────────────────────────────────────────── #
st.set_page_config(page_title="US StockAI Predictor Pro",
                   page_icon="🇺🇸", layout="wide",
                   initial_sidebar_state="expanded")

# ------------------------------------------------------------------------------
# Background image + CSS + ticker watermark
# ------------------------------------------------------------------------------
background_url = "https://images.unsplash.com/photo-1549421263-3aa73c07e4d4"   # <- change if desired
tickers_json   = json.dumps(["AAPL","GOOGL","MSFT","TSLA","AMZN","NVDA"])

st.markdown(
f"""
<style>
/*  Full-page background image */
.stApp {{
  background: url({background_url}) no-repeat center center fixed;
  background-size: cover;
}}
/*  Semi-opaque layer so text is legible */
.stApp::before {{
  content:"";
  position:fixed; inset:0; z-index:-1;
  background:rgba(255,255,255,0.88);
}}
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
body {{font-family:'Inter',sans-serif;}}

.main-header {{
  font-size:3.5rem;font-weight:800;
  background:linear-gradient(90deg,#0B5394,#D62828);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
  background-clip:text;margin-bottom:.2rem;text-align:center;
}}
.subtitle {{text-align:center;font-size:1.15rem;color:#444;margin-bottom:.4rem;}}
.welcome-right {{
  position:fixed; top:12px; right:18px; font-weight:600;
  background:rgba(11,83,148,.1); padding:6px 14px; border-radius:8px;
}}
/* ticker canvas */
#watermark {{position:fixed;inset:0;pointer-events:none;z-index:-1;opacity:.06;}}
</style>

<div class="welcome-right" id="user-greet"></div>
<canvas id="watermark"></canvas>

<script>
const cvs=document.getElementById('watermark');
const ctx=cvs.getContext('2d');
function resize(){{cvs.width=innerWidth; cvs.height=innerHeight;}}
addEventListener('resize',resize); resize();
const tickers={tickers_json};
let quotes=tickers.map(t=>({{sym:t,px:(Math.random()*300+50).toFixed(2),
                              x:Math.random()*innerWidth,
                              y:Math.random()*innerHeight}}));
function loop(){{
  ctx.clearRect(0,0,cvs.width,cvs.height);
  ctx.font='bold 26px Inter'; ctx.fillStyle='#0B5394';
  quotes.forEach(q=>{{ q.x-=0.5;
    if(q.x<-160){{q.x=cvs.width+Math.random()*200;q.px=(+q.px+(Math.random()-0.5)*3).toFixed(2);}}
    ctx.fillText(q.sym+' '+q.px,q.x,q.y);
  }});
  requestAnimationFrame(loop);
}} loop();
</script>
""",
unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# Alpha Vantage
# ------------------------------------------------------------------------------
ALPHA_VANTAGE_API_KEY = (
    st.secrets.get("ALPHA_VANTAGE_API_KEY","").strip()
    or os.getenv("ALPHA_VANTAGE_API_KEY","").strip()
    or "U6C4TOUUYCXNM53B"
)
AV_URL="https://www.alphavantage.co/query"
PERIOD_DAYS={"1mo":30,"3mo":90,"6mo":180,"1y":365,"2y":730,"5y":1825}

def normalize(sym:str)->str:
    sym=sym.strip().upper()
    return sym.replace("-",".") if "-" in sym and "." not in sym else sym

@st.cache_data(ttl=300)
def get_history(sym:str,period:str)->pd.DataFrame:
    try:
        params={"function":"TIME_SERIES_DAILY","symbol":normalize(sym),
                "apikey":ALPHA_VANTAGE_API_KEY,"outputsize":"full"}
        data=requests.get(AV_URL,params=params,timeout=25).json()
        ts=data["Time Series (Daily)"]
        df=pd.DataFrame(ts).T.astype(float)
        df=df.rename(columns={"1. open":"Open","2. high":"High","3. low":"Low",
                              "4. close":"Close","5. volume":"Volume"})
        df.index=pd.to_datetime(df.index)
        df=df.sort_index().reset_index().rename(columns={"index":"Date"})
        df=df[df["Date"]>=datetime.now()-timedelta(days=PERIOD_DAYS[period])]
        return df
    except Exception as e:
        st.warning(f"Alpha-Vantage error, using sample ({e})")
        rng=pd.date_range(end=datetime.now(),periods=PERIOD_DAYS[period],freq="B")
        base=random.uniform(50,300)
        prices=base*np.cumprod(1+np.random.normal(0,0.02,len(rng)))
        df=pd.DataFrame({"Date":rng,"Close":prices})
        df["Open"]=df["High"]=df["Low"]=df["Close"]
        df["Volume"]=np.random.randint(100000,400000,len(rng))
        return df

def add_indicators(df):
    if len(df)<55: return pd.DataFrame()
    df["MA20"]=df["Close"].rolling(20).mean()
    df["MA50"]=df["Close"].rolling(50).mean()
    delta=df["Close"].diff()
    gain=delta.clip(lower=0).rolling(14).mean()
    loss=(-delta.clip(upper=0)).rolling(14).mean()
    df["RSI"]=100-100/(1+gain/loss)
    for lag in [1,2,3,5]:
        df[f"Lag{lag}"]=df["Close"].shift(lag)
    return df.dropna()

def train_model(df):
    feats=["Open","High","Low","Volume","MA20","MA50","RSI",
           "Lag1","Lag2","Lag3","Lag5"]
    X=df[feats]; y=df["Close"]
    if len(X)<80: return None
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,shuffle=False)
    scaler=StandardScaler().fit(Xtr)
    mdl=RandomForestRegressor(n_estimators=200,max_depth=12,random_state=42).fit(
        scaler.transform(Xtr),ytr)
    metrics=dict(R2=r2_score(yte,mdl.predict(scaler.transform(Xte))),
                 RMSE=np.sqrt(mean_squared_error(yte,mdl.predict(scaler.transform(Xte)))))
    return mdl,scaler,metrics,feats

# ------------------------------------------------------------------------------
#  USER NAME (auto from query param ?user=YourName  OR ask once)
# ------------------------------------------------------------------------------
if "username" not in st.session_state:
    qp=st.query_params.get("user")
    if qp: st.session_state.username=qp.split("@")[0].title()
    else:
        with st.sidebar:
            st.session_state.username=st.text_input("Enter your name","Guest")

st.markdown(f"""
<script>document.getElementById('user-greet').innerText =
  "Welcome {st.session_state.username}";</script>
""",unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# Sidebar controls
# ------------------------------------------------------------------------------
with st.sidebar:
    st.header("Select stock")
    symbol=st.text_input("Ticker (US)", "AAPL")
    period=st.selectbox("History", list(PERIOD_DAYS), index=3)
    run=st.button("🔮 Predict")

# ------------------------------------------------------------------------------
# Main area
# ------------------------------------------------------------------------------
st.markdown('<h1 class="main-header">US StockAI Predictor Pro</h1>',unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-powered next-day stock prediction</p>',unsafe_allow_html=True)

if run:
    raw=get_history(symbol,period)
    proc=add_indicators(raw.copy())
    if proc.empty:
        st.error("Not enough rows for indicators; choose a longer history.")
        st.stop()

    trained=train_model(proc)
    if trained is None:
        st.error("Need ≥80 rows to train; use longer history.")
        st.stop()
    model,scaler,metrics,feat_list=trained

    latest=scaler.transform(proc[feat_list].iloc[[-1]])
    pred=float(model.predict(latest)[0])
    current=float(proc["Close"].iloc[-1])
    pct=(pred-current)/current*100

    st.metric("Current",f"${current:.2f}")
    st.metric("Predicted next close",f"${pred:.2f}",f"{pct:+.2f}%")

    fig=go.Figure()
    fig.add_trace(go.Scatter(x=proc["Date"],y=proc["Close"],name="Close",mode="lines"))
    fig.add_trace(go.Scatter(x=proc["Date"],y=proc["MA20"],name="MA20"))
    fig.add_trace(go.Scatter(x=proc["Date"],y=proc["MA50"],name="MA50"))
    fig.update_layout(template="plotly_white",height=380)
    st.plotly_chart(fig,use_container_width=True)

    st.write(f"Model R² {metrics['R2']:.3f} · RMSE {metrics['RMSE']:.2f}")
    st.subheader("Last 40 rows"); st.dataframe(proc.tail(40),use_container_width=True)
else:
    st.info("Press **Predict** in the sidebar to run.")
