from pathlib import Path
import pandas as pd, plotly.express as px, streamlit as st
from src.data_utils import load_data
from src.booking_analysis import booking_summary,recommend_booking_window
ROOT=Path(__file__).parent
st.set_page_config(page_title="AI Travel Analyst",page_icon="✈️",layout="wide")
@st.cache_data
def data(): return load_data(ROOT/"data/flight_pricing_dataset.csv")
df=data()
st.title("✈️ AI Travel Analyst")
st.caption("VIT Chennai AIML Club • Track 3: Data Science & Visualisation")
with st.sidebar:
    st.header("Filters")
    opts=lambda c:["All"]+sorted(df[c].dropna().astype(str).unique())
    airline=st.selectbox("Airline",opts("Airline")); source=st.selectbox("Source",opts("Source")); destination=st.selectbox("Destination",opts("Destination")); travel_class=st.selectbox("Travel Class",opts("Travel_Class"))
f=df.copy()
for c,v in [("Airline",airline),("Source",source),("Destination",destination),("Travel_Class",travel_class)]:
    if v!="All": f=f[f[c].astype(str)==v]
a,b=st.columns(2); a.metric("Rows",f"{len(df):,}"); b.metric("Usable prices",f"{df.Price.notna().sum():,}")
t1,t2,t3,t4=st.tabs(["📊 Exploration","🤖 Modeling","💰 Cheapest Booking","🧠 Insights"])
with t1:
    x,y=st.columns(2)
    with x:
        g=f.dropna(subset=["Price"]).groupby("Airline",as_index=False).Price.median().sort_values("Price")
        st.plotly_chart(px.bar(g,x="Price",y="Airline",orientation="h",title="Median price by airline"),use_container_width=True)
    with y:
        g=f.dropna(subset=["Price"]).groupby("Travel_Class",as_index=False).Price.median()
        st.plotly_chart(px.bar(g,x="Travel_Class",y="Price",title="Median price by travel class"),use_container_width=True)
    x,y=st.columns(2)
    with x:
        s=f.dropna(subset=["Price"]).sample(min(5000,len(f.dropna(subset=["Price"]))),random_state=42)
        st.plotly_chart(px.scatter(s,x="Distance_km",y="Price",color="Travel_Class",title="Distance vs price"),use_container_width=True)
    with y:
        g=f.dropna(subset=["Price"]).groupby("Total_Stops",as_index=False).Price.median()
        st.plotly_chart(px.bar(g,x="Total_Stops",y="Price",title="Median price by stops"),use_container_width=True)
    s=booking_summary(f); st.plotly_chart(px.line(s,x="Booking_Window",y="median",markers=True,title="Median price by booking lead time"),use_container_width=True)
    s=f.dropna(subset=["Price"]); s=s.sample(min(10000,len(s)),random_state=42)
    st.plotly_chart(px.histogram(s,x="Price",nbins=50,title="Price distribution"),use_container_width=True)
with t2:
    st.subheader("Model training status")
    st.write("The repository includes a CatBoost training script that fits on all rows with a known price and saves a model artifact.")
    st.warning("A reproducible held-out evaluation is not implemented. Stored metric and feature-importance artifacts do not have enough checked-in provenance to present as validated test results.")
    st.info("CatBoost accepts categorical features directly. The training script models log1p(price); a separate held-out evaluation is needed before making claims about prediction quality.")
with t3:
    r=recommend_booking_window(df,airline,source,destination,travel_class)
    if r["fallback_used"]: st.warning("Limited history for this exact combination; recommendation falls back to the full dataset.")
    st.success(f"Historically cheaper booking window: {r['window']}")
    x,y,z=st.columns(3); x.metric("Median",f"₹{r['median_price']:,.0f}"); y.metric("Mean",f"₹{r['mean_price']:,.0f}"); z.metric("Samples",f"{r['sample_count']:,}")
    st.plotly_chart(px.line(r["summary"],x="Booking_Window",y="median",markers=True,title="Historical median by booking window"),use_container_width=True)
    st.caption("Historical evidence only — not a guaranteed future fare.")
with t4:
    st.subheader("Key findings")
    s=booking_summary(df); lo=s.loc[s["median"].idxmin()]; hi=s.loc[s["median"].idxmax()]
    st.markdown(f"**Booking lead time:** the lowest overall historical median is **{lo['Booking_Window']}** (₹{lo['median']:,.0f}); the highest is **{hi['Booking_Window']}** (₹{hi['median']:,.0f}).")
    st.markdown("**Price is multi-factor:** airline, class, route, distance, duration, stops and lead time interact.")
    st.markdown("**Recommendation design:** a minimum sample threshold prevents a tiny group from producing a misleading recommendation.")
