import pandas as pd
BINS=[-1,2,7,14,30,60,90,180,3650]
LABELS=["0–2 days","3–7 days","8–14 days","15–30 days","31–60 days","61–90 days","91–180 days","181+ days"]
def booking_summary(df):
    w=df[df.Price.notna() & df.Days_Before_Departure.notna()].copy(); w["Booking_Window"]=pd.cut(w.Days_Before_Departure,BINS,labels=LABELS)
    return w.groupby("Booking_Window",observed=True).Price.agg(["count","median","mean"]).reset_index()
def recommend_booking_window(df,airline=None,source=None,destination=None,travel_class=None):
    w=df[df.Price.notna() & df.Days_Before_Departure.notna()].copy()
    for col,val in [("Airline",airline),("Source",source),("Destination",destination),("Travel_Class",travel_class)]:
        if val and val!="All": w=w[w[col].astype(str)==str(val)]
    fallback=len(w)<30
    if fallback: w=df[df.Price.notna() & df.Days_Before_Departure.notna()].copy()
    w["Booking_Window"]=pd.cut(w.Days_Before_Departure,BINS,labels=LABELS); s=w.groupby("Booking_Window",observed=True).Price.agg(["count","median","mean"]).reset_index()
    e=s[s["count"]>=20] if (s["count"]>=20).any() else s; b=e.loc[e["median"].idxmin()]
    return {"window":str(b["Booking_Window"]),"median_price":float(b["median"]),"mean_price":float(b["mean"]),"sample_count":int(b["count"]),"fallback_used":fallback,"summary":s}
