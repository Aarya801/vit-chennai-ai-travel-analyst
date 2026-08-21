import re, numpy as np, pandas as pd
CAT_FEATURES=["Airline","Source","Destination","Travel_Class","Season","Weekday","Aircraft_Type","Booking_Channel"]
NUM_FEATURES=["Distance_km","Days_Before_Departure","Passenger_Count","Duration_Minutes","Total_Stops","Departure_Month","Departure_DayOfYear"]
MODEL_FEATURES=CAT_FEATURES+NUM_FEATURES
def parse_numeric(x):
    if pd.isna(x): return np.nan
    s=str(x).strip().lower().replace(",","")
    try: return float(s)
    except:
        m=re.findall(r"[-+]?\d*\.?\d+",s); return float(m[0]) if m else np.nan
def parse_duration_minutes(x):
    if pd.isna(x): return np.nan
    s=str(x).strip().lower()
    if re.fullmatch(r"\d+(?:\.\d+)?",s): return float(s)*60
    h=re.search(r"(\d+(?:\.\d+)?)\s*h",s); m=re.search(r"(\d+(?:\.\d+)?)\s*m",s)
    v=(float(h.group(1))*60 if h else 0)+(float(m.group(1)) if m else 0)
    return v if h or m else np.nan
def clean_dataframe(df):
    o=df.copy()
    for c in CAT_FEATURES: o[c]=o[c].fillna("Unknown").astype(str).str.strip()
    o["Price"]=o["Price"].map(parse_numeric); o["Distance_km"]=o["Distance_km"].map(parse_numeric); o["Days_Before_Departure"]=o["Days_Before_Departure"].map(parse_numeric)
    o["Passenger_Count"]=o["Passenger_Count"].map(lambda x: {"one":1,"two":2,"three":3,"four":4,"five":5,"six":6}.get(str(x).strip().lower(),parse_numeric(x)))
    o["Duration_Minutes"]=o["Duration"].map(parse_duration_minutes)
    o["Total_Stops"]=o["Total_Stops"].map(lambda x: 0.0 if pd.isna(x) else (0.0 if "non" in str(x).lower() else parse_numeric(x)))
    d=pd.to_datetime(o["Departure_Date"],errors="coerce"); o["Departure_Month"]=d.dt.month; o["Departure_DayOfYear"]=d.dt.dayofyear; o["Departure_Year"]=d.dt.year
    return o
def load_data(path="data/flight_pricing_dataset.csv"): return clean_dataframe(pd.read_csv(path))
