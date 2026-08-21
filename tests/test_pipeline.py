import pandas as pd
from src.data_utils import clean_dataframe
from src.booking_analysis import recommend_booking_window
def test_schema_and_cleaning():
    r=pd.read_csv("data/flight_pricing_dataset.csv"); d=clean_dataframe(r)
    assert len(d)==len(r) and d.Price.notna().sum()>90000 and "Duration_Minutes" in d
def test_booking_recommendation():
    d=clean_dataframe(pd.read_csv("data/flight_pricing_dataset.csv")); x=recommend_booking_window(d)
    assert x["sample_count"]>=20 and x["median_price"]>0
