import pandas as pd
from src.data_utils import clean_dataframe
from src.booking_analysis import recommend_booking_window
def test_schema_and_cleaning():
    r=pd.read_csv("data/flight_pricing_dataset.csv"); d=clean_dataframe(r)
    assert len(d)==len(r) and d.Price.notna().sum()>90000 and "Duration_Minutes" in d
def test_booking_recommendation():
    d=clean_dataframe(pd.read_csv("data/flight_pricing_dataset.csv")); x=recommend_booking_window(d)
    assert x["sample_count"]>=20 and x["median_price"]>0

def test_departure_day_of_year_values():
    raw = pd.DataFrame({
        "Airline": ["Example"] * 3,
        "Source": ["A"] * 3,
        "Destination": ["B"] * 3,
        "Travel_Class": ["Economy"] * 3,
        "Season": ["Winter"] * 3,
        "Weekday": ["Thursday"] * 3,
        "Aircraft_Type": ["Example"] * 3,
        "Booking_Channel": ["Direct"] * 3,
        "Price": [100, 200, 300],
        "Distance_km": [1000] * 3,
        "Days_Before_Departure": [30] * 3,
        "Passenger_Count": [1] * 3,
        "Duration": ["2h"] * 3,
        "Total_Stops": [0] * 3,
        "Departure_Date": ["2024-02-29", "2025-12-31", "invalid"],
    })

    cleaned = clean_dataframe(raw)

    assert cleaned["Departure_DayOfYear"].iloc[:2].tolist() == [60, 365]
    assert pd.isna(cleaned["Departure_DayOfYear"].iloc[2])
