import numpy as np
from catboost import CatBoostRegressor
from .data_utils import CAT_FEATURES,MODEL_FEATURES,load_data
def prepare_features(df):
    x=df[MODEL_FEATURES].copy()
    for c in CAT_FEATURES: x[c]=x[c].fillna("Unknown").astype(str)
    return x
def load_model(path="models/price_model.cbm"):
    m=CatBoostRegressor(); m.load_model(path); return m
def train_and_save(data_path="data/flight_pricing_dataset.csv",model_path="models/price_model.cbm"):
    d=load_data(data_path); d=d[d.Price.notna()].copy(); x=prepare_features(d); y=np.log1p(d.Price.astype(float))
    m=CatBoostRegressor(iterations=350,depth=8,learning_rate=.08,loss_function="RMSE",random_seed=42,verbose=False); m.fit(x,y,cat_features=CAT_FEATURES); m.save_model(model_path); return m
