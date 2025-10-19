import pandas as pd
from haversine import haversine_vector, Unit
import logging
import time

RANDOM_STATE = 42

def fillna(df: pd.DataFrame) -> pd.DataFrame:
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    cat_cols = df.select_dtypes(include=['object']).columns

    for col in num_cols:
        df[col].fillna(df[col].median(), inplace=True)

    for col in cat_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)

    return df

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df['transaction_time'] = pd.to_datetime(df['transaction_time'])

    df['tx_hour'] = df['transaction_time'].dt.hour
    df['tx_minute'] = df['transaction_time'].dt.minute
    df['tx_dow'] = df['transaction_time'].dt.dayofweek  # 0=Mon
    df['tx_day'] = df['transaction_time'].dt.day
    df['tx_month'] = df['transaction_time'].dt.month
    df['is_weekend'] = df['tx_dow'].isin([5, 6]).astype(int)
    df['tx_unix'] = (df['transaction_time'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')

    df = df.drop(columns=['transaction_time'])

    return df

def add_distance_features(df: pd.DataFrame) -> pd.DataFrame:
    df['distance'] = haversine_vector(
        list(zip(df['lat'], df['lon'])),
        list(zip(df['merchant_lat'], df['merchant_lon'])),
        unit=Unit.KILOMETERS
    )
    return df.drop(columns=['lat', 'lon', 'merchant_lat', 'merchant_lon'])

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("Starting data preprocessing...")
    start_time = time.time()

    df = fillna(df)
    df = add_distance_features(df)
    df = add_time_features(df)

    end_time = time.time()
    logging.info(f"Preprocessing finished in {end_time - start_time:.2f} seconds.")
    return df