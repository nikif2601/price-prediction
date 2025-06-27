import os
import streamlit as st
import pandas as pd
import numpy as np
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import joblib

# Опит за импорт на statsmodels; ако не е инсталиран, деактивирай сезонния анализ
try:
    from statsmodels.tsa.seasonal import seasonal_decompose
    HAS_STATS = True
except ImportError:
    HAS_STATS = False
    st.warning("Модулът statsmodels не е инсталиран, сезонният анализ е деактивиран. Добавете 'statsmodels' в requirements.txt и деплойнете отново.")

# ==================== КОНФИГУРАЦИЯ ====================
ENTSOE_API_KEY = st.secrets.get("ENTSOE_API_KEY")
WEATHER_API_KEY = st.secrets.get("WEATHER_API_KEY")
YEARS_BACK = 3
LAT, LON = 42.7, 23.3  # България
CACHE_TTL = 24 * 60 * 60  # 1 ден

# ==================== ФУНКЦИИ ЗА ЗАРЕЖДАНЕ НА ДАННИ ====================
@st.cache_data(ttl=CACHE_TTL)
def fetch_ibex_day_ahead(start_date: str, end_date: str) -> pd.DataFrame:
    """Извлича исторически цени Ден Напред от ENTSO-E API."""
    url = "https://transparency.entsoe.eu/api"
    params = {
        'documentType': 'A44',
        'in_Domain': '10YBG-ESO------Y',
        'out_Domain': '10YBG-ESO------Y',
        'periodStart': start_date + '0000',
        'periodEnd': end_date + '2300',
        'securityToken': ENTSOE_API_KEY
    }
    r = requests.get(url, params=params)
    # Опит за парсване с pandas; изисква lxml
    try:
        df_raw = pd.read_xml(r.content, xpath='//TimeSeries')
    except ImportError:
        st.error("Моля, добавете 'lxml' в requirements.txt и презаредете приложението.")
        return pd.DataFrame()
    df_raw['date'] = pd.to_datetime(df_raw['timeStamp']).dt.date
    df_raw['hour'] = pd.to_datetime(df_raw['timeStamp']).dt.hour
    df = df_raw.pivot_table(index='date', columns='hour', values='price.amount')
    df.columns = [f'hour_{h}' for h in df.columns]
    return df.reset_index()

@st.cache_data(ttl=CACHE_TTL)
def fetch_weather_history(lat: float, lon: float, start: datetime, end: datetime) -> pd.DataFrame:
    records = []
    date = start
    while date <= end:
        ts = int(date.replace(hour=12).timestamp())
        url = "https://api.openweathermap.org/data/2.5/onecall/timemachine"
        params = {'lat': lat, 'lon': lon, 'dt': ts, 'appid': WEATHER_API_KEY, 'units': 'metric'}
        j = requests.get(url, params=params).json()
        temps = [h['temp'] for h in j.get('hourly', [])]
        winds = [h['wind_speed'] for h in j.get('hourly', [])]
        uvis = [h.get('uvi', 0) for h in j.get('hourly', [])]
        if temps:
            records.append({'date': date, 'avg_temp': np.mean(temps), 'avg_wind_speed': np.mean(winds), 'solar_radiation': np.mean(uvis)})
        date += timedelta(days=1)
    return pd.DataFrame(records)

@st.cache_data(ttl=CACHE_TTL)
def fetch_weather_forecast(lat: float, lon: float) -> pd.DataFrame:
    url = "https://api.openweathermap.org/data/2.5/onecall"
    params = {'lat': lat, 'lon': lon, 'exclude': 'current,minutely,hourly,alerts', 'appid': WEATHER_API_KEY, 'units': 'metric'}
    j = requests.get(url, params=params).json()
    rec = []
    for d in j.get('daily', []):
        rec.append({'date': datetime.fromtimestamp(d['dt']).date(), 'avg_temp': d['temp']['day'], 'avg_wind_speed': d['wind_speed'], 'solar_radiation': d.get('uvi', 0)})
    return pd.DataFrame(rec)

# ==================== ПРЕДОБРАБОТКА ====================
def create_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data['date'] = pd.to_datetime(data['date'])
    data['dayofyear'] = data['date'].dt.dayofyear
    data['dayofweek'] = data['date'].dt.dayofweek
    data['is_weekend'] = data['dayofweek'] >= 5
    for lag in [1, 7, 365]:
        for h in range(24):
            data[f'hour_{h}_lag{lag}'] = data[f'hour_{h}'].shift(lag)
    for col in ['avg_temp', 'avg_wind_speed', 'solar_radiation']:
        data[f'{col}_lag1'] = data[col].shift(1)
    data = data.dropna().reset_index(drop=True)
    return data

# ==================== ПОДГОТВКА НА НАБОР ====================
def prepare_dataset(df_feat: pd.DataFrame, horizon_days=7) -> tuple:
    y_cols = []
    for d in range(1, horizon_days+1):
        for h in range(24):
            y_cols.append(f'hour_{h}_lead{d}')
    df = df_feat.copy()
    for d in range(1, horizon_days+1):
        df_shift = df_feat[[f'hour_{h}' for h in range(24)]].shift(-24*(d-1))
        df_shift.columns = [f'hour_{h}_lead{d}' for h in range(24)]
        df = pd.concat([df, df_shift], axis=1)
    df = df.dropna().reset_index(drop=True)
    X = df[[c for c in df.columns if 'lead' not in c and c != 'date']]
    y = df[y_cols]
    return X, y

# ==================== ОБУЧЕНИЕ НА МОДЕЛ

def train_multioutput_model(X: pd.DataFrame, y: pd.DataFrame):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    tscv = TimeSeriesSplit(n_splits=5)
    maes = []
    for tr, va in tscv.split(Xs):
        m = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=4)
        m.fit(Xs[tr], y.iloc[tr])
        p = m.predict(Xs[va])
        maes.append(mean_absolute_error(y.iloc[va], p))
    model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=4)
    model.fit(Xs, y)
    joblib.dump(model, 'multi_day_model.pkl')
    joblib.dump(scaler, 'multi_day_scaler.pkl')
    print(f"CV MAE: {np.mean(maes):.3f}")
    return model, scaler

# ==================== ПРОГНОЗА ЗА СЕДМИЦА

def forecast_week(df_hist: pd.DataFrame, model, scaler):
    df_feat = create_features(df_hist)
    X0 = df_feat.drop(columns=['date'] + [f'hour_{h}' for h in range(24)])
    Xs = scaler.transform(X0.tail(1))
    pred = model.predict(Xs)[0]
    hours, prices = [], []
    today = df_hist['date'].max()
    for d in range(1, 8):
        for h in range(24):
            hours.append(datetime.combine(today + timedelta(days=d), datetime.min.time()) + timedelta(hours=h))
            prices.append(pred[(d-1)*24 + h])
    return pd.DataFrame({'datetime': hours, 'pred_price': prices})

# ==================== АНАЛИЗ НА ВРЪЗКАТА

def run_analysis(df: pd.DataFrame):
    df2 = df.copy()
    df2['date'] = pd.to_datetime(df2['date'])
    corr = df2[['avg_temp', 'avg_wind_speed', 'solar_radiation'] + [f'hour_{h}' for h in range(24)]].corr()
    trend, seasonal = None, None
    if HAS_STATS:
        df2['mean_price'] = df2[[f'hour_{h}' for h in range(24)]].mean(axis=1)
        res = seasonal_decompose(df2.set_index('date')['mean_price'], model='additive', period=365)
        trend, seasonal = res.trend, res.seasonal
    return corr, trend, seasonal

# ==================== STREAMLIT UI

def main():
    st.title("IBEX 7-дневна прогноза DEN AHEAD с Метео и Сезонни Тенденции")
    end = datetime.utcnow().date() - timedelta(days=1)
    start = end - timedelta(days=YEARS_BACK * 365)
    start_str, end_str = start.strftime('%Y%m%d'), end.strftime('%Y%m%d')
    with st.spinner("Зареждане на данни..."):
        df_price = fetch_ibex_day_ahead(start_str, end_str)
        df_hist_weather = fetch_weather_history(LAT, LON, start, end)
        df = df_price.merge(df_hist_weather, on='date', how='left')
    st.success("Данните заредени")
    st.subheader("Анализ на връзката между цена и метео")
    corr, trend, seasonal = run_analysis(df)
    st.write(corr)
    if HAS_STATS:
        st.line_chart(trend)
        st.line_chart(seasonal)
    else:
        st.info("Сезонният анализ е деактивиран поради липса на statsmodels.")

    df_full = pd.concat([df, fetch_weather_forecast(LAT, LON)], ignore_index=True)
    df_full['date'] = pd.to_datetime(df_full['date'])
    df_feat = create_features(df_full)
    X, y = prepare_dataset(df_feat)
    model, scaler = train_multioutput_model(X, y)
    df_pred = forecast_week(df_full, model, scaler)
    st.subheader("Прогнозирани цени за следващата седмица (почасово)")
    df_pred['datetime'] = pd.to_datetime(df_pred['datetime'])
    df_pred = df_pred.set_index('datetime')
    st.line_chart(df_pred['pred_price'])
    st.dataframe(df_pred)

if __name__ == "__main__":
    main()




