import os
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import joblib

# Опит за импорт на statsmodels; ако не е инсталиран, деактивирай сезонния анализ
try:
    from statsmodels.tsa.seasonal import seasonal_decompose
    HAS_STATS = True
except ImportError:
    HAS_STATS = False
    st.warning("Не е инсталиран 'statsmodels'. Сезонният анализ е деактивиран.")

# ==================== КОНФИГУРАЦИЯ ====================
ENTSOE_API_KEY = st.secrets.get("ENTSOE_API_KEY")
WEATHER_API_KEY = st.secrets.get("WEATHER_API_KEY")
YEARS_BACK = 3
LOCAL_ZONE = ZoneInfo("Europe/Sofia")
CACHE_TTL = 24 * 60 * 60  # кеширане 1 ден

# ==================== ЗАРЕЖДАНЕ НА ДАННИ ====================
@st.cache_data(ttl=CACHE_TTL)
def fetch_ibex_day_ahead() -> pd.DataFrame:
    """Изтегля ден-напред цени за последните 3 години чрез entsoe-py."""
    try:
        from entsoe import EntsoePandasClient
    except ImportError:
        st.error("Добавете 'entsoe-py' в requirements.txt.")
        return pd.DataFrame()
    client = EntsoePandasClient(api_key=ENTSOE_API_KEY)
    end = pd.Timestamp.utcnow().floor('D') - pd.Timedelta(days=1)
    start = end - pd.Timedelta(days=YEARS_BACK*365)
    # ENTSO-E очаква UTC timestamps
    start_utc = start.tz_localize(timezone.utc)
    end_utc = end.tz_localize(timezone.utc) + pd.Timedelta(hours=23)
    try:
        series = client.query_day_ahead_prices(
            country_code='BG', start=start_utc, end=end_utc
        )
    except Exception as e:
        st.error(f"ENTSO-E API error: {e}")
        return pd.DataFrame()
    if series.empty:
        st.error("Не са намерени ценови данни.")
        return pd.DataFrame()
    df = series.reset_index().rename(columns={'time': 'datetime', series.name: 'price'})
    df['datetime'] = df['datetime'].dt.tz_convert(LOCAL_ZONE)
    df['date'] = df['datetime'].dt.date
    df['hour'] = df['datetime'].dt.hour
    pivot = df.pivot(index='date', columns='hour', values='price')
    pivot.columns = [f'hour_{h}' for h in pivot.columns]
    return pivot.reset_index()

@st.cache_data(ttl=CACHE_TTL)
def fetch_weather_history(lat: float, lon: float) -> pd.DataFrame:
    """Изтегля исторически дневни метео данни от OpenWeatherMap за същия период."""
    end = datetime.now(LOCAL_ZONE).date() - timedelta(days=1)
    start = end - timedelta(days=YEARS_BACK*365)
    records = []
    for single_date in pd.date_range(start, end, freq='D'):
        dt_local = pd.Timestamp(single_date, tz=LOCAL_ZONE).replace(hour=12)
        ts = int(dt_local.astimezone(timezone.utc).timestamp())
        url = "https://api.openweathermap.org/data/2.5/onecall/timemachine"
        params = {
            'lat': lat, 'lon': lon, 'dt': ts,
            'appid': WEATHER_API_KEY, 'units': 'metric'
        }
        data = requests.get(url, params=params).json()
        temps = [h['temp'] for h in data.get('hourly', [])]
        winds = [h['wind_speed'] for h in data.get('hourly', [])]
        uvis = [h.get('uvi', 0) for h in data.get('hourly', [])]
        if temps:
            records.append({
                'date': single_date.date(),
                'avg_temp': np.mean(temps),
                'avg_wind_speed': np.mean(winds),
                'solar_radiation': np.mean(uvis)
            })
    return pd.DataFrame(records)

@st.cache_data(ttl=CACHE_TTL)
def fetch_weather_forecast(lat: float, lon: float) -> pd.DataFrame:
    """Изтегля 7-дневна прогноза от OpenWeatherMap."""
    url = "https://api.openweathermap.org/data/2.5/onecall"
    params = { 'lat': lat, 'lon': lon, 'exclude': 'current,minutely,hourly,alerts', 'appid': WEATHER_API_KEY, 'units': 'metric' }
    data = requests.get(url, params=params).json().get('daily', [])
    rec = []
    for d in data:
        dt_local = datetime.fromtimestamp(d['dt'], tz=timezone.utc).astimezone(LOCAL_ZONE)
        rec.append({
            'date': dt_local.date(),
            'avg_temp': d['temp']['day'],
            'avg_wind_speed': d['wind_speed'],
            'solar_radiation': d.get('uvi', 0)
        })
    return pd.DataFrame(rec)

# ==================== ПРЕДОБРАБОТКА И АНАЛИЗ ====================
def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['dayofyear'] = df['date'].dt.dayofyear
    df['dayofweek'] = df['date'].dt.dayofweek
    df['is_weekend'] = df['dayofweek'] >= 5
    # лагове
    for lag in [1, 7, 365]:
        for h in range(24): df[f'hour_{h}_lag{lag}'] = df[f'hour_{h}'].shift(lag)
    for col in ['avg_temp', 'avg_wind_speed', 'solar_radiation']:
        df[f'{col}_lag1'] = df[col].shift(1)
    df = df.dropna().reset_index(drop=True)
    return df

# ==================== МНОГОИЗХОДНО ОБУЧЕНИЕ ====================
def prepare_dataset(df: pd.DataFrame, horizon_days=7):
    df_feat = create_features(df)
    # target cols
    y_cols = [f'hour_{h}_lead{d}' for d in range(1, horizon_days+1) for h in range(24)]
    for d in range(1, horizon_days+1):
        shifted = df_feat[[f'hour_{h}' for h in range(24)]].shift(-24*(d-1))
        shifted.columns = [f'hour_{h}_lead{d}' for h in range(24)]
        df_feat = pd.concat([df_feat, shifted], axis=1)
    df_feat = df_feat.dropna().reset_index(drop=True)
    X = df_feat[[c for c in df_feat.columns if 'lead' not in c and c!='date']]
    y = df_feat[y_cols]
    return X, y

# ==================== ПРОГНОЗА ====================
def forecast_week(df_hist: pd.DataFrame, model, scaler):
    X, _ = prepare_dataset(df_hist)
    last_X = X.tail(1)
    preds = model.predict(scaler.transform(last_X))
    base = df_hist['date'].max()
    hours = []
    prices = preds[0].tolist()
    for i, price in enumerate(prices):
        day = i // 24 + 1
        hour = i % 24
        dt = datetime.combine(base + timedelta(days=day), datetime.min.time()) + timedelta(hours=hour)
        hours.append(dt)
    return pd.DataFrame({'datetime': hours, 'pred_price': prices})

# ==================== STREAMLIT UI ====================
def main():
    st.title("Прогноза на електрически цени "Den Ahead" за България – 7 дни напред")
    # Задаваме координати
    lat, lon = 42.7, 23.3

    # Зареждане данни
    df_price = fetch_ibex_day_ahead()
    df_hist_weather = fetch_weather_history(lat, lon)
    df_weather_fc = fetch_weather_forecast(lat, lon)

    if df_price.empty:
        st.error("Неуспех при зареждане на ценови данни.")
        return

    # Сливане
    df = df_price.merge(df_hist_weather, on='date', how='left')

    # Анализ на връзката
    st.subheader("Корелация между цена и метео")
    corr = df[[c for c in df.columns if c.startswith('hour_')] + ['avg_temp', 'avg_wind_speed', 'solar_radiation']].corr()
    st.write(corr)
    if HAS_STATS:
        df['mean_price'] = df[[f'hour_{h}' for h in range(24)]].mean(axis=1)
        res = seasonal_decompose(df.set_index('date')['mean_price'], model='additive', period=365)
        st.line_chart(res.trend)
        st.line_chart(res.seasonal)

    # Обучение и прогноза
    df_full = pd.concat([df_price, df_weather_fc], ignore_index=True)
    X, y = prepare_dataset(df_full)
    model, scaler = train_multioutput_model(X, y)
    df_pred = forecast_week(df_full, model, scaler)

    st.subheader("Прогнозирани цени (почасово) за следващата седмица")
    df_pred['datetime'] = pd.to_datetime(df_pred['datetime'])
    st.line_chart(df_pred.set_index('datetime')['pred_price'])
    st.dataframe(df_pred)

if __name__ == '__main__':
    main()
