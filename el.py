import os
import streamlit as st
import pandas as pd
import numpy as np
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
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
# Зона за България (CET/CEST)
LOCAL_ZONE = ZoneInfo("Europe/Sofia")
CACHE_TTL = 24 * 60 * 60  # 1 ден

# ==================== ФУНКЦИИ ЗА ЗАРЕЖДАНЕ НА ДАННИ ====================
@st.cache_data(ttl=CACHE_TTL)
def fetch_ibex_day_ahead(start_date: str, end_date: str) -> pd.DataFrame:
    """Извлича исторически цени Ден Напред от ENTSO-E API, конвертира в локално време CET."""
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
    try:
        root = ET.fromstring(r.content)
    except ET.ParseError:
        st.error("Грешка при парсване на XML от ENTSO-E API.")
        return pd.DataFrame()
    ns = {None: root.tag.split('}')[0].strip('{')}
    data = []
    for ts in root.findall('.//{'+ns[None]+'}TimeSeries'):
        period = ts.find('.//{'+ns[None]+'}Period')
        start_ts = period.find('{'+ns[None]+'}timeInterval/{'+ns[None]+'}start').text
        dt_start = datetime.fromisoformat(start_ts).replace(tzinfo=timezone.utc)
        for pt in period.findall('{'+ns[None]+'}Point'):
            pos = int(pt.find('{'+ns[None]+'}position').text)
            price = float(pt.find('{'+ns[None]+'}price.amount').text)
            dt_utc = dt_start + timedelta(hours=pos-1)
            dt_local = dt_utc.astimezone(LOCAL_ZONE)
            data.append({'datetime': dt_local, 'price': price})
    df = pd.DataFrame(data)
    if df.empty:
        st.error("Няма данни от ENTSO-E API за посочения период.")
        return df
    df['date'] = df['datetime'].dt.date
    df['hour'] = df['datetime'].dt.hour
    df_pivot = df.pivot(index='date', columns='hour', values='price')
    df_pivot.columns = [f'hour_{h}' for h in df_pivot.columns]
    return df_pivot.reset_index()

@st.cache_data(ttl=CACHE_TTL)
def fetch_weather_history(lat: float, lon: float, start: datetime, end: datetime) -> pd.DataFrame:
    """Изтегля исторически дневни данни за метео в локално време."""
    records = []
    date = start
    while date <= end:
        dt_local_noon = datetime(year=date.year, month=date.month, day=date.day, hour=12, tzinfo=LOCAL_ZONE)
        ts = int(dt_local_noon.astimezone(timezone.utc).timestamp())
        url = "https://api.openweathermap.org/data/2.5/onecall/timemachine"
        params = {'lat': lat, 'lon': lon, 'dt': ts, 'appid': WEATHER_API_KEY, 'units': 'metric'}
        j = requests.get(url, params=params).json()
        temps, winds, uvis = [], [], []
        for h in j.get('hourly', []):
            dt_loc = datetime.fromtimestamp(h['dt'], tz=timezone.utc).astimezone(LOCAL_ZONE)
            if dt_loc.date() == date:
                temps.append(h['temp']); winds.append(h['wind_speed']); uvis.append(h.get('uvi', 0))
        if temps:
            records.append({'date': date, 'avg_temp': np.mean(temps), 'avg_wind_speed': np.mean(winds), 'solar_radiation': np.mean(uvis)})
        date += timedelta(days=1)
    return pd.DataFrame(records)

@st.cache_data(ttl=CACHE_TTL)
def fetch_weather_forecast(lat: float, lon: float) -> pd.DataFrame:
    """Изтегля 7-дневна прогноза и конвертира дати към локална зона."""
    url = "https://api.openweathermap.org/data/2.5/onecall"
    params = {'lat': lat, 'lon': lon, 'exclude': 'current,minutely,hourly,alerts', 'appid': WEATHER_API_KEY, 'units': 'metric'}
    j = requests.get(url, params=params).json()
    rec = []
    for d in j.get('daily', []):
        dt_local = datetime.fromtimestamp(d['dt'], tz=timezone.utc).astimezone(LOCAL_ZONE)
        rec.append({'date': dt_local.date(), 'avg_temp': d['temp']['day'], 'avg_wind_speed': d['wind_speed'], 'solar_radiation': d.get('uvi', 0)})
    return pd.DataFrame(rec)

# ==================== ПРЕДОБРАБОТКА ====================
def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['dayofyear'] = df['date'].dt.dayofyear
    df['dayofweek'] = df['date'].dt.dayofweek
    df['is_weekend'] = df['dayofweek'] >= 5
    for lag in [1, 7, 365]:
        for h in range(24): df[f'hour_{h}_lag{lag}'] = df[f'hour_{h}'].shift(lag)
    for col in ['avg_temp', 'avg_wind_speed', 'solar_radiation']:
        df[f'{col}_lag1'] = df[col].shift(1)
    return df.dropna().reset_index(drop=True)

# ==================== ПОДГОТВКА НА НАБОР ====================
def prepare_dataset(df_feat: pd.DataFrame, horizon_days=7) -> tuple[pd.DataFrame, pd.DataFrame]:
    y_cols = [f'hour_{h}_lead{d}' for d in range(1, horizon_days+1) for h in range(24)]
    df = df_feat.copy()
    for d in range(1, horizon_days+1):
        shift_df = df_feat[[f'hour_{h}' for h in range(24)]].shift(-24*(d-1))
        shift_df.columns = [f'hour_{h}_lead{d}' for h in range(24)]
        df = pd.concat([df, shift_df], axis=1)
    df = df.dropna().reset_index(drop=True)
    X = df[[c for c in df.columns if 'lead' not in c and c != 'date']]
    y = df[y_cols]
    return X, y

# ==================== ОБУЧЕНИЕ НА МОДЕЛ ====================
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
    st.write(f"CV MAE: {np.mean(maes):.3f}")
    return model, scaler

# ==================== ПРОГНОЗА ЗА СЕДМИЦА ====================
def forecast_week(df_hist: pd.DataFrame, model, scaler) -> pd.DataFrame:
    df_feat = create_features(df_hist)
    X0 = df_feat.drop(columns=['date'] + [f'hour_{h}' for h in range(24)])
    Xs = scaler.transform(X0.tail(1))
    pred = model.predict(Xs)[0]
    hours, prices = [], []
    base_date = df_hist['date'].max()
    for d in range(1, 8):
        for h in range(24):
            dt = datetime.combine(base_date + timedelta(days=d), datetime.min.time()) + timedelta(hours=h)
            hours.append(dt)
            prices.append(pred[(d-1)*24 + h])
    return pd.DataFrame({'datetime': hours, 'pred_price': prices})

# ==================== АНАЛИЗ НА ВРЪЗКАТА ====================
def run_analysis(df: pd.DataFrame):
    df2 = df.copy()
    df2['date'] = pd.to_datetime(df2['date'])
    corr = df2[[ 'avg_temp','avg_wind_speed','solar_radiation'] + [f'hour_{h}' for h in range(24)]].corr()
    trend, seasonal = None, None
    if HAS_STATS:
        df2['mean_price'] = df2[[f'hour_{h}' for h in range(24)]].mean(axis=1)
        res = seasonal_decompose(df2.set_index('date')['mean_price'], model='additive', period=365)
        trend, seasonal = res.trend, res.seasonal
    return corr, trend, seasonal

# ==================== STREAMLIT UI ====================
def main():
    st.title("IBEX 7-дневна прогноза DEN AHEAD с Метео и Сезонни Тенденции")
    end = datetime.utcnow().date() - timedelta(days=1)
    start = end - timedelta(days=YEARS_BACK * 365)
    start_str, end_str = start.strftime('%Y%m%d'), end.strftime('%Y%m%d')
    with st.spinner("Зареждане на данни..."):
        df_price = fetch_ibex_day_ahead(start_str, end_str)
        if df_price.empty:
            return
        df_hist_weather = fetch_weather_history(42.7, 23.3, start, end)
        df = df_price.merge(df_hist_weather, on='date', how='left')
    st.success("Данните заредени")
    st.subheader("Анализ на връзката между цена и метео")
    corr, trend, seasonal = run_analysis(df)
    st.write(corr)
    if HAS_STATS and trend is not None:
        st.line_chart(trend)
        st.line_chart(seasonal)

    df_full = pd.concat([df, fetch_weather_forecast(42.7, 23.3)], ignore_index=True)
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

