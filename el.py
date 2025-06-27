import os
import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import joblib

# ==================== КОНФИГУРАЦИЯ ====================
ENTSOE_API_KEY = st.secrets.get("ENTSOE_API_KEY")
WEATHER_API_KEY = st.secrets.get("WEATHER_API_KEY")

# ==================== ФУНКЦИИ ЗА ЗАРЕЖДАНЕ НА ДАННИ ====================

def fetch_ibex_day_ahead(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Извлича исторически цени Ден Напред от ENTSO-E API.
    start_date/end_date във формат YYYYMMDD.
    Връща DataFrame с колони: date, hour_0...hour_23.
    """
    url = "https://transparency.entsoe.eu/api"
    params = {
        'documentType': 'A44',
        'in_Domain': '10YBG-ESO------Y',
        'out_Domain': '10YBG-ESO------Y',
        'periodStart': start_date + '0000',
        'periodEnd': end_date + '2300',
        'securityToken': ENTSOE_API_KEY
    }
    response = requests.get(url, params=params)
    # Потребителят трябва да парсне XML отговор. Тук приемаме, че е конвертиран до CSV.
    df = pd.read_xml(response.content, xpath='//TimeSeries')
    # Примерна обработка за трансформиране в wide формат
    df['date'] = pd.to_datetime(df['timeStamp']).dt.date
    df['hour'] = pd.to_datetime(df['timeStamp']).dt.hour
    df_pivot = df.pivot_table(index='date', columns='hour', values='price.amount')
    df_pivot.columns = [f'hour_{h}' for h in df_pivot.columns]
    df_pivot = df_pivot.reset_index()
    return df_pivot


def fetch_weather_history(lat: float, lon: float, start: datetime, end: datetime) -> pd.DataFrame:
    """
    Изтегля агрегирани дневни данни за температура, вятър и радиация.
    Използва OpenWeatherMap One Call Timemachine.
    """
    records = []
    date = start
    while date <= end:
        unix_ts = int(date.timestamp())
        url = f"https://api.openweathermap.org/data/2.5/onecall/timemachine"
        params = {
            'lat': lat,
            'lon': lon,
            'dt': unix_ts,
            'appid': WEATHER_API_KEY,
            'units': 'metric'
        }
        r = requests.get(url, params=params).json()
        temps = [h['temp'] for h in r['hourly']]
        winds = [h['wind_speed'] for h in r['hourly']]
        solar = [h.get('uvi', 0) for h in r['hourly']]
        records.append({
            'date': date,
            'avg_temp': np.mean(temps),
            'avg_wind_speed': np.mean(winds),
            'solar_radiation': np.mean(solar)
        })
        date += timedelta(days=1)
    return pd.DataFrame(records)


def fetch_weather_forecast(lat: float, lon: float) -> pd.DataFrame:
    """
    Изтегля прогноза за следващите 7 дни.
    """
    url = f"https://api.openweathermap.org/data/2.5/onecall"
    params = {
        'lat': lat,
        'lon': lon,
        'exclude': 'current,minutely,hourly,alerts',
        'appid': WEATHER_API_KEY,
        'units': 'metric'
    }
    j = requests.get(url, params=params).json()
    records = []
    for d in j['daily']:
        records.append({
            'date': datetime.fromtimestamp(d['dt']).date(),
            'avg_temp': d['temp']['day'],
            'avg_wind_speed': d['wind_speed'],
            'solar_radiation': d.get('uvi', 0)
        })
    return pd.DataFrame(records)


def load_ibex_and_weather(start_str: str, end_str: str, lat: float, lon: float) -> pd.DataFrame:
    # IBEX данни
    df_price = fetch_ibex_day_ahead(start_str, end_str)
    # Исторически метео
    start_dt = datetime.strptime(start_str, '%Y%m%d')
    end_dt = datetime.strptime(end_str, '%Y%m%d')
    df_weather_hist = fetch_weather_history(lat, lon, start_dt, end_dt)
    # Merge
    df = df_price.merge(df_weather_hist, on='date', how='left')
    return df

# Останалите функции (create_features, prepare_dataset, train_model, forecast_next_day) остават без промяна.
# ==================== STREAMLIT UI ====================

def main():
    st.title("IBEX Ден Напред Прогноза на Цени с Автоматично Зареждане на Данни")
    years = st.slider("Колко години назад?", 1, 5, 3)
    lat = st.number_input("Ширина (latitude)", value=42.7)
    lon = st.number_input("Дължина (longitude)", value=23.3)

    end = datetime.utcnow().date() - timedelta(days=1)
    start = end - timedelta(days=365 * years)
    start_str = start.strftime('%Y%m%d')
    end_str = end.strftime('%Y%m%d')

    with st.spinner("Зареждане на IBEX и метео данни..."):
        df = load_ibex_and_weather(start_str, end_str, lat, lon)
    st.success("Данните са заредени")
    st.write(df.head())
    
    df_feat = create_features(df)
    st.write(f"Feature сет: {df_feat.shape[0]} реда, {df_feat.shape[1]} колони")

    hour = st.slider("Изберете час за прогноза", 0, 23, 0)
    if st.button("Обучи модел и прогноза"):
        X, y = prepare_dataset(df_feat, target_hour=hour)
        model, scaler, mae, rmse = train_model(X, y)
        st.success(f"Обучено! CV MAE: {mae:.3f}, RMSE: {rmse:.3f}")

        # Прогнозиране с weather forecast
        df_fc_weather = fetch_weather_forecast(lat, lon)
        df_full = pd.concat([df_price:=df[['date'] + [f'hour_{h}' for h in range(24)]], df_fc_weather], ignore_index=True)
        preds = forecast_next_day(df_full, model, scaler)
        df_pred = pd.DataFrame({'hour': range(24), 'pred_price': preds})
        st.line_chart(df_pred.set_index('hour'))
        st.write(df_pred)

if __name__ == "__main__":
    main()


