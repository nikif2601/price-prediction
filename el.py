import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from entsoe import EntsoePandasClient
import requests
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import joblib

# Опит за импортиране на seasonal_decompose
try:
    from statsmodels.tsa.seasonal import seasonal_decompose
    HAS_STATS = True
except ImportError:
    HAS_STATS = False
    st.warning("Не е инсталиран 'statsmodels'. Сезонният анализ е деактивиран.")

# Конфигурация
YEARS_BACK = 3
LOCAL_ZONE = ZoneInfo("Europe/Sofia")
CACHE_TTL = 24 * 60 * 60  # 1 ден
ENTSOE_API_KEY = st.secrets.get("ENTSOE_API_KEY")
WEATHER_API_KEY = st.secrets.get("WEATHER_API_KEY")

# ==================== ФУНКЦИИ ====================
@st.cache_data(ttl=CACHE_TTL)
def fetch_ibex_data():
    """Изтегля исторически ден-напред цени директно от сайта на IBEX чрез scrape на HTML таблици."""
    records = []
    end = pd.Timestamp.utcnow().floor('D') - pd.Timedelta(days=1)
    start = end - pd.Timedelta(days=YEARS_BACK*365)
    for single_date in pd.date_range(start, end, freq='D'):
        date_str = single_date.strftime('%Y-%m-%d')
        url = f"https://www.ibex.bg/bg/dna/index.php?mn=140&dd={date_str}"
        try:
            tables = pd.read_html(url)
        except Exception as e:
            st.warning(f"Неуспех при зареждане на IBEX страница за {date_str}: {e}")
            continue
        # Предполага, че таблицата със стойности е втората таблица
        if len(tables) < 2:
            st.warning(f"Няма таблица за {date_str}")
            continue
        df_day = tables[1]
        # Очаква колони: 'Час' и 'Цена'
        if 'Час' not in df_day.columns or 'Цена' not in df_day.columns:
            st.warning(f"Неочаквани колони на IBEX за {date_str}")
            continue
        for _, row in df_day.iterrows():
            try:
                hour = int(row['Час'])
                price = float(str(row['Цена']).replace(',', '.'))
            except:
                continue
            records.append({'date': single_date.date(), 'hour': hour, 'price': price})
    if not records:
        st.error("Не са намерени данни на IBEX за посочения период.")
        return pd.DataFrame()
    df = pd.DataFrame(records)
    df['datetime'] = pd.to_datetime(df['date']) + pd.to_timedelta(df['hour'], unit='h')
    df['datetime'] = df['datetime'].dt.tz_localize(LOCAL_ZONE)
    pivot = df.pivot(index='date', columns='hour', values='price')
    pivot.columns = [f'hour_{h}' for h in pivot.columns]
    return pivot.reset_index()
    except Exception:
        pass
    # Fallback: direct CSV download
    st.info("Използва се CSV fallback за IBEX цени")
    end = pd.Timestamp.utcnow().floor('D') - pd.Timedelta(days=1)
    start = end - pd.Timedelta(days=YEARS_BACK*365)
    url = "https://transparency.entsoe.eu/api"
    params = {
        'documentType': 'A44',
        'in_Domain': '10YBG-ESO------Y',
        'out_Domain': '10YBG-ESO------Y',
        'periodStart': start.strftime('%Y%m%d') + '0000',
        'periodEnd': end.strftime('%Y%m%d') + '2300',
        'securityToken': ENTSOE_API_KEY,
        'download': 'true'
    }
    r = requests.get(url, params=params)
    if r.status_code != 200:
        st.error(f"CSV fallback error: {r.status_code}")
        return pd.DataFrame()
    from io import StringIO
    try:
        df_csv = pd.read_csv(StringIO(r.text), sep=';')
        df_csv['datetime'] = pd.to_datetime(df_csv['DateTime'], utc=True).dt.tz_convert(LOCAL_ZONE)
        df_csv['date'] = df_csv['datetime'].dt.date
        df_csv['hour'] = df_csv['datetime'].dt.hour
        pivot = df_csv.pivot_table(index='date', columns='hour', values='Price')
        pivot.columns = [f'hour_{h}' for h in pivot.columns]
        return pivot.reset_index()
    except Exception as e:
        st.error(f"CSV parsing error: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=CACHE_TTL)
def fetch_weather_history(lat, lon):
    records = []
    end = datetime.now(LOCAL_ZONE).date() - timedelta(days=1)
    start = end - timedelta(days=YEARS_BACK*365)
    for single_date in pd.date_range(start, end, freq='D'):
        dt_local = pd.Timestamp(single_date, tz=LOCAL_ZONE).replace(hour=12)
        ts = int(dt_local.astimezone(timezone.utc).timestamp())
        url = "https://api.openweathermap.org/data/2.5/onecall/timemachine"
        params = {'lat': lat, 'lon': lon, 'dt': ts,
                  'appid': WEATHER_API_KEY, 'units': 'metric'}
        j = requests.get(url, params=params).json()
        temps = [h['temp'] for h in j.get('hourly', [])]
        winds = [h['wind_speed'] for h in j.get('hourly', [])]
        uvis = [h.get('uvi', 0) for h in j.get('hourly', [])]
        if temps:
            records.append({'date': single_date.date(),
                            'avg_temp': np.mean(temps),
                            'avg_wind_speed': np.mean(winds),
                            'solar_radiation': np.mean(uvis)})
    return pd.DataFrame(records)

@st.cache_data(ttl=CACHE_TTL)
def fetch_weather_forecast(lat, lon):
    url = "https://api.openweathermap.org/data/2.5/onecall"
    params = {'lat': lat, 'lon': lon, 'exclude': 'current,minutely,hourly,alerts',
              'appid': WEATHER_API_KEY, 'units': 'metric'}
    data = requests.get(url, params=params).json().get('daily', [])
    rec = []
    for d in data:
        dt_local = datetime.fromtimestamp(d['dt'], tz=timezone.utc).astimezone(LOCAL_ZONE)
        rec.append({'date': dt_local.date(),
                    'avg_temp': d['temp']['day'],
                    'avg_wind_speed': d['wind_speed'],
                    'solar_radiation': d.get('uvi', 0)})
    return pd.DataFrame(rec)

# Предобработка

def create_features(df):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['dayofyear'] = df['date'].dt.dayofyear
    df['dayofweek'] = df['date'].dt.dayofweek
    df['is_weekend'] = df['dayofweek'] >= 5
    for lag in [1, 7, 365]:
        for h in range(24):
            df[f'hour_{h}_lag{lag}'] = df[f'hour_{h}'].shift(lag)
    for col in ['avg_temp', 'avg_wind_speed', 'solar_radiation']:
        df[f'{col}_lag1'] = df[col].shift(1)
    return df.dropna().reset_index(drop=True)

# Подготовка за многодневна прогноза

def prepare_dataset(df, horizon_days=7):
    df_feat = create_features(df)
    y_cols = [f'hour_{h}_lead{d}' for d in range(1, horizon_days+1) for h in range(24)]
    for d in range(1, horizon_days+1):
        shift = df_feat[[f'hour_{h}' for h in range(24)]].shift(-24*(d-1))
        shift.columns = [f'hour_{h}_lead{d}' for h in range(24)]
        df_feat = pd.concat([df_feat, shift], axis=1)
    df_feat = df_feat.dropna().reset_index(drop=True)
    X = df_feat[[c for c in df_feat.columns if 'lead' not in c and c != 'date']]
    y = df_feat[y_cols]
    return X, y

# Обучение

def train_model(X, y):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    tscv = TimeSeriesSplit(n_splits=5)
    maes = []
    for tr, va in tscv.split(Xs):
        model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=4)
        model.fit(Xs[tr], y.iloc[tr])
        maes.append(mean_absolute_error(y.iloc[va], model.predict(Xs[va])))
    model_final = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=4)
    model_final.fit(Xs, y)
    joblib.dump(model_final, 'model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    return model_final, scaler, np.mean(maes)

# Прогноза за седмица

def forecast_week(df, model, scaler):
    X, _ = prepare_dataset(df)
    last = X.tail(1)
    preds = model.predict(scaler.transform(last))[0]
    base = df['date'].max()
    hours = [base + timedelta(days=i//24+1, hours=i%24) for i in range(len(preds))]
    prices = preds.tolist()
    return pd.DataFrame({'datetime': hours, 'pred_price': prices})

# Streamlit UI

def main():
    st.title("7-дневна почасова прогноза на IBEX Day-Ahead")
    # Зареждане
    df_price = fetch_ibex_data()
    if df_price.empty:
        return
    df_hist_weather = fetch_weather_history(42.7, 23.3)
    df_weather_fc = fetch_weather_forecast(42.7, 23.3)

    # Merge исторически
    df = df_price.merge(df_hist_weather, on='date', how='left')

    # Анализ корелация
    st.subheader("Корелация цена и метео")
    cols = [c for c in df.columns if c.startswith('hour_')] + ['avg_temp', 'avg_wind_speed', 'solar_radiation']
    st.write(df[cols].corr())
    if HAS_STATS:
        df['mean_price'] = df[[f'hour_{h}' for h in range(24)]].mean(axis=1)
        res = seasonal_decompose(df.set_index('date')['mean_price'], model='additive', period=365)
        st.line_chart(res.trend); st.line_chart(res.seasonal)

    # Обучение и прогноза
    df_full = pd.concat([df_price, df_weather_fc], ignore_index=True)
    X, y = prepare_dataset(df_full)
    model, scaler, mae = train_model(X, y)
    st.success(f"CV MAE: {mae:.3f}")
    df_pred = forecast_week(df_full, model, scaler)

    st.subheader("Прогнозирани цени (почасово) следващите 7 дни")
    df_pred['datetime'] = pd.to_datetime(df_pred['datetime'])
    st.line_chart(df_pred.set_index('datetime')['pred_price'])
    st.dataframe(df_pred)

if __name__ == '__main__':
    main()


