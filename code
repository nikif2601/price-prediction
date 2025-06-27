import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import joblib

# ==================== ДАННИ ====================

def load_weather_data(weather_csv):
    dfw = pd.read_csv(weather_csv, parse_dates=['date'])
    dfw = dfw.sort_values('date').reset_index(drop=True)
    return dfw


def load_ibex_data(price_csv, weather_csv=None):
    df = pd.read_csv(price_csv, parse_dates=['date'])
    df = df.sort_values('date').reset_index(drop=True)
    if weather_csv:
        dfw = load_weather_data(weather_csv)
        df = df.merge(dfw, on='date', how='left')
    return df

# ==================== ФИЧЪРИ ====================

def create_features(df):
    data = df.copy()
    data['dayofweek'] = data['date'].dt.dayofweek
    data['is_weekend'] = data['dayofweek'].isin([5,6]).astype(int)
    for lag in [1, 7]:
        lag_price = [f'hour_{h}_lag{lag}' for h in range(24)]
        data[lag_price] = data[[f'hour_{h}' for h in range(24)]].shift(lag)
    if 'avg_temp' in data.columns:
        for col in ['avg_temp', 'avg_wind_speed', 'solar_radiation']:
            data[f'{col}_lag1'] = data[col].shift(1)
    data = data.dropna().reset_index(drop=True)
    return data

# ==================== ПОДГОТОВКА ====================

def prepare_dataset(data, target_hour):
    price_cols = [f'hour_{h}' for h in range(24)]
    non_features = ['date'] + price_cols
    X = data.drop(columns=non_features)
    y = data[f'hour_{target_hour}']
    return X, y

# ==================== ОБУЧЕНИЕ ====================

def train_model(X, y):
    tscv = TimeSeriesSplit(n_splits=5)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    maes, rmses = [], []
    for train_idx, val_idx in tscv.split(X_scaled):
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=4)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        maes.append(mean_absolute_error(y_val, preds))
        rmses.append(mean_squared_error(y_val, preds, squared=False))
    return model, scaler, np.mean(maes), np.mean(rmses)

# ==================== ПРОГНОЗА ====================

def forecast_next_day(df_hist, model, scaler):
    df_feat = create_features(df_hist)
    last = df_feat.iloc[[-1]].copy()
    preds = []
    for hour in range(24):
        X_pred = last.drop(columns=[f'hour_{h}' for h in range(24)])
        X_scaled = scaler.transform(X_pred)
        p = model.predict(X_scaled)[0]
        preds.append(p)
        last[f'hour_{hour}_lag1'] = p
    return preds

# ==================== STREAMLIT UI ====================

def main():
    st.title("IBEX Ден Напред Прогноза на Цени с Метео Фактори")
    price_file = st.file_uploader("Качете CSV с исторически цени", type="csv")
    weather_file = st.file_uploader("(По избор) CSV с исторически метео данни", type="csv")

    if price_file:
        df = load_ibex_data(price_file, weather_file)
        st.subheader("Преглед на данните")
        st.write(df.head())

        df_feat = create_features(df)
        st.write(f"След премахване на NaN: {df_feat.shape[0]} реда и {df_feat.shape[1]} колони")

        hour = st.slider("Изберете час за прогнозиране", min_value=0, max_value=23, value=0)
        if st.button("Обучи и валидирай модел"):            
            X, y = prepare_dataset(df_feat, target_hour=hour)
            model, scaler, mae, rmse = train_model(X, y)
            joblib.dump(model, 'model_hour_{}.pkl'.format(hour))
            joblib.dump(scaler, 'scaler_hour_{}.pkl'.format(hour))
            st.success(f"Модел обучен. CV MAE: {mae:.3f}, RMSE: {rmse:.3f}")

        if st.button("Прогнозирай за следващия ден"):
            model = joblib.load(f'model_hour_{hour}.pkl')
            scaler = joblib.load(f'scaler_hour_{hour}.pkl')
            preds = forecast_next_day(df, model, scaler)
            df_pred = pd.DataFrame({'hour': list(range(24)), 'pred_price': preds})
            st.line_chart(df_pred.set_index('hour'))
            st.write(df_pred)

if __name__ == "__main__":
    main()

