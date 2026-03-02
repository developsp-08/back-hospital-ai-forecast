import pandas as pd
import numpy as np
import os
import joblib
from prophet import Prophet
from prophet.serialize import model_to_json
from sklearn.ensemble import RandomForestRegressor

def get_mock_opd_data():
    dates = pd.date_range(start='2024-01-01', end='2026-02-28', freq='D')
    df = pd.DataFrame({'date': dates})
    df['volume'] = 1200 + np.random.normal(0, 100, size=len(df))
    return df

def get_mock_er_data():
    dates = pd.date_range(start='2025-01-01', end='2026-02-28', freq='h')
    df = pd.DataFrame({'datetime': dates})
    df['hour'] = df['datetime'].dt.hour
    df['is_weekend'] = df['datetime'].dt.dayofweek >= 5
    df['er_load'] = 40 + (df['hour'] == 18) * 50 + np.random.normal(0, 10, size=len(df))
    return df

def get_mock_dengue_data():
    dates = pd.date_range(start='2020-01-01', end='2026-02-28', freq='W')
    df = pd.DataFrame({'date': dates})
    df['rainfall_14d'] = np.random.uniform(0, 200, size=len(df))
    df['humidity'] = np.random.uniform(60, 95, size=len(df))
    df['dengue_cases'] = (df['rainfall_14d'] * 0.5) + (df['humidity'] * 0.2) + np.random.normal(0, 5, size=len(df))
    return df

if __name__ == "__main__":
    print("🚀 เริ่มกระบวนการ Train Models...")
    if not os.path.exists('models'):
        os.makedirs('models')
        
    print("▶️ 1/3 กำลังเทรน OPD...")
    model_opd = Prophet(yearly_seasonality=True, weekly_seasonality=True)
    model_opd.fit(get_mock_opd_data().rename(columns={'date': 'ds', 'volume': 'y'}))
    with open('models/opd_model.json', 'w') as f:
        f.write(model_to_json(model_opd))

    print("▶️ 2/3 กำลังเทรน ER...")
    df_er = get_mock_er_data()
    model_er = RandomForestRegressor(n_estimators=10, random_state=42)
    model_er.fit(df_er[['hour', 'is_weekend']], df_er['er_load'])
    joblib.dump(model_er, 'models/er_model.pkl')

    print("▶️ 3/3 กำลังเทรน Dengue...")
    df_dengue = get_mock_dengue_data()
    model_dengue = RandomForestRegressor(n_estimators=10, random_state=42)
    model_dengue.fit(df_dengue[['rainfall_14d', 'humidity']], df_dengue['dengue_cases'])
    joblib.dump(model_dengue, 'models/dengue_model.pkl')
    
    print("🎉 เทรนเสร็จสมบูรณ์! ไฟล์ถูกบันทึกไว้ที่โฟลเดอร์ /models")