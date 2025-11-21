import pandas as pd
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
import openmeteo_requests
import requests_cache
from retry_requests import retry

# --- 1. BAŞLANGIÇ AYARLARI VE MODEL YÜKLEME ---
app = FastAPI(title="Otopark Doluluk Tahmin API")

# Model ve Scaler'ı global değişken olarak tutalım
model = None
scaler = None

# Sabit Park Bilgileri (Senin kodundaki map yapısı)
# Gerçek hayatta bunları veritabanından çekmek daha iyidir.
PARK_MAP = {
    "P_A": {"encoded_id": 0, "capacity": 100},
    "P_B": {"encoded_id": 1, "capacity": 150},
    "P_C": {"encoded_id": 2, "capacity": 80},
    "P_D": {"encoded_id": 3, "capacity": 200}
}

# Portekiz Tatilleri (Senin listenden)
HOLIDAYS = [
    '2020/01/01', '2020/04/10', '2020/04/13', '2020/04/25', '2020/05/01',
    '2020/06/10', '2020/08/15', '2020/10/05', '2020/12/01', '2020/12/08', '2020/12/25',
    # 2025 için de eklemeler yapılmalı, şimdilik örnek tutuyoruz
]

@app.on_event("startup")
def load_files():
    global model, scaler
    try:
        model = joblib.load('retrained_occupancy_model.joblib')
        scaler = joblib.load('retrained_standard_scaler.joblib')
        print("✅ Model ve Scaler başarıyla yüklendi.")
    except Exception as e:
        print(f"❌ Dosyalar yüklenemedi: {e}")

# --- 2. HAVA DURUMU FONKSİYONU (Open-Meteo Kullanarak) ---
def get_weather_forecast(target_time: datetime, lat=38.7223, lon=-9.1393):
    """
    Belirtilen saat için hava durumu tahmini çeker.
    """
    # Open-Meteo Setup
    cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
    retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
    openmeteo = openmeteo_requests.Client(session = retry_session)

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ["temperature_2m", "precipitation", "wind_speed_10m", "surface_pressure"],
        "forecast_days": 3
    }
    
    try:
        responses = openmeteo.weather_api(url, params=params)
        response = responses[0]
        hourly = response.Hourly()
        
        # Gelen veriyi DataFrame'e çevir
        hourly_data = {
            "date": pd.to_datetime(hourly.Time(), unit="s", utc=True),
            "temperature": hourly.Variables(0).ValuesAsNumpy(),
            "precipitation": hourly.Variables(1).ValuesAsNumpy(),
            "wind_speed": hourly.Variables(2).ValuesAsNumpy(),
            "pressure": hourly.Variables(3).ValuesAsNumpy()
        }
        df_weather = pd.DataFrame(data=hourly_data)
        
        # Hedef saate en yakın tahmini bul
        # Not: Zaman dilimi uyumsuzluğunu önlemek için her ikisi de UTC olmalı veya strip edilmeli
        target_time_utc = pd.to_datetime(target_time).tz_localize('UTC') if target_time.tzinfo is None else target_time
        
        # En yakın saati bul
        closest_row = df_weather.iloc[(df_weather['date'] - target_time_utc).abs().argsort()[:1]]
        
        return {
            "temperature": float(closest_row['temperature'].values[0]),
            "precipitation": float(closest_row['precipitation'].values[0]),
            "wind_speed": float(closest_row['wind_speed'].values[0]),
            "pressure": float(closest_row['pressure'].values[0]) # hPa cinsinden
        }
        
    except Exception as e:
        print(f"Hava durumu hatası: {e}")
        # Hata olursa ortalama değerler dön (Fallback)
        return {"temperature": 15.0, "precipitation": 0.0, "wind_speed": 10.0, "pressure": 1013.0}

# --- 3. İSTEK ŞEMASI (FLUTTER'DAN GELECEK VERİ) ---
class PredictionRequest(BaseModel):
    park_id: str  # Örn: "P_A"
    prediction_time: datetime # Örn: "2025-11-22T14:30:00"

# --- 4. TAHMİN ENDPOINT'İ ---
@app.post("/predict")
async def predict_occupancy(request: PredictionRequest):
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model sunucuda yüklü değil.")

    # 1. Park Bilgilerini Al
    park_info = PARK_MAP.get(request.park_id)
    if not park_info:
        raise HTTPException(status_code=404, detail="Geçersiz Park ID")

    # 2. Tarih Özelliklerini Çıkar
    dt = request.prediction_time
    hour = dt.hour
    dayofweek = dt.weekday() # 0=Pazartesi, 6=Pazar
    is_weekend = 1 if dayofweek >= 5 else 0
    
    date_str = dt.strftime('%Y/%m/%d')
    is_holiday = 1 if date_str in HOLIDAYS else 0

    # 3. Hava Durumu Tahminini Çek
    weather = get_weather_forecast(dt)

    # 4. Model İçin DataFrame Hazırla (Sıralama EĞİTİM ile AYNI olmalı!)
    # Eğitimdeki sütun sırası: 'hour', 'dayofweek', 'is_weekend', 'is_holiday', 'park_id_encoded', 'max_capacity', 'temp', ...
    
    input_data = pd.DataFrame([{
        'hour': hour,
        'dayofweek': dayofweek,
        'is_weekend': is_weekend,
        'is_holiday': is_holiday,
        'park_id_encoded': park_info['encoded_id'],
        'max_capacity': park_info['capacity'],
        'temperature': weather['temperature'],
        'precipitation': weather['precipitation'],
        'wind_speed': weather['wind_speed'],
        'pressure': weather['pressure']
    }])

    # 5. Ölçeklendirme (Sadece belirli sütunlar)
    cols_to_scale = ['hour', 'max_capacity', 'temperature', 'precipitation', 'wind_speed', 'pressure']
    
    # Gelen veriyi scale et
    input_data[cols_to_scale] = scaler.transform(input_data[cols_to_scale])

    # FEATURE SIRALAMASINI GARANTİLE
    FEATURES = [
        'hour', 'dayofweek', 'is_weekend', 'is_holiday', 'park_id_encoded',
        'max_capacity', 'temperature', 'precipitation', 'wind_speed', 'pressure'
    ]
    input_data = input_data[FEATURES]

    # 6. Tahmin Yap
    predicted_ratio = model.predict(input_data)[0]
    
    # Mantıksız sonuçları kırp (0 ile 1 arası)
    predicted_ratio = max(0.0, min(1.0, predicted_ratio))
    
    predicted_cars = int(predicted_ratio * park_info['capacity'])

    # Renk Kodu Belirle
    status_color = "GREEN"
    if predicted_ratio > 0.5: status_color = "YELLOW"
    if predicted_ratio > 0.85: status_color = "RED"

    return {
        "park_id": request.park_id,
        "prediction_time": dt,
        "occupancy_ratio": round(predicted_ratio, 2),
        "estimated_cars": predicted_cars,
        "max_capacity": park_info['capacity'],
        "status": status_color,
        "weather_summary": weather
    }