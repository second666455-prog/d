from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# تحميل الموديل والميزات
model = joblib.load('/home/ubuntu/car_price_model.pkl')
features = joblib.load('/home/ubuntu/model_features.pkl')

app = FastAPI(title="Car Price Prediction API")

class CarInput(BaseModel):
    Brand: int
    Model_Year: int
    Mileage_KM: float
    Fuel_Type: int
    Transmission: int

@app.get("/")
def home():
    return {"message": "Welcome to Car Price Prediction API"}

@app.post("/predict")
def predict(car: CarInput):
    # تحويل المدخلات إلى DataFrame
    input_data = pd.DataFrame([car.dict()])
    
    # هندسة المميزات للمدخلات الجديدة
    current_year = 2026
    input_data['Car_Age'] = current_year - input_data['Model_Year']
    input_data['KM_per_Year'] = input_data['Mileage_KM'] / (input_data['Car_Age'] + 1)
    
    # التأكد من ترتيب الأعمدة كما في التدريب
    input_data = input_data[features]
    
    # التنبؤ
    prediction = model.predict(input_data)
    
    return {"predicted_price": float(prediction[0])}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
