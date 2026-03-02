from fastapi import APIRouter
from app.services import predictor

router = APIRouter()

@router.get("/forecast")
def get_er_forecast():
    result = predictor.predict_er_hourly()
    return {"status": "success", "data": result}