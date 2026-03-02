from fastapi import APIRouter
from app.services import predictor

router = APIRouter()

@router.get("/forecast")
def get_icu_forecast():
    result = predictor.predict_icu_daily()
    return {"status": "success", "data": result}