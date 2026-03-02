from fastapi import APIRouter
from app.services import predictor

router = APIRouter()

@router.get("/forecast")
def get_opd_forecast(days: int = 7):
    result = predictor.predict_opd_daily(days)
    return {"status": "success", "data": result}