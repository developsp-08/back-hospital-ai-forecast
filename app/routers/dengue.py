from fastapi import APIRouter
from app.services import predictor

router = APIRouter()

@router.get("/forecast")
def get_dengue_forecast():
    result = predictor.predict_dengue_risk()
    return {"status": "success", "data": result}