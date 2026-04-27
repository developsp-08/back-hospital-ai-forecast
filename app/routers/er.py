from fastapi import APIRouter, UploadFile, File, Form
from app.services import predictor

router = APIRouter()

@router.get("/forecast")
def get_er_forecast():
    result = predictor.predict_er_hourly()
    return {"status": "success", "data": result}

@router.post("/upload-data")
async def upload_data(
    file: UploadFile = File(...), 
    upload_type: str = Form(...)
):
    content = await file.read()
    
    if upload_type == "roster":
        result = predictor.process_raw_roster(content, file.filename)
    else:
        result = predictor.process_patient_load(content, file.filename)
        
    return {
        "status": "success", 
        "message": f"{upload_type} processed successfully", 
        "recommendations": result.get("recommendations", []),
        "nurses": result.get("nurses", []),
        "chart_data": result.get("chart_data", []),
        "detailed_schedule": result.get("detailed_schedule", []) # ส่งตารางเวรที่จัดเสร็จแล้วกลับไป
    }