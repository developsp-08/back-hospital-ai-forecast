from fastapi import APIRouter, UploadFile, File, Form, Body
from app.services import predictor
from typing import Dict

router = APIRouter()

@router.get("/forecast")
def get_er_forecast():
    result = predictor.predict_er_hourly()
    return {"status": "success", "data": result}

# API สำหรับบันทึกการจัดเวรลง Database Neon
@router.post("/save-schedule")
async def save_schedule(payload: Dict = Body(...)):
    assignments = payload.get("data", [])
    if not assignments:
        return {"status": "error", "message": "No assignments to save"}
        
    success = predictor.save_shift_assignments(assignments)
    
    if success:
        return {"status": "success", "message": "Schedule saved to Neon DB"}
    else:
        return {"status": "error", "message": "Failed to save to database"}

@router.post("/upload-data")
async def upload_data(
    file: UploadFile = File(...), 
    upload_type: str = Form(...)
):
    content = await file.read()
    
    if upload_type == "roster":
        result = predictor.process_raw_roster(content, file.filename)
    else:
        # แม้จะมีการอัปโหลด patient load เข้ามา แต่เราจะพยากรณ์จากตารางเวรแทน
        result = predictor.predict_er_hourly()
        
    return {
        "status": "success", 
        "message": f"{upload_type} processed successfully", 
        "recommendations": result.get("recommendations", []),
        "nurses": result.get("nurses", []),
        "chart_data": result.get("chart_data", []),
        "detailed_schedule": result.get("detailed_schedule", []),
        "llm_explanation": result.get("llm_explanation", "") 
    }