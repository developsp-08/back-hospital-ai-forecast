from fastapi import APIRouter, UploadFile, File, Form, Body
from app.services import predictor
from typing import Dict, Optional
from pydantic import BaseModel

router = APIRouter()

class ActualCensusPayload(BaseModel):
    date: str
    ward: str = "ER"
    day_patients: Optional[int] = None
    night_patients: Optional[int] = None

@router.get("/forecast")
def get_er_forecast():
    result = predictor.predict_er_hourly()
    return {"status": "success", "data": result}

# ให้ frontend ดึงเกณฑ์เดียวกับที่ CP-SAT ใช้ ไปเช็คตอนลากมือ (single source of truth)
@router.get("/scheduling-config")
def get_scheduling_config():
    return {"status": "success", "data": predictor.SCHEDULING_CONFIG}

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
        "llm_explanation": result.get("llm_explanation", ""),
        "patient_forecast": result.get("patient_forecast", {}),
        "fte_info": result.get("fte_info", {})
    }

# API สำหรับกรอกจำนวนคนไข้จริง (manual key-in ย้อนหลังได้ เพื่อใช้แก้/เทรนพยากรณ์เดือนถัดไป)
@router.get("/actual-census")
def get_actual_census(month: str, ward: str = "ER"):
    data = predictor.get_actual_census(month, ward)
    return {"status": "success", "data": data}

@router.post("/actual-census")
def save_actual_census(payload: ActualCensusPayload):
    success = predictor.save_actual_census(
        payload.date, payload.ward, payload.day_patients, payload.night_patients
    )
    if success:
        return {"status": "success", "message": "Actual census saved"}
    return {"status": "error", "message": "Failed to save actual census"}