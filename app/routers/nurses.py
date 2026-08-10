from fastapi import APIRouter
from app.services import predictor
from typing import Optional
from pydantic import BaseModel

router = APIRouter()

class NurseCreate(BaseModel):
    employee_id: str
    name: str
    level: str
    ward: str = "ER"

class NurseUpdate(BaseModel):
    name: Optional[str] = None
    level: Optional[str] = None
    ward: Optional[str] = None

class NurseStatus(BaseModel):
    is_active: bool

@router.get("")
def get_nurses(ward: Optional[str] = None, active_only: bool = False):
    data = predictor.list_nurses(ward, active_only)
    return {"status": "success", "data": data}

@router.post("")
def create_nurse(payload: NurseCreate):
    ok = predictor.add_nurse(payload.employee_id, payload.name, payload.level, payload.ward)
    if ok:
        return {"status": "success", "message": "Nurse added"}
    return {"status": "error", "message": "Failed to add nurse"}

@router.put("/{employee_id}")
def edit_nurse(employee_id: str, payload: NurseUpdate):
    ok = predictor.update_nurse(employee_id, payload.name, payload.level, payload.ward)
    if ok:
        return {"status": "success", "message": "Nurse updated"}
    return {"status": "error", "message": "Nurse not found or update failed"}

@router.patch("/{employee_id}/status")
def change_nurse_status(employee_id: str, payload: NurseStatus):
    ok = predictor.set_nurse_active(employee_id, payload.is_active)
    if ok:
        return {"status": "success", "message": "Status updated"}
    return {"status": "error", "message": "Nurse not found"}
