import random

# ในอนาคตเราจะเขียนโค้ดโหลดไฟล์จากโฟลเดอร์ /models ที่นี่
# แต่ตอนนี้เราส่งข้อมูลจำลองกลับไปก่อน เพื่อให้รัน API ผ่านและหน้าบ้านมีข้อมูล

def predict_opd_daily(days):
    return [{"date": f"Day {i+1}", "volume": random.randint(1200, 1600)} for i in range(days)]

def predict_er_hourly():
    return {"current_load": "80%", "peak_hour": "20:00", "trend": "increasing"}

def predict_icu_daily():
    return {"occupancy_rate": 88, "available_beds": 2}

def predict_dengue_risk():
    return {"risk_level": "High", "action": "Prepare for surge in 2 weeks"}