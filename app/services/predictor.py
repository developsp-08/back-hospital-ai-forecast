import pandas as pd
import numpy as np
import os
import io
import random
from datetime import datetime, timedelta

# สร้างโฟลเดอร์สำหรับเก็บไฟล์ Database (Data Directory)
DATA_DIR = os.path.join(os.getcwd(), "data")
if not os.path.exists(DATA_DIR): 
    os.makedirs(DATA_DIR)

# ===============================================
# ฟังก์ชันหลัก: พยากรณ์คนไข้ + คำนวณพยาบาล + จัดตารางอัตโนมัติ
# ===============================================
def generate_real_recommendations():
    history_path = os.path.join(DATA_DIR, "er_hourly_history.csv")
    roster_path = os.path.join(DATA_DIR, "clean_nurse_schedule_feb2026.csv")
    
    # 1. พยากรณ์จำนวนคนไข้ของ "พรุ่งนี้"
    morning_pts, night_pts = 45, 65 
    chart_data = []
    
    if os.path.exists(history_path):
        df_history = pd.read_csv(history_path)
        df_history['datetime'] = pd.to_datetime(df_history['datetime'])
        df_history['hour'] = df_history['datetime'].dt.hour
        df_history['weekday'] = df_history['datetime'].dt.weekday
        
        tomorrow_weekday = (datetime.today().weekday() + 1) % 7
        df_tomorrow = df_history[df_history['weekday'] == tomorrow_weekday]
        if df_tomorrow.empty: df_tomorrow = df_history
            
        morning_pts = int(df_tomorrow[(df_tomorrow['hour'] >= 8) & (df_tomorrow['hour'] < 16)]['patient_count'].mean() * 8)
        night_pts = int(df_tomorrow[(df_tomorrow['hour'] >= 16) & (df_tomorrow['hour'] < 24)]['patient_count'].mean() * 8)
        
        hourly_avg = df_history.groupby('hour')['patient_count'].mean().round().astype(int).reset_index()
        display_hours = [8, 10, 12, 14, 16, 18, 20]
        for hr in display_hours:
            val = hourly_avg[hourly_avg['hour'] == hr]['patient_count']
            load = val.values[0] if not val.empty else 0
            chart_data.append({"hour": f"{hr:02d}:00", "load": int(load)})

    # 2. ตรวจสอบจำนวนพยาบาลปัจจุบัน
    current_morning, current_night = 5, 6 
    nurses_pool = [] # ถังรายชื่อพยาบาลสำหรับจัดเวร
    
    if os.path.exists(roster_path):
        df_roster = pd.read_csv(roster_path)
        current_morning = len(df_roster[df_roster['shift_type'] == 'Day']) // 28
        current_night = len(df_roster[df_roster['shift_type'] == 'Night']) // 28
        if current_morning == 0: current_morning = 5
        if current_night == 0: current_night = 6
        
        # ดึงรายชื่อพยาบาลจริงจากไฟล์มาเตรียมจัดเวร
        for emp_id, group in df_roster.groupby('employee_id'):
            nurses_pool.append({"id": str(int(emp_id)), "name": str(group['name'].iloc[0]).replace("RN. ", "RN. ")})
    else:
        # รายชื่อจำลองกรณีที่ยังไม่อัปโหลดไฟล์ตารางเวร
        nurses_pool = [
            {"id": "1", "name": "RN. Orawan"}, {"id": "2", "name": "RN. Somchai"},
            {"id": "3", "name": "RN. Wipada"}, {"id": "4", "name": "RN. Nipa"},
            {"id": "5", "name": "RN. Kittipong"}, {"id": "6", "name": "RN. Somsri"},
            {"id": "7", "name": "RN. Malee"}, {"id": "8", "name": "RN. Suda"},
            {"id": "9", "name": "RN. Wichai"}, {"id": "10", "name": "RN. Mana"},
            {"id": "11", "name": "RN. Aree"}, {"id": "12", "name": "RN. Kanda"},
            {"id": "13", "name": "RN. Piti"}, {"id": "14", "name": "RN. Sunee"}
        ]

    # 3. คำนวณความต้องการพยาบาลด้วย Rule-based (Ratio 1:6)
    rec_morning = max(1, morning_pts // 6)
    rec_night = max(1, night_pts // 6)
    
    recommendations = [
        { "ward": "ER", "shift": "Morning (08-16)", "predictedPatients": morning_pts, "currentStaff": current_morning, "recommendedStaff": rec_morning, "status": "Shortage" if current_morning < rec_morning else "Optimal" },
        { "ward": "ER", "shift": "Afternoon (16-00)", "predictedPatients": night_pts, "currentStaff": current_night, "recommendedStaff": rec_night, "status": "Shortage" if current_night < rec_night else "Optimal" }
    ]

    # ===============================================
    # 4. 🚀 Auto-Scheduler: จัดตารางเวร Detailed Shift แบบอัตโนมัติ
    # ===============================================
    random.shuffle(nurses_pool) # สับเปลี่ยนคิวเพื่อกระจายงาน (ในอนาคตจะเปลี่ยนเป็นการเช็คชั่วโมงการทำงาน)
    detailed_schedule = []
    
    # จ่ายงานกะเช้า (Day Shift) ตามจำนวนที่ AI แนะนำ (rec_morning)
    for i in range(min(rec_morning, len(nurses_pool))):
        nurse = nurses_pool[i]
        detailed_schedule.append({
            "id": nurse["id"], "name": nurse["name"], "ward": "ER",
            "shiftType": "Day", "time": "08:00 - 16:00", "hours": 8
        })

    # จ่ายงานกะดึก (Night Shift) ให้พยาบาลคนถัดไป ตามจำนวนที่ AI แนะนำ (rec_night)
    start_night_idx = rec_morning
    for i in range(start_night_idx, min(start_night_idx + rec_night, len(nurses_pool))):
        nurse = nurses_pool[i]
        detailed_schedule.append({
            "id": nurse["id"], "name": nurse["name"], "ward": "ER",
            "shiftType": "Night", "time": "16:00 - 00:00", "hours": 8
        })
    
    return recommendations, chart_data, detailed_schedule

# ===============================================
# ระบบจัดการไฟล์อัปโหลด Roster 
# ===============================================
def process_raw_roster(file_content, filename):
    try:
        df_raw = pd.read_excel(io.BytesIO(file_content), sheet_name='Schedule', header=1)
        cleaned_columns = []
        for c in df_raw.columns:
            if isinstance(c, pd.Timestamp) or hasattr(c, 'day'): c_str = str(c.day)
            else:
                c_str = str(c).strip()
                if '.' in c_str: c_str = c_str.split('.')[0] if c_str.split('.')[0].isdigit() else c_str
            cleaned_columns.append(c_str)
        df_raw.columns = cleaned_columns
        
        df_clean = df_raw.rename(columns={'ID': 'employee_id', 'Name of staff': 'name', 'Level': 'level'})
        df_clean['employee_id'] = pd.to_numeric(df_clean['employee_id'], errors='coerce')
        df_clean = df_clean.dropna(subset=['employee_id'])

        date_cols = [col for col in df_clean.columns if col.isdigit() and 1 <= int(col) <= 31]
        df_melted = pd.melt(df_clean, id_vars=['employee_id', 'name', 'level'], value_vars=date_cols, var_name='day', value_name='shift_code')
        df_melted['shift_code'] = df_melted['shift_code'].astype(str).str.strip().str.upper()
        df_melted = df_melted[~df_melted['shift_code'].isin(['X', 'R', 'OFF', 'NAN', '', 'NONE', 'IN-CH', 'HP'])]

        def parse_shift_code(code):
            if 'D12' in code: return pd.Series(['Day', 7, 12, 'Scheduled'])
            elif 'N12' in code or 'SN12' in code: return pd.Series(['Night', 19, 12, 'Scheduled'])
            elif 'D10' in code: return pd.Series(['Day', 7, 10, 'Scheduled'])
            elif 'D8' in code: return pd.Series(['Day', 8, 8, 'Scheduled'])
            elif code == 'D': return pd.Series(['Day', 8, 8, 'Scheduled'])
            elif code == 'N': return pd.Series(['Night', 20, 8, 'Scheduled'])
            else: return pd.Series(['Unknown', 0, 0, code])
        df_melted[['shift_type', 'start_hour', 'duration_hours', 'status']] = df_melted['shift_code'].apply(parse_shift_code)

        df_melted['date'] = pd.to_datetime(dict(year=2026, month=2, day=df_melted['day'].astype(int)), errors='coerce')
        df_melted = df_melted.dropna(subset=['date'])
        df_melted['date'] = df_melted['date'].dt.strftime('%Y-%m-%d')
        df_melted['ward'] = 'ER'
        
        df_final = df_melted[['employee_id', 'name', 'level', 'ward', 'date', 'shift_type', 'start_hour', 'duration_hours', 'status']].sort_values(by=['date', 'employee_id'])
        save_path = os.path.join(DATA_DIR, "clean_nurse_schedule_feb2026.csv")
        df_final.to_csv(save_path, index=False, encoding='utf-8-sig')

        nurses_list = []
        for emp_id, group in df_final.groupby('employee_id'):
            nurses_list.append({
                "id": str(int(emp_id)), "name": str(group['name'].iloc[0]), "level": str(group['level'].iloc[0]), "ward": str(group['ward'].iloc[0]),
                "maxHours": 12, "dayHours": int(group[group['shift_type'] == 'Day']['duration_hours'].sum()), "nightHours": int(group[group['shift_type'] == 'Night']['duration_hours'].sum())
            })

        recs, chart_data, detailed_schedule = generate_real_recommendations()

        return { "nurses": nurses_list, "recommendations": recs, "chart_data": chart_data, "detailed_schedule": detailed_schedule }

    except Exception as e:
        print(f"❌ Error Processing Roster: {e}")
        return {"nurses": [], "recommendations": [], "chart_data": [], "detailed_schedule": []}

# ===============================================
# ระบบจัดการไฟล์อัปโหลด Patient Load
# ===============================================
def process_patient_load(file_content, filename):
    save_path = os.path.join(DATA_DIR, "er_hourly_history.csv")
    with open(save_path, "wb") as f: f.write(file_content)
    
    recs, chart_data, detailed_schedule = generate_real_recommendations()
    return { "nurses": [], "recommendations": recs, "chart_data": chart_data, "detailed_schedule": detailed_schedule }

# ===============================================
# ฟังก์ชันดึงข้อมูล Dashboard (เรียกตอนโหลดหน้าแรก)
# ===============================================
def predict_er_hourly():
    recs, chart_data, detailed_schedule = generate_real_recommendations()
    peak_hour = max(chart_data, key=lambda x: x['load'])['hour'] if chart_data else "20:00"
        
    return {
        "current_load": "85%", "peak_hour": peak_hour, "trend": "increasing",
        "chart_data": chart_data, "recommendations": recs, "detailed_schedule": detailed_schedule
    }

def predict_opd_daily(days): return [{"date": f"Day {i+1}", "volume": random.randint(1200, 1600)} for i in range(days)]
def predict_icu_daily(): return {"occupancy_rate": 88, "available_beds": 2}
def predict_dengue_risk(): return {"risk_level": "High", "action": "Prepare for surge in 2 weeks"}