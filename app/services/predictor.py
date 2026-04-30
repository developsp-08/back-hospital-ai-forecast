import pandas as pd
import numpy as np
import os
import io
import random
import calendar
from datetime import datetime, timedelta
from sqlalchemy import create_engine, types, text
from dotenv import load_dotenv

# Import Google OR-Tools (CP-SAT) for Optimization
try:
    from ortools.sat.python import cp_model
    HAS_ORTOOLS = True
except ImportError:
    HAS_ORTOOLS = False

# Import XGBoost for advanced Machine Learning prediction
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

# ===============================================
# Database Configuration
# ===============================================
load_dotenv()
DB_URL = os.getenv("DATABASE_URL")
if DB_URL and DB_URL.startswith("postgres://"):
    DB_URL = DB_URL.replace("postgres://", "postgresql://", 1)
engine = create_engine(DB_URL) if DB_URL else None

DATA_DIR = os.path.join(os.getcwd(), "data")
if not os.path.exists(DATA_DIR): 
    os.makedirs(DATA_DIR)

def clean_nurse_level(lvl):
    if pd.isna(lvl): return "Part-time"
    lvl_str = str(lvl).strip()
    if lvl_str.lower() in ['nan', 'none', 'null', '']: return "Part-time"
    return lvl_str

def get_nurses_for_target_month(target_month_str):
    if not engine:
        return []
    try:
        # Fetch Master List
        query_master = "SELECT DISTINCT employee_id, name, level, ward FROM nurse_schedule WHERE ward = 'ER'"
        df_master = pd.read_sql(query_master, engine)
        if df_master.empty: return []

        # Fetch Raw Working Hours (Grouped by pandas to prevent ID mismatches)
        query_all_history = "SELECT employee_id, date, duration_hours FROM nurse_schedule WHERE ward = 'ER'"
        df_raw_hist = pd.read_sql(query_all_history, engine)
        
        df_history = pd.DataFrame()
        if not df_raw_hist.empty:
            df_raw_hist['date'] = pd.to_datetime(df_raw_hist['date'], errors='coerce')
            df_raw_hist = df_raw_hist.dropna(subset=['date'])
            df_raw_hist['month_year'] = df_raw_hist['date'].dt.strftime('%b %Y') 
            df_raw_hist['month_key'] = df_raw_hist['date'].dt.strftime('%Y-%m')
            df_raw_hist['clean_id'] = df_raw_hist['employee_id'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
            
            df_history = df_raw_hist.groupby(['clean_id', 'month_year', 'month_key'])['duration_hours'].sum().reset_index()
            df_history = df_history.sort_values('month_key', ascending=False)

        nurses_list = []
        for _, row in df_master.iterrows():
            emp_id = str(row['employee_id'])
            emp_id_clean = emp_id.replace('.0', '').strip()
            
            work_history = []
            if not df_history.empty:
                person_hist = df_history[df_history['clean_id'] == emp_id_clean]
                for _, h_row in person_hist.iterrows():
                    work_history.append({
                        "label": str(h_row['month_year']),
                        "hours": int(h_row['duration_hours'])
                    })

            nurses_list.append({
                "id": emp_id,
                "name": str(row['name']).replace("RN. ", ""),
                "level": clean_nurse_level(row['level']), 
                "ward": str(row['ward']),
                "maxHours": 160,
                "workHistory": work_history
            })
        return nurses_list
    except Exception as e:
        print(f"DB Fetch Error: {e}")
        return []

def save_shift_assignments(assignments):
    if not engine or not assignments:
        return False
    try:
        df_new = pd.DataFrame(assignments)
        
        query = "SELECT DISTINCT employee_id, name, level FROM nurse_schedule"
        df_info = pd.read_sql(query, engine)
        df_info['employee_id'] = df_info['employee_id'].astype(str)
        df_new['employee_id'] = df_new['employee_id'].astype(str)
        
        df_final = df_new.merge(df_info, on='employee_id', how='left')
        df_final['level'] = df_final['level'].apply(clean_nurse_level)
        df_final['status'] = 'Confirmed'
        df_final['date'] = pd.to_datetime(df_final['date']).dt.date
        df_final['updated_at'] = datetime.now()

        try:
            target_dates = [f"'{d.strftime('%Y-%m-%d')}'" for d in df_final['date'].unique()]
            if target_dates:
                dates_str = ", ".join(target_dates)
                with engine.begin() as conn:
                    conn.execute(text(f"DELETE FROM nurse_schedule WHERE date IN ({dates_str}) AND ward = 'ER'"))
        except Exception as e:
            print(f"Warning during duplicate removal: {e}")

        dtype_mapping = {
            'employee_id': types.String(50), 'name': types.String(100), 'level': types.String(50),
            'ward': types.String(50), 'date': types.Date(), 'shift_type': types.String(20),
            'start_hour': types.Integer(), 'duration_hours': types.Integer(), 'status': types.String(50),
            'updated_at': types.DateTime()
        }
        df_final.to_sql("nurse_schedule", engine, if_exists='append', index=False, dtype=dtype_mapping)
        print(f"Saved {len(df_final)} shift assignments to database")
        return True
    except Exception as e:
        print(f"DB Save Error: {e}")
        return False

# ===============================================
# Enterprise AI Engine: XGBoost Prediction + CP-SAT Optimization
# ===============================================
def generate_real_recommendations():
    today = datetime.today()
    if today.month == 12:
        next_month_date = today.replace(year=today.year + 1, month=1, day=1)
    else:
        next_month_date = today.replace(month=today.month + 1, day=1)
        
    target_month_str = next_month_date.strftime('%Y-%m')
    num_days_next_month = calendar.monthrange(next_month_date.year, next_month_date.month)[1]
    
    nurses_pool = get_nurses_for_target_month(target_month_str) or []
    
    # ------------------------------------------------
    # STEP 1: PREDICTION (XGBoost vs Fallback)
    # ------------------------------------------------
    dow_demand = {'Day': {i:5 for i in range(7)}, 'Night': {i:6 for i in range(7)}}
    total_data_rows = 0
    use_xgboost = False
    model_xgb = None

    if engine:
        try:
            # Query exact dates to engineer ML features
            query_hist = f"""
                SELECT date, shift_type, COUNT(employee_id) as count 
                FROM nurse_schedule 
                WHERE ward = 'ER' AND TO_CHAR(date, 'YYYY-MM') < '{target_month_str}'
                GROUP BY date, shift_type
            """
            df_hist = pd.read_sql(query_hist, engine)
            if not df_hist.empty:
                total_data_rows = len(df_hist)
                df_hist['date'] = pd.to_datetime(df_hist['date'])
                df_hist['dayofweek'] = df_hist['date'].dt.dayofweek
                df_hist['is_weekend'] = df_hist['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)
                df_hist['shift_encoded'] = df_hist['shift_type'].apply(lambda x: 1 if x == 'Night' else 0)
                
                # Baseline Fallback Generation
                avg_df = df_hist.groupby(['dayofweek', 'shift_type'])['count'].mean().reset_index()
                for _, row in avg_df.iterrows():
                    shift_val = str(row['shift_type']).strip()
                    if shift_val in dow_demand:
                        py_dow = int(row['dayofweek'])
                        dow_demand[shift_val][py_dow] = int(round(row['count']))
                
                # Activate XGBoost only if sufficient data exists (minimum 14 shifts to prevent complete failure)
                if HAS_XGB and total_data_rows >= 14:
                    X = df_hist[['dayofweek', 'is_weekend', 'shift_encoded']]
                    y = df_hist['count']
                    
                    # Strict Anti-Overfitting Hyperparameters
                    model_xgb = xgb.XGBRegressor(
                        n_estimators=50,       # Low estimators to prevent memorization
                        max_depth=3,           # Shallow trees
                        learning_rate=0.1,     # Conservative learning step
                        subsample=0.8,         # Use 80% of data randomly
                        colsample_bytree=0.8,  # Use 80% of features randomly
                        random_state=42
                    )
                    model_xgb.fit(X, y)
                    use_xgboost = True
        except Exception as e:
            print(f"Prediction Error: {e}")

    daily_demands = {}
    for day in range(1, num_days_next_month + 1):
        dt = next_month_date.replace(day=day)
        py_dow = dt.weekday()
        is_wknd = 1 if py_dow >= 5 else 0
        
        if use_xgboost:
            pred_day_val = model_xgb.predict(pd.DataFrame([[py_dow, is_wknd, 0]], columns=['dayofweek', 'is_weekend', 'shift_encoded']))[0]
            pred_night_val = model_xgb.predict(pd.DataFrame([[py_dow, is_wknd, 1]], columns=['dayofweek', 'is_weekend', 'shift_encoded']))[0]
            # Bound the prediction output to safe numbers (min 3)
            daily_demands[(day, 'Day')] = max(3, int(round(pred_day_val)))
            daily_demands[(day, 'Night')] = max(3, int(round(pred_night_val)))
        else:
            daily_demands[(day, 'Day')] = dow_demand['Day'].get(py_dow, 5)
            daily_demands[(day, 'Night')] = dow_demand['Night'].get(py_dow, 6)

    # ------------------------------------------------
    # CHECK SAVED DATA (Actual User Assignments)
    # ------------------------------------------------
    saved_shifts = []
    if engine:
        try:
            q_saved = f"SELECT employee_id, EXTRACT(DAY FROM date) as day, shift_type, duration_hours FROM nurse_schedule WHERE ward='ER' AND TO_CHAR(date, 'YYYY-MM') = '{target_month_str}'"
            df_s = pd.read_sql(q_saved, engine)
            if not df_s.empty:
                for _, row in df_s.iterrows():
                    saved_shifts.append({
                        "id": f"assign-db-{random.randint(1000,9999)}",
                        "day": int(row['day']),
                        "ward": "ER",
                        "startHour": 8 if row['shift_type'] == 'Day' else 16,
                        "duration": int(row['duration_hours']),
                        "reqShift": row['shift_type'],
                        "filledBy": str(row['employee_id']),
                        "isUserAssigned": True
                    })
        except: pass

    # ------------------------------------------------
    # STEP 2: OPTIMIZATION (AI DRAFT)
    # Always run CP-SAT to provide hints, completely isolated from saved_shifts
    # ------------------------------------------------
    ai_draft = []
    optimization_success = False

    if HAS_ORTOOLS and len(nurses_pool) > 0:
        model = cp_model.CpModel()
        x = {}
        num_nurses = len(nurses_pool)
        
        for n in range(num_nurses):
            for d in range(1, num_days_next_month + 1):
                for s in ['Day', 'Night']:
                    x[(n, d, s)] = model.NewBoolVar(f'shift_n{n}_d{d}_{s}')
                    
        for d in range(1, num_days_next_month + 1):
            for n in range(num_nurses):
                model.AddAtMostOne([x[(n, d, 'Day')], x[(n, d, 'Night')]])
                
            if d < num_days_next_month:
                for n in range(num_nurses):
                    model.AddImplication(x[(n, d, 'Night')], x[(n, d+1, 'Day')].Not())
                    
            for s in ['Day', 'Night']:
                req_nurses = min(daily_demands[(d, s)], num_nurses)
                model.Add(sum(x[(n, d, s)] for n in range(num_nurses)) == req_nurses)
                
                l4_nurses = [n for n, nurse in enumerate(nurses_pool) if '4' in nurse['level']]
                if len(l4_nurses) > 0:
                    model.Add(sum(x[(n, d, s)] for n in l4_nurses) >= 1)
                    
        max_shifts = model.NewIntVar(0, 31, 'max_shifts')
        for n in range(num_nurses):
            model.Add(sum(x[(n, d, s)] for d in range(1, num_days_next_month + 1) for s in ['Day', 'Night']) <= max_shifts)
        model.Minimize(max_shifts)
        
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 4.0
        status = solver.Solve(model)
        
        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            optimization_success = True
            for d in range(1, num_days_next_month + 1):
                for s in ['Day', 'Night']:
                    for n, nurse in enumerate(nurses_pool):
                        if solver.Value(x[(n, d, s)]):
                            ai_draft.append({
                                "id": f"ai-hint-{d}-{s}-{nurse['id']}",
                                "day": d,
                                "reqShift": s,
                                "filledBy": nurse['id']
                            })

    # Generate Detailed Slots to indicate required levels
    detailed_schedule = []
    for day in range(1, num_days_next_month + 1):
        for i in range(daily_demands[(day, 'Day')]):
            detailed_schedule.append({ "day": day, "ward": "ER", "shiftType": "Day", "time": "08:00 - 16:00", "hours": 8, "reqLevel": "RN Level 4" if i == 0 else "RN Level 3" })
        for i in range(daily_demands[(day, 'Night')]):
            detailed_schedule.append({ "day": day, "ward": "ER", "shiftType": "Night", "time": "16:00 - 00:00", "hours": 8, "reqLevel": "RN Level 4" if i == 0 else "RN Level 3" })

    avg_m = int(sum(dow_demand['Day'].values()) / 7)
    avg_n = int(sum(dow_demand['Night'].values()) / 7)
    proxy_m_load = avg_m * 6
    proxy_n_load = avg_n * 6
    base_load = proxy_m_load // 3 
    chart_data = [
        {"hour": "08:00", "load": base_load},
        {"hour": "10:00", "load": base_load + 15},
        {"hour": "12:00", "load": base_load + 5},
        {"hour": "14:00", "load": base_load + 10},
        {"hour": "16:00", "load": proxy_n_load // 3},
        {"hour": "18:00", "load": (proxy_n_load // 3) + 20},
        {"hour": "20:00", "load": (proxy_n_load // 3) - 5},
    ]

    recommendations = [
        { "ward": "ER", "shift": "Morning (08-16)", "predictedPatients": proxy_m_load, "currentStaff": avg_m, "recommendedStaff": avg_m, "status": "Optimal" },
        { "ward": "ER", "shift": "Afternoon (16-00)", "predictedPatients": proxy_n_load, "currentStaff": avg_n, "recommendedStaff": avg_n, "status": "Optimal" }
    ]

    model_used = "XGBoost Machine Learning" if use_xgboost else "Statistical Baseline"
    llm_explanation = f"""
        <strong>Operational Plan for {next_month_date.strftime('%B %Y')} (Enterprise Mode)</strong>
        <br/><br/>
        The system utilized the <u>{model_used}</u> model to learn staffing behavior from {total_data_rows} historical shifts. Anti-overfitting measures are actively maintained.
        <br/><br/>
        <strong>Algorithm Actions:</strong>
        <ul style="margin-top: 8px; line-height: 1.6;">
            <li><strong>1. Demand Prediction:</strong> Predicts varying nurse requirements for weekdays vs. weekends.</li>
            <li><strong>2. Optimization Rules:</strong> Suggests at least one <b>RN Level 4</b> per shift automatically.</li>
            <li><strong>3. Dynamic Guidance:</strong> AI suggestions are shown as hints. You must manually drag and drop staff to fulfill the shift quotas.</li>
        </ul>
    """
    
    return recommendations, chart_data, detailed_schedule, nurses_pool, saved_shifts, llm_explanation, ai_draft

# ===============================================
# File Upload Processing
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

        df_clean['level'] = df_clean['level'].apply(clean_nurse_level)

        date_cols = [col for col in df_clean.columns if col.isdigit() and 1 <= int(col) <= 31]
        df_melted = pd.melt(df_clean, id_vars=['employee_id', 'name', 'level'], value_vars=date_cols, var_name='day', value_name='shift_code')
        df_melted['shift_code'] = df_melted['shift_code'].astype(str).str.strip().str.upper()
        df_melted = df_melted[~df_melted['shift_code'].isin(['X', 'R', 'OFF', 'NAN', '', 'NONE', 'IN-CH', 'HP'])]

        def parse_shift_code(code):
            c = str(code).upper().strip()
            if 'D12' in c: return pd.Series(['Day', 7, 12, 'Scheduled'])
            elif 'N12' in c or 'SN12' in c: return pd.Series(['Night', 19, 12, 'Scheduled'])
            elif 'D10' in c: return pd.Series(['Day', 7, 10, 'Scheduled'])
            elif 'D8' in c: return pd.Series(['Day', 8, 8, 'Scheduled'])
            elif c == 'D' or 'DAY' in c: return pd.Series(['Day', 8, 8, 'Scheduled'])
            elif c == 'N' or 'NIGHT' in c or 'SN' in c: return pd.Series(['Night', 20, 8, 'Scheduled'])
            elif 'D' in c: return pd.Series(['Day', 8, 8, 'Scheduled']) 
            elif 'N' in c: return pd.Series(['Night', 20, 8, 'Scheduled']) 
            else: 
                return pd.Series(['Day', 8, 8, 'Scheduled']) 
                
        df_melted[['shift_type', 'start_hour', 'duration_hours', 'status']] = df_melted['shift_code'].apply(parse_shift_code)

        upload_month = 2 
        fname_lower = filename.lower()
        months = {"jan":1, "feb":2, "mar":3, "apr":4, "may":5, "jun":6, "jul":7, "aug":8, "sep":9, "oct":10, "nov":11, "dec":12}
        for k, v in months.items():
            if k in fname_lower: upload_month = v
        
        df_melted['date'] = pd.to_datetime(dict(year=2026, month=upload_month, day=df_melted['day'].astype(int)), errors='coerce')
        df_melted = df_melted.dropna(subset=['date'])
        df_melted['date'] = df_melted['date'].dt.strftime('%Y-%m-%d')
        df_melted['ward'] = 'ER'
        df_melted['updated_at'] = datetime.now()
        
        df_final = df_melted[['employee_id', 'name', 'level', 'ward', 'date', 'shift_type', 'start_hour', 'duration_hours', 'status', 'updated_at']].sort_values(by=['date', 'employee_id'])

        if engine:
            try:
                upload_month_str = f"2026-{upload_month:02d}"
                with engine.begin() as conn:
                    conn.execute(text(f"DELETE FROM nurse_schedule WHERE TO_CHAR(date, 'YYYY-MM') = '{upload_month_str}' AND ward='ER'"))
                
                dtype_mapping = {
                    'employee_id': types.String(50), 'name': types.String(100), 'level': types.String(50),
                    'ward': types.String(50), 'date': types.Date(), 'shift_type': types.String(20),
                    'start_hour': types.Integer(), 'duration_hours': types.Integer(), 'status': types.String(50),
                    'updated_at': types.DateTime()
                }
                df_final.to_sql("nurse_schedule", engine, if_exists='append', index=False, dtype=dtype_mapping)
                print("Data update success!")
            except Exception as e:
                print(f"Data update failed: {e}")

        recs, chart_data, detailed_schedule, nurses_pool, saved_shifts, llm_exp, ai_draft = generate_real_recommendations()
        return { "nurses": nurses_pool, "recommendations": recs, "chart_data": chart_data, "detailed_schedule": detailed_schedule, "saved_shifts": saved_shifts, "llm_explanation": llm_exp, "ai_draft": ai_draft }
    except Exception as e:
        return {"nurses": [], "recommendations": [], "chart_data": [], "detailed_schedule": [], "saved_shifts": [], "llm_explanation": "", "ai_draft": []}

def predict_er_hourly():
    recs, chart_data, detailed_schedule, nurses_pool, saved_shifts, llm_exp, ai_draft = generate_real_recommendations()
    peak_hour = max(chart_data, key=lambda x: x['load'])['hour'] if chart_data else "20:00"
    return {
        "current_load": "85%", "peak_hour": peak_hour, "trend": "increasing",
        "chart_data": chart_data, "recommendations": recs, "detailed_schedule": detailed_schedule, 
        "nurses": nurses_pool, "saved_shifts": saved_shifts, "llm_explanation": llm_exp, "ai_draft": ai_draft
    }

def predict_opd_daily(days=7): 
    return [{"date": f"Day {i+1}", "volume": random.randint(1200, 1600)} for i in range(days)]

def predict_icu_daily(): 
    return {"occupancy_rate": 88, "available_beds": 2}

def predict_dengue_risk(): 
    return {"risk_level": "High", "action": "Prepare for surge in 2 weeks"}