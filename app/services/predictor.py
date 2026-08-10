import pandas as pd
import numpy as np
import os
import io
import random
import calendar
from datetime import datetime, timedelta
from sqlalchemy import create_engine, types, text
from dotenv import load_dotenv

try:
    import holidays
    KH_HOLIDAYS = holidays.Cambodia()
except ImportError:
    KH_HOLIDAYS = None

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
# Scheduling Config
# Defaults below come from the nurse-scheduling reference doc. They are
# starting points only — flagged there as needing sign-off from a domain
# expert (HOD/nursing admin) before being trusted as real policy. Keep them
# here as a single adjustable block rather than scattered magic numbers.
# ===============================================
SCHEDULING_CONFIG = {
    "MAX_CONSEC_DAY": 4,           # S1: max consecutive Day shifts before a break is preferred
    "MAX_CONSEC_NIGHT": 2,         # S2: max consecutive Night shifts before a break is preferred
    "WEEKLY_HOUR_CAP": 48,         # H8: soft cap on hours per (non-overlapping) 7-day block
    "HOD_MAX_GAP_DAYS": 14,        # S9: HOD should appear at least once every N days
    "SHIFT_HOURS": 8,              # hours per AI-drafted shift, used only for the weekly-cap check
    "SOLVER_TIMEOUT_SECONDS": 8.0,
    # Objective weights — all "how much do we care" knobs. Relative magnitude matters
    # more than absolute value: coverage shortage should hurt far more than a fairness
    # wobble. Needs sign-off from a domain expert before treating as real policy.
    "SHORTAGE_PENALTY": 1000,       # H1: unmet total headcount for a shift
    "LEVEL_SHORTAGE_PENALTY": 200,  # H1 extension: unmet per-level (skill mix) target
    "OVERSTAFF_PENALTY": 1,         # mild nudge against scheduling more than demand
    "INCHARGE_PENALTY": 50,         # H3: no L3/L4/HOD present on a shift
    "CONSECUTIVE_PENALTY": 5,       # S1/S2: streak longer than allowed
    "HOD_GAP_PENALTY": 10,          # S9: HOD absent for too long
    "WEEKLY_HOUR_PENALTY": 20,      # H8: over the weekly hour cap
    "NIGHT_LEDGER_WEIGHT": 1,       # S4: spread cumulative (historical + this month) night shifts
    "FAIRNESS_WEIGHT": 1,           # baseline: spread total shift count this month
    # FTE formula inputs (Section 1 of the reference doc): FTE = ADC x NHPOUS x 7/40 x
    # NON_PRODUCTIVE_FACTOR x FT_RATIO. NHPOUS_ER=3.5 comes from the doc's "ER Department"
    # section; the others are its stated defaults. All flagged ⚠️ there as needing
    # confirmation from a domain expert — treat as a starting point, not verified policy.
    "NHPOUS_ER": 3.5,
    "NON_PRODUCTIVE_FACTOR": 0.925,
    "FT_RATIO": 0.8,
}

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

def ensure_actual_census_table():
    if not engine:
        return
    try:
        with engine.begin() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS er_actual_census (
                    date DATE NOT NULL,
                    ward VARCHAR(50) NOT NULL DEFAULT 'ER',
                    day_patients INTEGER,
                    night_patients INTEGER,
                    updated_at TIMESTAMP,
                    PRIMARY KEY (date, ward)
                )
            """))
    except Exception as e:
        print(f"Table Init Error (er_actual_census): {e}")

ensure_actual_census_table()

def ensure_nurses_table():
    """Master nurse roster, independent of nurse_schedule (which is shift history only).
    Needed so a nurse can exist in the system before ever being assigned a shift."""
    if not engine:
        return
    try:
        with engine.begin() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS nurses (
                    employee_id VARCHAR(50) PRIMARY KEY,
                    name VARCHAR(100) NOT NULL,
                    level VARCHAR(50) NOT NULL,
                    ward VARCHAR(50) NOT NULL DEFAULT 'ER',
                    is_active BOOLEAN NOT NULL DEFAULT TRUE,
                    created_at TIMESTAMP DEFAULT now(),
                    updated_at TIMESTAMP DEFAULT now()
                )
            """))
            # One-time backfill from existing shift history — safe to run every startup,
            # ON CONFLICT DO NOTHING makes it a no-op after the first successful run.
            conn.execute(text("""
                INSERT INTO nurses (employee_id, name, level, ward, is_active, created_at, updated_at)
                SELECT DISTINCT ON (employee_id) employee_id, name, level, ward, TRUE, now(), now()
                FROM nurse_schedule
                WHERE ward = 'ER'
                ORDER BY employee_id, date DESC
                ON CONFLICT (employee_id) DO NOTHING
            """))
    except Exception as e:
        print(f"Table Init Error (nurses): {e}")

ensure_nurses_table()

def ensure_patient_history_tables():
    """Tables for real ER patient-volume data:
    - er_patient_history: daily Day/Night patient counts, bootstrapped once from
      data/er_hourly_history.csv (2 years of real hourly counts, never wired in before).
    - er_patient_monthly_stats: the hospital's own monthly totals (from the ER Patient
      Statistics sheet) — coarser than daily, kept only as a cross-check/reference,
      not as model training data.
    """
    if not engine:
        return
    try:
        with engine.begin() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS er_patient_history (
                    date DATE PRIMARY KEY,
                    day_patients INTEGER,
                    night_patients INTEGER,
                    is_weekend INTEGER,
                    is_holiday INTEGER,
                    has_local_event INTEGER,
                    is_raining INTEGER
                )
            """))
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS er_patient_monthly_stats (
                    year INTEGER NOT NULL,
                    month INTEGER NOT NULL,
                    total_patients INTEGER,
                    PRIMARY KEY (year, month)
                )
            """))
    except Exception as e:
        print(f"Table Init Error (er_patient_history): {e}")

def bootstrap_patient_history_from_csv():
    """One-time load of data/er_hourly_history.csv into er_patient_history, bucketed into
    Day (07:00-19:00) / Night (19:00-07:00) per the hospital's own Day time/Night time
    definitions. No-op if the table already has data (idempotent across restarts)."""
    if not engine:
        return
    try:
        with engine.connect() as conn:
            existing = conn.execute(text("SELECT COUNT(*) FROM er_patient_history")).scalar()
        if existing and existing > 0:
            return

        csv_path = os.path.join(DATA_DIR, "er_hourly_history.csv")
        if not os.path.exists(csv_path):
            return

        df = pd.read_csv(csv_path)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['date'] = df['datetime'].dt.date
        df['is_day_bucket'] = df['datetime'].dt.hour.between(7, 18)

        daily = df.groupby('date').apply(
            lambda g: pd.Series({
                'day_patients': int(g.loc[g['is_day_bucket'], 'patient_count'].sum()),
                'night_patients': int(g.loc[~g['is_day_bucket'], 'patient_count'].sum()),
                'is_weekend': int(g['is_weekend'].max()),
                'is_holiday': int(g['is_holiday'].max()),
                'has_local_event': int(g['has_local_event'].max()),
                'is_raining': int(g['is_raining'].max()),
            })
        ).reset_index()

        dtype_mapping = {
            'date': types.Date(), 'day_patients': types.Integer(), 'night_patients': types.Integer(),
            'is_weekend': types.Integer(), 'is_holiday': types.Integer(),
            'has_local_event': types.Integer(), 'is_raining': types.Integer(),
        }
        daily.to_sql("er_patient_history", engine, if_exists='append', index=False, dtype=dtype_mapping)
        print(f"Bootstrapped {len(daily)} days of ER patient history from CSV")
    except Exception as e:
        print(f"Patient History Bootstrap Error: {e}")

ensure_patient_history_tables()
bootstrap_patient_history_from_csv()

def clean_nurse_level(lvl):
    if pd.isna(lvl): return "Part-time"
    lvl_str = str(lvl).strip()
    if lvl_str.lower() in ['nan', 'none', 'null', '']: return "Part-time"
    return lvl_str

def clean_employee_id(eid):
    """Normalize employee_id to a plain string (strip float '.0' artifacts from Excel/pandas)."""
    if pd.isna(eid): return ""
    s = str(eid).strip()
    if s.endswith('.0'):
        s = s[:-2]
    return s

# Fallback used only when there isn't enough historical data to compute a real ratio.
DEFAULT_LEVEL_RATIO = {'RN Level 3': 1.0}
MIN_LEVEL4_PER_SHIFT = 1

def get_level_ratios(target_month_str):
    """Historical proportion of each nurse level within each shift (Day/Night), ward=ER,
    using ALL history (Scheduled + Confirmed alike — both represent real staffing, and
    Confirmed rows are the ground truth manager-approved schedules that future months
    should learn from)."""
    ratios = {'Day': dict(DEFAULT_LEVEL_RATIO), 'Night': dict(DEFAULT_LEVEL_RATIO)}
    if not engine:
        return ratios
    try:
        query = text("""
            SELECT shift_type, level, COUNT(*) as cnt
            FROM nurse_schedule
            WHERE ward = 'ER' AND TO_CHAR(date, 'YYYY-MM') < :target_month
            GROUP BY shift_type, level
        """)
        df = pd.read_sql(query, engine, params={"target_month": target_month_str})
        if df.empty:
            return ratios
        df['level'] = df['level'].apply(clean_nurse_level)
        df = df.groupby(['shift_type', 'level'])['cnt'].sum().reset_index()
        for shift in ['Day', 'Night']:
            df_shift = df[df['shift_type'] == shift]
            total = df_shift['cnt'].sum()
            if total > 0:
                ratios[shift] = {row['level']: row['cnt'] / total for _, row in df_shift.iterrows()}
    except Exception as e:
        print(f"Level Ratio Fetch Error: {e}")
    return ratios

def allocate_levels(total, level_ratio, min_level4=MIN_LEVEL4_PER_SHIFT):
    """Split a total headcount target into per-level counts using historical ratios,
    via largest-remainder apportionment (so counts always sum to `total`), then enforce
    the 'at least one Level 4 (or higher-charge) nurse per shift' business rule."""
    if total <= 0 or not level_ratio:
        return {}

    raw = {lvl: total * ratio for lvl, ratio in level_ratio.items()}
    counts = {lvl: int(v) for lvl, v in raw.items()}
    remainder = total - sum(counts.values())
    fractional_order = sorted(raw.keys(), key=lambda lvl: raw[lvl] - counts[lvl], reverse=True)
    for lvl in fractional_order:
        if remainder <= 0:
            break
        counts[lvl] += 1
        remainder -= 1

    l4_key = 'RN Level 4'
    if total >= min_level4 and counts.get(l4_key, 0) < min_level4:
        counts[l4_key] = min_level4
        excess = sum(counts.values()) - total
        donors = [lvl for lvl in counts if lvl != l4_key and counts[lvl] > 0]
        while excess > 0 and donors:
            donors.sort(key=lambda lvl: counts[lvl], reverse=True)
            counts[donors[0]] -= 1
            excess -= 1
            if counts[donors[0]] == 0:
                donors.pop(0)

    return {lvl: cnt for lvl, cnt in counts.items() if cnt > 0}

# ===============================================
# Nurse Master Data (nurses table) — CRUD
# ===============================================
def list_nurses(ward=None, active_only=False):
    if not engine:
        return []
    try:
        query = "SELECT employee_id, name, level, ward, is_active FROM nurses WHERE 1=1"
        params = {}
        if ward:
            query += " AND ward = :ward"
            params["ward"] = ward
        if active_only:
            query += " AND is_active = TRUE"
        query += " ORDER BY name"
        df = pd.read_sql(text(query), engine, params=params)
        return df.to_dict(orient='records')
    except Exception as e:
        print(f"List Nurses Error: {e}")
        return []

def add_nurse(employee_id, name, level, ward='ER'):
    if not engine:
        return False
    try:
        employee_id = clean_employee_id(employee_id)
        if not employee_id or not name:
            return False
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO nurses (employee_id, name, level, ward, is_active, created_at, updated_at)
                VALUES (:eid, :name, :level, :ward, TRUE, now(), now())
                ON CONFLICT (employee_id) DO UPDATE SET
                    name = EXCLUDED.name, level = EXCLUDED.level, ward = EXCLUDED.ward,
                    is_active = TRUE, updated_at = now()
            """), {"eid": employee_id, "name": name, "level": clean_nurse_level(level), "ward": ward})
        return True
    except Exception as e:
        print(f"Add Nurse Error: {e}")
        return False

def update_nurse(employee_id, name=None, level=None, ward=None):
    if not engine:
        return False
    try:
        employee_id = clean_employee_id(employee_id)
        fields = {}
        if name: fields['name'] = name
        if level: fields['level'] = clean_nurse_level(level)
        if ward: fields['ward'] = ward
        if not fields:
            return False
        set_clause = ", ".join(f"{k} = :{k}" for k in fields)
        fields['eid'] = employee_id
        with engine.begin() as conn:
            result = conn.execute(text(f"UPDATE nurses SET {set_clause}, updated_at = now() WHERE employee_id = :eid"), fields)
        return result.rowcount > 0
    except Exception as e:
        print(f"Update Nurse Error: {e}")
        return False

def set_nurse_active(employee_id, is_active):
    if not engine:
        return False
    try:
        employee_id = clean_employee_id(employee_id)
        with engine.begin() as conn:
            result = conn.execute(text("UPDATE nurses SET is_active = :active, updated_at = now() WHERE employee_id = :eid"),
                                   {"active": is_active, "eid": employee_id})
        return result.rowcount > 0
    except Exception as e:
        print(f"Set Nurse Active Error: {e}")
        return False

def get_nurses_for_target_month(target_month_str):
    if not engine:
        return []
    try:
        # Fetch Master List from the `nurses` table — independent of shift history, so a
        # newly-added nurse with zero shifts still shows up in the scheduling pool.
        query_master = "SELECT employee_id, name, level, ward FROM nurses WHERE ward = 'ER' AND is_active = TRUE"
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
            df_raw_hist['clean_id'] = df_raw_hist['employee_id'].apply(clean_employee_id)

            df_history = df_raw_hist.groupby(['clean_id', 'month_year', 'month_key'])['duration_hours'].sum().reset_index()
            df_history = df_history.sort_values('month_key', ascending=False)

        nurses_list = []
        for _, row in df_master.iterrows():
            emp_id_clean = clean_employee_id(row['employee_id'])

            work_history = []
            if not df_history.empty:
                person_hist = df_history[df_history['clean_id'] == emp_id_clean]
                for _, h_row in person_hist.iterrows():
                    work_history.append({
                        "label": str(h_row['month_year']),
                        "hours": int(h_row['duration_hours'])
                    })

            nurses_list.append({
                "id": emp_id_clean,
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
        
        query = "SELECT employee_id, name, level FROM nurses"
        df_info = pd.read_sql(query, engine)
        df_info['employee_id'] = df_info['employee_id'].apply(clean_employee_id)
        df_new['employee_id'] = df_new['employee_id'].apply(clean_employee_id)
        
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
# Actual Patient Census (manual key-in, used to correct/train future forecasts)
# ===============================================
def save_actual_census(date_str, ward, day_patients, night_patients):
    if not engine:
        return False
    try:
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO er_actual_census (date, ward, day_patients, night_patients, updated_at)
                VALUES (:date, :ward, :day_patients, :night_patients, :updated_at)
                ON CONFLICT (date, ward) DO UPDATE SET
                    day_patients = EXCLUDED.day_patients,
                    night_patients = EXCLUDED.night_patients,
                    updated_at = EXCLUDED.updated_at
            """), {
                "date": date_str, "ward": ward,
                "day_patients": day_patients, "night_patients": night_patients,
                "updated_at": datetime.now()
            })
        return True
    except Exception as e:
        print(f"Actual Census Save Error: {e}")
        return False

def get_actual_census(target_month_str, ward='ER'):
    if not engine:
        return []
    try:
        query = text("""
            SELECT date, day_patients, night_patients
            FROM er_actual_census
            WHERE ward = :ward AND TO_CHAR(date, 'YYYY-MM') = :ym
            ORDER BY date
        """)
        df = pd.read_sql(query, engine, params={"ward": ward, "ym": target_month_str})
        if df.empty:
            return []
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
        return df.to_dict(orient='records')
    except Exception as e:
        print(f"Actual Census Fetch Error: {e}")
        return []

# ===============================================
# Patient Demand Forecast — real, independent of staffing history
# (replaces the old chart proxy that just multiplied staffing counts)
# ===============================================
def predict_patient_demand(target_month_str, next_month_date, num_days):
    """Predict Day/Night patient counts for each day of the target month.
    Trains on er_patient_history (bootstrapped from 2 years of real hourly data) plus
    any real er_actual_census entries logged so far — both are the same Day/Night
    patient-count shape, so they combine directly as one growing training set."""
    daily_patients = {}
    for day in range(1, num_days + 1):
        daily_patients[day] = {"day_patients": 12, "night_patients": 8}  # crude fallback

    use_xgboost = False
    total_rows = 0
    if not engine:
        return daily_patients, use_xgboost, total_rows

    try:
        df_hist = pd.read_sql(text("""
            SELECT date, day_patients, night_patients, is_weekend, is_holiday
            FROM er_patient_history WHERE TO_CHAR(date, 'YYYY-MM') < :target_month
        """), engine, params={"target_month": target_month_str})
        df_actual = pd.read_sql(text("""
            SELECT date, day_patients, night_patients
            FROM er_actual_census WHERE TO_CHAR(date, 'YYYY-MM') < :target_month
        """), engine, params={"target_month": target_month_str})

        if not df_actual.empty:
            df_actual['is_weekend'] = pd.to_datetime(df_actual['date']).dt.dayofweek.apply(lambda d: 1 if d >= 5 else 0)
            df_actual['is_holiday'] = 0
            df_hist = pd.concat([df_hist, df_actual], ignore_index=True)
        df_hist = df_hist.dropna(subset=['day_patients', 'night_patients'])

        if df_hist.empty:
            return daily_patients, use_xgboost, total_rows

        total_rows = len(df_hist)
        df_hist['date'] = pd.to_datetime(df_hist['date'])
        df_hist['dayofweek'] = df_hist['date'].dt.dayofweek

        dow_avg_day = df_hist.groupby('dayofweek')['day_patients'].mean()
        dow_avg_night = df_hist.groupby('dayofweek')['night_patients'].mean()

        model_day = model_night = None
        if HAS_XGB and total_rows >= 30:
            X = df_hist[['dayofweek', 'is_weekend', 'is_holiday']]
            xgb_params = dict(n_estimators=50, max_depth=3, learning_rate=0.1,
                               subsample=0.8, colsample_bytree=0.8, random_state=42)
            model_day = xgb.XGBRegressor(**xgb_params).fit(X, df_hist['day_patients'])
            model_night = xgb.XGBRegressor(**xgb_params).fit(X, df_hist['night_patients'])
            use_xgboost = True

        for day in range(1, num_days + 1):
            dt = next_month_date.replace(day=day)
            py_dow = dt.weekday()
            is_wknd = 1 if py_dow >= 5 else 0
            is_hol = 1 if (KH_HOLIDAYS and dt.date() in KH_HOLIDAYS) else 0
            if use_xgboost:
                X_pred = pd.DataFrame([[py_dow, is_wknd, is_hol]], columns=['dayofweek', 'is_weekend', 'is_holiday'])
                day_val = max(0, int(round(model_day.predict(X_pred)[0])))
                night_val = max(0, int(round(model_night.predict(X_pred)[0])))
            else:
                day_val = int(round(dow_avg_day.get(py_dow, dow_avg_day.mean())))
                night_val = int(round(dow_avg_night.get(py_dow, dow_avg_night.mean())))
            daily_patients[day] = {"day_patients": day_val, "night_patients": night_val}
    except Exception as e:
        print(f"Patient Demand Forecast Error: {e}")

    return daily_patients, use_xgboost, total_rows

# ===============================================
# Enterprise AI Engine: XGBoost Prediction + CP-SAT Optimization
# ===============================================
def generate_real_recommendations():
    cfg = SCHEDULING_CONFIG
    today = datetime.today()
    if today.month == 12:
        next_month_date = today.replace(year=today.year + 1, month=1, day=1)
    else:
        next_month_date = today.replace(month=today.month + 1, day=1)

    target_month_str = next_month_date.strftime('%Y-%m')
    num_days_next_month = calendar.monthrange(next_month_date.year, next_month_date.month)[1]

    nurses_pool = get_nurses_for_target_month(target_month_str) or []

    # ------------------------------------------------
    # PATIENT DEMAND FORECAST — real prediction (er_patient_history + er_actual_census),
    # independent of staffing history. Feeds the FTE formula and the dashboard's
    # "Predicted Patients" card; replaces the old chart proxy that was just
    # staffing_count * 6 with no independent patient signal at all.
    # ------------------------------------------------
    daily_patients, patient_use_xgb, patient_data_rows = predict_patient_demand(
        target_month_str, next_month_date, num_days_next_month
    )
    avg_day_patients = sum(v['day_patients'] for v in daily_patients.values()) / num_days_next_month
    avg_night_patients = sum(v['night_patients'] for v in daily_patients.values()) / num_days_next_month
    predicted_adc = avg_day_patients + avg_night_patients
    month_total_patients = sum(v['day_patients'] + v['night_patients'] for v in daily_patients.values())

    # ------------------------------------------------
    # STEP 1: PREDICTION (XGBoost vs Fallback)
    # Uses ALL history (Scheduled + Confirmed) since a manager-confirmed month is
    # real ground truth too, not test data — only shift_type IN ('Day','Night')
    # is counted so Leave rows (H10) never leak into the demand signal.
    # ------------------------------------------------
    dow_demand = {'Day': {i:5 for i in range(7)}, 'Night': {i:6 for i in range(7)}}
    total_data_rows = 0
    use_xgboost = False
    model_xgb = None

    if engine:
        try:
            # Query exact dates to engineer ML features
            query_hist = text("""
                SELECT date, shift_type, COUNT(employee_id) as count
                FROM nurse_schedule
                WHERE ward = 'ER' AND shift_type IN ('Day', 'Night')
                  AND TO_CHAR(date, 'YYYY-MM') < :target_month
                GROUP BY date, shift_type
            """)
            df_hist = pd.read_sql(query_hist, engine, params={"target_month": target_month_str})
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
    # PER-LEVEL DEMAND (Phase 1) — split each day's total headcount target into
    # per-level counts using historical level ratios (see get_level_ratios/allocate_levels)
    # ------------------------------------------------
    level_ratios = get_level_ratios(target_month_str)
    daily_level_demands = {}
    for day in range(1, num_days_next_month + 1):
        for s in ['Day', 'Night']:
            daily_level_demands[(day, s)] = allocate_levels(daily_demands[(day, s)], level_ratios[s])

    # ------------------------------------------------
    # CHECK SAVED DATA (Actual User Assignments) — Leave rows are absence
    # records, not shifts, so they're excluded via shift_type IN ('Day','Night')
    # ------------------------------------------------
    saved_shifts = []
    if engine:
        try:
            q_saved = text("""
                SELECT employee_id, EXTRACT(DAY FROM date) as day, shift_type, duration_hours
                FROM nurse_schedule
                WHERE ward='ER' AND shift_type IN ('Day', 'Night')
                  AND TO_CHAR(date, 'YYYY-MM') = :target_month
            """)
            df_s = pd.read_sql(q_saved, engine, params={"target_month": target_month_str})
            if not df_s.empty:
                for _, row in df_s.iterrows():
                    saved_shifts.append({
                        "id": f"assign-db-{random.randint(1000,9999)}",
                        "day": int(row['day']),
                        "ward": "ER",
                        "startHour": 8 if row['shift_type'] == 'Day' else 16,
                        "duration": int(row['duration_hours']),
                        "reqShift": row['shift_type'],
                        "filledBy": clean_employee_id(row['employee_id']),
                        "isUserAssigned": True
                    })
        except: pass

    # ------------------------------------------------
    # LEAVE (H10) — nurses marked on leave this month must not be assigned
    # ------------------------------------------------
    leave_set = set()
    if engine:
        try:
            q_leave = text("""
                SELECT employee_id, EXTRACT(DAY FROM date) as day
                FROM nurse_schedule
                WHERE ward='ER' AND status = 'Leave' AND TO_CHAR(date, 'YYYY-MM') = :target_month
            """)
            df_leave = pd.read_sql(q_leave, engine, params={"target_month": target_month_str})
            for _, row in df_leave.iterrows():
                leave_set.add((clean_employee_id(row['employee_id']), int(row['day'])))
        except Exception as e:
            print(f"Leave Fetch Error: {e}")

    # ------------------------------------------------
    # CROSS-MONTH FAIRNESS LEDGER (S4) — cumulative Night-shift count from prior
    # months, so this month's optimizer doesn't reset fairness to zero every run
    # ------------------------------------------------
    prior_night_counts = {}
    if engine:
        try:
            q_ledger = text("""
                SELECT employee_id, COUNT(*) as cnt
                FROM nurse_schedule
                WHERE ward='ER' AND shift_type = 'Night' AND TO_CHAR(date, 'YYYY-MM') < :target_month
                GROUP BY employee_id
            """)
            df_ledger = pd.read_sql(q_ledger, engine, params={"target_month": target_month_str})
            for _, row in df_ledger.iterrows():
                prior_night_counts[clean_employee_id(row['employee_id'])] = int(row['cnt'])
        except Exception as e:
            print(f"Ledger Fetch Error: {e}")

    # ------------------------------------------------
    # STEP 2: OPTIMIZATION (AI DRAFT)
    # Always run CP-SAT to provide hints, completely isolated from saved_shifts.
    # Coverage/skill-mix/in-charge/hours/consecutive/HOD-presence are modeled as
    # SOFT constraints (shortage/penalty variables) rather than hard equalities,
    # so a bad day (e.g. everyone senior is on leave) degrades gracefully with a
    # flagged shortage instead of making the whole month unsolvable.
    # ------------------------------------------------
    ai_draft = []
    optimization_success = False
    shortage_summary = {"coverage": 0, "level_mix": 0, "incharge_days": 0}

    if HAS_ORTOOLS and len(nurses_pool) > 0:
        model = cp_model.CpModel()
        x = {}
        num_nurses = len(nurses_pool)
        objective_terms = []

        for n in range(num_nurses):
            for d in range(1, num_days_next_month + 1):
                for s in ['Day', 'Night']:
                    x[(n, d, s)] = model.NewBoolVar(f'shift_n{n}_d{d}_{s}')

        # H10: leave — zero out any (nurse, day) marked on leave
        for n, nurse in enumerate(nurses_pool):
            for d in range(1, num_days_next_month + 1):
                if (nurse['id'], d) in leave_set:
                    model.Add(x[(n, d, 'Day')] == 0)
                    model.Add(x[(n, d, 'Night')] == 0)

        senior_levels = {'RN Level 3', 'RN Level 4', 'HOD'}
        senior_nurses = [n for n, nurse in enumerate(nurses_pool) if nurse['level'] in senior_levels]
        incharge_shortage_vars = []
        coverage_shortage_vars = []
        overstaff_vars = []
        level_shortage_vars = []

        for d in range(1, num_days_next_month + 1):
            # H6: no double shift same day
            for n in range(num_nurses):
                model.AddAtMostOne([x[(n, d, 'Day')], x[(n, d, 'Night')]])

            # H7: rest after night — no Night(d) -> Day(d+1)
            if d < num_days_next_month:
                for n in range(num_nurses):
                    model.AddImplication(x[(n, d, 'Night')], x[(n, d+1, 'Day')].Not())

            for s in ['Day', 'Night']:
                demand = daily_demands[(d, s)]
                assigned = sum(x[(n, d, s)] for n in range(num_nurses))

                # H1: total coverage — soft (shortage + light overstaff penalty)
                shortfall = model.NewIntVar(0, demand, f'short_{d}_{s}')
                overfill = model.NewIntVar(0, num_nurses, f'over_{d}_{s}')
                model.Add(assigned - demand == overfill - shortfall)
                coverage_shortage_vars.append(shortfall)
                overstaff_vars.append(overfill)

                # H1-extension: per-level (skill mix) targets — soft, capped by pool size
                for lvl, target_cnt in daily_level_demands[(d, s)].items():
                    lvl_pool = [n for n, nurse in enumerate(nurses_pool) if nurse['level'] == lvl]
                    if not lvl_pool:
                        continue
                    capped_target = min(target_cnt, len(lvl_pool))
                    lvl_shortfall = model.NewIntVar(0, capped_target, f'lvlshort_{d}_{s}_{lvl}')
                    model.Add(sum(x[(n, d, s)] for n in lvl_pool) + lvl_shortfall >= capped_target)
                    level_shortage_vars.append(lvl_shortfall)

                # H3: in-charge — at least one L3/L4/HOD present, soft
                if senior_nurses:
                    present = model.NewBoolVar(f'senior_{d}_{s}')
                    senior_sum = sum(x[(n, d, s)] for n in senior_nurses)
                    model.Add(senior_sum >= 1).OnlyEnforceIf(present)
                    model.Add(senior_sum == 0).OnlyEnforceIf(present.Not())
                    incharge_shortage_vars.append(present.Not())

        # H8: weekly hour cap — soft, non-overlapping 7-day blocks
        overtime_vars = []
        week_starts = list(range(1, num_days_next_month + 1, 7))
        for n in range(num_nurses):
            for wstart in week_starts:
                wend = min(wstart + 6, num_days_next_month)
                week_shifts = sum(x[(n, d, s)] for d in range(wstart, wend + 1) for s in ['Day', 'Night'])
                max_hours = (wend - wstart + 1) * 2 * cfg["SHIFT_HOURS"]
                week_hours = model.NewIntVar(0, max_hours, f'wh_{n}_{wstart}')
                model.Add(week_hours == week_shifts * cfg["SHIFT_HOURS"])
                overtime = model.NewIntVar(0, max_hours, f'ot_{n}_{wstart}')
                model.Add(overtime >= week_hours - cfg["WEEKLY_HOUR_CAP"])
                overtime_vars.append(overtime)

        # S1/S2: consecutive Day/Night streaks — soft
        streak_vars = []
        for shift_type, max_consec in [('Day', cfg["MAX_CONSEC_DAY"]), ('Night', cfg["MAX_CONSEC_NIGHT"])]:
            window = max_consec + 1
            if window > num_days_next_month:
                continue
            for n in range(num_nurses):
                for d in range(1, num_days_next_month - window + 2):
                    days_window = [x[(n, dd, shift_type)] for dd in range(d, d + window)]
                    viol = model.NewBoolVar(f'streak_{shift_type}_{n}_{d}')
                    model.AddBoolAnd(days_window).OnlyEnforceIf(viol)
                    model.AddBoolOr([v.Not() for v in days_window]).OnlyEnforceIf(viol.Not())
                    streak_vars.append(viol)

        # S9: HOD presence at least once every HOD_MAX_GAP_DAYS — soft
        hod_gap_vars = []
        hod_nurses = [n for n, nurse in enumerate(nurses_pool) if nurse['level'] == 'HOD']
        gap = cfg["HOD_MAX_GAP_DAYS"]
        if hod_nurses and gap <= num_days_next_month:
            for d in range(1, num_days_next_month - gap + 2):
                hod_present = model.NewBoolVar(f'hodpresent_{d}')
                window_assigns = [x[(n, dd, s)] for n in hod_nurses for dd in range(d, d + gap) for s in ['Day', 'Night']]
                model.Add(sum(window_assigns) >= 1).OnlyEnforceIf(hod_present)
                model.Add(sum(window_assigns) == 0).OnlyEnforceIf(hod_present.Not())
                hod_gap_vars.append(hod_present.Not())

        # S4: cross-month night fairness ledger — minimize the max cumulative
        # (prior months + this month) night count across nurses
        cum_night_vars = []
        for n, nurse in enumerate(nurses_pool):
            prior_n = prior_night_counts.get(nurse['id'], 0)
            this_month_nights = sum(x[(n, d, 'Night')] for d in range(1, num_days_next_month + 1))
            cum_night = model.NewIntVar(0, num_days_next_month + prior_n, f'cumnight_{n}')
            model.Add(cum_night == this_month_nights + prior_n)
            cum_night_vars.append(cum_night)
        max_cum_night = model.NewIntVar(0, num_days_next_month + max(prior_night_counts.values(), default=0), 'max_cum_night')
        for v in cum_night_vars:
            model.Add(v <= max_cum_night)

        # Baseline fairness: spread total shifts this month
        max_shifts = model.NewIntVar(0, num_days_next_month, 'max_shifts')
        for n in range(num_nurses):
            model.Add(sum(x[(n, d, s)] for d in range(1, num_days_next_month + 1) for s in ['Day', 'Night']) <= max_shifts)

        # Multi-term weighted objective
        objective_terms.append(cfg["SHORTAGE_PENALTY"] * sum(coverage_shortage_vars))
        objective_terms.append(cfg["OVERSTAFF_PENALTY"] * sum(overstaff_vars))
        objective_terms.append(cfg["LEVEL_SHORTAGE_PENALTY"] * sum(level_shortage_vars))
        if incharge_shortage_vars:
            objective_terms.append(cfg["INCHARGE_PENALTY"] * sum(incharge_shortage_vars))
        objective_terms.append(cfg["WEEKLY_HOUR_PENALTY"] * sum(overtime_vars))
        if streak_vars:
            objective_terms.append(cfg["CONSECUTIVE_PENALTY"] * sum(streak_vars))
        if hod_gap_vars:
            objective_terms.append(cfg["HOD_GAP_PENALTY"] * sum(hod_gap_vars))
        objective_terms.append(cfg["NIGHT_LEDGER_WEIGHT"] * max_cum_night)
        objective_terms.append(cfg["FAIRNESS_WEIGHT"] * max_shifts)
        model.Minimize(sum(objective_terms))

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = cfg["SOLVER_TIMEOUT_SECONDS"]
        status = solver.Solve(model)

        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            optimization_success = True
            shortage_summary["coverage"] = sum(solver.Value(v) for v in coverage_shortage_vars)
            shortage_summary["level_mix"] = sum(solver.Value(v) for v in level_shortage_vars)
            shortage_summary["incharge_days"] = sum(solver.Value(v) for v in incharge_shortage_vars)

            for d in range(1, num_days_next_month + 1):
                for s in ['Day', 'Night']:
                    shift_assignees = [n for n in range(num_nurses) if solver.Value(x[(n, d, s)])]
                    # Post-hoc: flag whichever assigned nurse (if any) is senior as in-charge (H3)
                    incharge_id = None
                    for n in shift_assignees:
                        if n in senior_nurses:
                            incharge_id = nurses_pool[n]['id']
                            break
                    for n in shift_assignees:
                        nurse = nurses_pool[n]
                        ai_draft.append({
                            "id": f"ai-hint-{d}-{s}-{nurse['id']}",
                            "day": d,
                            "reqShift": s,
                            "filledBy": nurse['id'],
                            "isInCharge": nurse['id'] == incharge_id
                        })

    # Generate Detailed Slots using the real per-level demand computed above (Phase 1)
    detailed_schedule = []
    level_time = {"Day": "08:00 - 16:00", "Night": "16:00 - 00:00"}
    for day in range(1, num_days_next_month + 1):
        for s in ['Day', 'Night']:
            for lvl, cnt in daily_level_demands[(day, s)].items():
                for _ in range(cnt):
                    detailed_schedule.append({
                        "day": day, "ward": "ER", "shiftType": s,
                        "time": level_time[s], "hours": SCHEDULING_CONFIG["SHIFT_HOURS"],
                        "reqLevel": lvl
                    })

    avg_m = int(sum(dow_demand['Day'].values()) / 7)
    avg_n = int(sum(dow_demand['Night'].values()) / 7)

    # Real predicted patient load, Day vs Night — this is the actual resolution the
    # model predicts at. Deliberately NOT split into fake hourly buckets: we don't have
    # an hourly-level model yet, and pretending to would be misleading, not more useful.
    day_load = round(avg_day_patients)
    night_load = round(avg_night_patients)
    chart_data = [
        {"period": "Day", "load": day_load},
        {"period": "Night", "load": night_load},
    ]

    recommendations = [
        { "ward": "ER", "shift": "Morning (08-16)", "predictedPatients": day_load, "currentStaff": avg_m, "recommendedStaff": avg_m, "status": "Optimal" },
        { "ward": "ER", "shift": "Afternoon (16-00)", "predictedPatients": night_load, "currentStaff": avg_n, "recommendedStaff": avg_n, "status": "Optimal" }
    ]

    # ------------------------------------------------
    # FTE — Required FTE = ADC x NHPOUS x 7/40 x non-productive-factor x FT-ratio
    # (NHPOUS/non-productive/FT-ratio are policy constants from the reference doc,
    # flagged there as needing sign-off before go-live — same caveat applies here.)
    # ------------------------------------------------
    active_fte = len(nurses_pool)
    required_fte = round(
        predicted_adc * cfg["NHPOUS_ER"] * (7 / 40) * cfg["NON_PRODUCTIVE_FACTOR"] * cfg["FT_RATIO"], 1
    )
    patient_forecast = {
        "avg_day_patients": round(avg_day_patients, 1),
        "avg_night_patients": round(avg_night_patients, 1),
        "predicted_adc": round(predicted_adc, 1),
        "month_total_patients": month_total_patients,
        "model_used": "XGBoost Machine Learning" if patient_use_xgb else "Statistical Baseline",
        "data_points": patient_data_rows,
    }
    fte_info = {
        "active_fte": active_fte,
        "required_fte": required_fte,
        "fill_rate_pct": round((active_fte / required_fte) * 100, 1) if required_fte > 0 else None,
        "nhpous": cfg["NHPOUS_ER"],
    }

    model_used = "XGBoost Machine Learning" if use_xgboost else "Statistical Baseline"
    llm_explanation = f"""
        <div class="ai-bullet-list">
            <div class="ai-bullet-item">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#64748b" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 7V5a2 2 0 0 1 2-2h2"></path><path d="M17 3h2a2 2 0 0 1 2 2v2"></path><path d="M21 17v2a2 2 0 0 1-2 2h-2"></path><path d="M7 21H5a2 2 0 0 1-2-2v-2"></path><circle cx="12" cy="12" r="2"></circle></svg>
                <span>Increase <b>Day Shift</b> coverage to address high patient volume predicted by the <b>{model_used}</b> model.</span>
            </div>
            <div class="ai-bullet-item">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#64748b" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 7V5a2 2 0 0 1 2-2h2"></path><path d="M17 3h2a2 2 0 0 1 2 2v2"></path><path d="M21 17v2a2 2 0 0 1-2 2h-2"></path><path d="M7 21H5a2 2 0 0 1-2-2v-2"></path><circle cx="12" cy="12" r="2"></circle></svg>
                <span>Add <b>{len(ai_draft)} Nurses</b> to upcoming shifts based on {total_data_rows} historical shift patterns to improve coverage.</span>
            </div>
            <div class="ai-bullet-item">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#64748b" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 7V5a2 2 0 0 1 2-2h2"></path><path d="M17 3h2a2 2 0 0 1 2 2v2"></path><path d="M21 17v2a2 2 0 0 1-2 2h-2"></path><path d="M7 21H5a2 2 0 0 1-2-2v-2"></path><circle cx="12" cy="12" r="2"></circle></svg>
                <span>{"No coverage or skill-mix gaps found" if shortage_summary["coverage"] == 0 and shortage_summary["level_mix"] == 0 else f'Flagged {shortage_summary["coverage"]} shift-slots short on headcount and {shortage_summary["level_mix"]} short on required level mix'}{f'; {shortage_summary["incharge_days"]} shift(s) have no Level 3/4/HOD in-charge available' if shortage_summary["incharge_days"] > 0 else ""} — review before confirming.</span>
            </div>
        </div>
    """

    return recommendations, chart_data, detailed_schedule, nurses_pool, saved_shifts, llm_explanation, ai_draft, patient_forecast, fte_info

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
        df_clean['employee_id'] = df_clean['employee_id'].apply(clean_employee_id)

        df_clean['level'] = df_clean['level'].apply(clean_nurse_level)

        date_cols = [col for col in df_clean.columns if col.isdigit() and 1 <= int(col) <= 31]
        df_melted = pd.melt(df_clean, id_vars=['employee_id', 'name', 'level'], value_vars=date_cols, var_name='day', value_name='shift_code')
        df_melted['shift_code'] = df_melted['shift_code'].astype(str).str.strip().str.upper()
        # Only drop truly-empty cells (no code entered at all). Leave/off codes below are
        # KEPT and stored as status='Leave' (H10) so the scheduler knows who is unavailable.
        EMPTY_CODES = {'NAN', '', 'NONE'}
        # Real codes from the hospital's own roster legend: "TR=train as schedule,
        # R=request Off, X=off, S=Supervisor, AL=Annual Leave, OC=Oncall, SL=Sick Leave,
        # O12/O14/O24=Oncall variants; D=07-15, E=15-23, N=23-07, D8=7-16, D9=8-17, D10=7-17, N12=19-7"
        LEAVE_CODES = {'X', 'R', 'OFF', 'AL', 'SL', 'HP'}
        # Annotations/roles seen in the day-cell grid that are neither a shift nor an
        # absence (e.g. "In-ch" marks who's in-charge on an already-coded shift, in a
        # sub-row our single-header read can't attach to the right cell) — and codes we
        # don't have a faithful Day/Night mapping for yet (Oncall, training marker).
        # Drop these rather than guess, so they can't corrupt data either direction.
        IGNORED_CODES = {'IN-CH', 'TR', 'S', 'O12', 'O14', 'O24', 'OC'}
        df_melted = df_melted[~df_melted['shift_code'].isin(EMPTY_CODES)]

        def parse_shift_code(code):
            c = str(code).upper().strip()
            if c in LEAVE_CODES: return pd.Series([None, None, 0, 'Leave'])
            if c in IGNORED_CODES: return pd.Series([None, None, None, None])
            if 'D12' in c: return pd.Series(['Day', 7, 12, 'Scheduled'])
            elif 'N12' in c or 'SN12' in c: return pd.Series(['Night', 19, 12, 'Scheduled'])
            elif 'D10' in c: return pd.Series(['Day', 7, 10, 'Scheduled'])
            elif 'D9' in c: return pd.Series(['Day', 8, 9, 'Scheduled'])
            elif 'D8' in c: return pd.Series(['Day', 7, 9, 'Scheduled'])
            elif c == 'E': return pd.Series(['Day', 15, 8, 'Scheduled'])
            elif c == 'D' or 'DAY' in c: return pd.Series(['Day', 7, 8, 'Scheduled'])
            elif c == 'N' or 'NIGHT' in c: return pd.Series(['Night', 23, 8, 'Scheduled'])
            elif 'D' in c: return pd.Series(['Day', 8, 8, 'Scheduled'])
            elif 'N' in c: return pd.Series(['Night', 23, 8, 'Scheduled'])
            else:
                # Unrecognized code — drop rather than silently guessing "worked a Day shift"
                return pd.Series([None, None, None, None])

        df_melted[['shift_type', 'start_hour', 'duration_hours', 'status']] = df_melted['shift_code'].apply(parse_shift_code)
        # Keep Leave rows (status='Leave') and recognized shifts (status='Scheduled');
        # drop ignored/unrecognized codes (status is None) instead of guessing.
        df_melted = df_melted[df_melted['status'].notna()]

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

                # Auto-register any employee_id from this roster that isn't in the nurses
                # master table yet (new hire). DO NOTHING on conflict so this never
                # clobbers a name/level someone already edited via nurse management.
                df_roster_nurses = df_final[['employee_id', 'name', 'level', 'ward']].drop_duplicates(subset=['employee_id'])
                with engine.begin() as conn:
                    for _, r in df_roster_nurses.iterrows():
                        conn.execute(text("""
                            INSERT INTO nurses (employee_id, name, level, ward, is_active, created_at, updated_at)
                            VALUES (:eid, :name, :level, :ward, TRUE, now(), now())
                            ON CONFLICT (employee_id) DO NOTHING
                        """), {"eid": r['employee_id'], "name": r['name'], "level": r['level'], "ward": r['ward']})
            except Exception as e:
                print(f"Data update failed: {e}")

        recs, chart_data, detailed_schedule, nurses_pool, saved_shifts, llm_exp, ai_draft, patient_forecast, fte_info = generate_real_recommendations()
        return { "nurses": nurses_pool, "recommendations": recs, "chart_data": chart_data, "detailed_schedule": detailed_schedule, "saved_shifts": saved_shifts, "llm_explanation": llm_exp, "ai_draft": ai_draft, "patient_forecast": patient_forecast, "fte_info": fte_info }
    except Exception as e:
        return {"nurses": [], "recommendations": [], "chart_data": [], "detailed_schedule": [], "saved_shifts": [], "llm_explanation": "", "ai_draft": [], "patient_forecast": {}, "fte_info": {}}

def predict_er_hourly():
    recs, chart_data, detailed_schedule, nurses_pool, saved_shifts, llm_exp, ai_draft, patient_forecast, fte_info = generate_real_recommendations()
    peak_period = max(chart_data, key=lambda x: x['load'])['period'] if chart_data else "Night"
    return {
        "current_load": "85%", "peak_hour": peak_period, "trend": "increasing",
        "chart_data": chart_data, "recommendations": recs, "detailed_schedule": detailed_schedule,
        "nurses": nurses_pool, "saved_shifts": saved_shifts, "llm_explanation": llm_exp, "ai_draft": ai_draft,
        "patient_forecast": patient_forecast, "fte_info": fte_info
    }

def predict_opd_daily(days=7): 
    return [{"date": f"Day {i+1}", "volume": random.randint(1200, 1600)} for i in range(days)]

def predict_icu_daily(): 
    return {"occupancy_rate": 88, "available_beds": 2}

def predict_dengue_risk(): 
    return {"risk_level": "High", "action": "Prepare for surge in 2 weeks"}