import os
import pandas as pd
from sqlalchemy import create_engine, types
from dotenv import load_dotenv

# 1. โหลดค่าตัวแปรจาก .env
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    print("❌ ไม่พบ DATABASE_URL ในไฟล์ .env")
    exit()

# แปลง postgres:// เป็น postgresql:// สำหรับ SQLAlchemy
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

print("⏳ กำลังเชื่อมต่อกับฐานข้อมูล Neon...")
engine = create_engine(DATABASE_URL)

# 2. อ่านไฟล์ CSV จากโฟลเดอร์ data
csv_file_path = os.path.join(os.getcwd(), "data", "clean_nurse_schedule_feb2026.csv")

if not os.path.exists(csv_file_path):
    print(f"❌ ไม่พบไฟล์ CSV ที่: {csv_file_path}")
    exit()

print("⏳ กำลังอ่านไฟล์ CSV...")
df = pd.read_csv(csv_file_path)

# 3. จัดการประเภทข้อมูล (Data Types) ให้สวยงามก่อนลง Database
df['employee_id'] = df['employee_id'].apply(lambda x: str(int(x)) if pd.notnull(x) else x)
df['date'] = pd.to_datetime(df['date'])

# กำหนดประเภทข้อมูลของแต่ละคอลัมน์ใน PostgreSQL
dtype_mapping = {
    'employee_id': types.String(50),
    'name': types.String(100),
    'level': types.String(50),
    'ward': types.String(50),
    'date': types.Date(),
    'shift_type': types.String(20),
    'start_hour': types.Integer(),
    'duration_hours': types.Integer(),
    'status': types.String(50)
}

# 4. นำเข้าข้อมูลสู่ PostgreSQL
table_name = "nurse_schedule"
print(f"⏳ กำลังนำเข้าข้อมูลจำนวน {len(df)} แถว ลงตาราง '{table_name}'...")

try:
    df.to_sql(table_name, engine, if_exists='replace', index=False, dtype=dtype_mapping)
    print(f"✅ นำเข้าข้อมูลสำเร็จ! ตาราง '{table_name}' ถูกสร้างเรียบร้อยแล้ว")
except Exception as e:
    print(f"❌ เกิดข้อผิดพลาดระหว่างนำเข้า: {e}")