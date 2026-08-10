import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

from app.routers import opd, er, icu, dengue, nurses

# โหลดค่า Environment Variables
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

# แก้ไข URL ให้รองรับ SQLAlchemy
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(DATABASE_URL) if DATABASE_URL else None

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Starting up API Server...")
    if engine:
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            print("✅ Successfully connected to Neon PostgreSQL Database!")
        except Exception as e:
            print(f"❌ Failed to connect to database: {e}")
    else:
        print("⚠️ Warning: DATABASE_URL not found in .env file.")
        
    yield 
    
    if engine:
        engine.dispose()
        print("🛑 Database connection closed.")

app = FastAPI(title="Hospital Forecasting API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(opd.router, prefix="/api/v1/opd", tags=["OPD"])
app.include_router(er.router, prefix="/api/v1/er", tags=["ER"])
app.include_router(icu.router, prefix="/api/v1/icu", tags=["ICU"])
app.include_router(dengue.router, prefix="/api/v1/dengue", tags=["Dengue"])
app.include_router(nurses.router, prefix="/api/v1/nurses", tags=["Nurses"])

@app.get("/")
def root():
    return {"message": "API is running, Database is connected!"}