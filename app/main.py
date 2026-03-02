from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import opd, er, icu, dengue

app = FastAPI(title="Hospital Forecasting API")

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

@app.get("/")
def root():
    return {"message": "API is running"}