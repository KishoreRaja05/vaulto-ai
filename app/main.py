from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.chat import router as chat_router
from app.api.v1.ml import router as ml_router
from app.config import settings

app = FastAPI(
    title="Vaulto AI Microservice",
    description="Machine Learning predictions, anomaly detection, and transaction categorization",
    version="1.0.0",
)

origins = [origin.strip() for origin in settings.CORS_ORIGINS.split(",") if origin.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ml_router, prefix="/api/v1")
app.include_router(chat_router, prefix="/api/v1")


@app.get("/")
def read_root():
    return {"message": "Vaulto AI Microservice is running ✅"}
