import os


from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from src.training import train
from src.predict import predict



app = FastAPI(title="Anomaly Detection API")

DATABASE_URL = os.getenv("DATABASE_URL")


if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL environment variable is not set")

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True
)

class TrainingRequest(BaseModel):
    category: str
    epochs: int = 30
    save_model: bool = True



class PredictionRequest(BaseModel):
    image_path: str
    category: str


@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "anomaly-detection-api"
    }



@app.get("/health/db")
def database_health():
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return {
            "status": "ok",
            "database": "connected"
        }
    except SQLAlchemyError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Database connection failed: {str(e)}"
        )



@app.post("/training")
def training_endpoint(request: TrainingRequest):
    try:
        result = train(
            request.category,
            epochs=request.epochs,
            save_model=request.save_model
        )
        return {
            "status": "success",
            "message": f"Training completed for {request.category}",
            "result": result
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict")
def prediction_endpoint(request: PredictionRequest):
    try:
        return predict(
            request.image_path,
            request.category
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
