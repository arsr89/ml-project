"""
Model-serving API for the churn prediction model.
"""
from fastapi import FastAPI
from pydantic import BaseModel
from prometheus_fastapi_instrumentator import Instrumentator
import mlflow.pyfunc

app = FastAPI(title="Churn Prediction API")
Instrumentator().instrument(app).expose(app)

mlflow.set_tracking_uri("http://localhost:5000")
model = mlflow.pyfunc.load_model("models:/churn-model@production")


class ChurnRequest(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: ChurnRequest):
    import pandas as pd
    df = pd.DataFrame([req.dict()])
    prediction = model.predict(df)
    return {"churn_prediction": int(prediction[0])}
