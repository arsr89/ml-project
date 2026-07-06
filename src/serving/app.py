"""
Minimal model-serving API.
Swap `load_model()` for mlflow.pyfunc.load_model("models:/<name>/Production")
once you have a registered model.
"""
from fastapi import FastAPI
from pydantic import BaseModel
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI(title="Model Serving API")
Instrumentator().instrument(app).expose(app)  # exposes /metrics for Prometheus


class PredictRequest(BaseModel):
    features: list[float]


def load_model():
    # Placeholder — replace with:
    # import mlflow
    # return mlflow.pyfunc.load_model("models:/your-model/Production")
    class DummyModel:
        def predict(self, X):
            return [sum(row) for row in X]
    return DummyModel()


model = load_model()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    prediction = model.predict([req.features])
    return {"prediction": prediction}
