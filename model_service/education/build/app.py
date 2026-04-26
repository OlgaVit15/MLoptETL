from fastapi import FastAPI

from model_ml.ensemble2 import ImpalaCascadeEnsemble

app = FastAPI()
model = ImpalaCascadeEnsemble.load("model_ml/impala_cascade_ensemble_model.joblib")


@app.post("/predict")
def predict(data: dict):
    # На вход ожидаем JSON с ключами feature_plan_...
    return model.predict(data)
