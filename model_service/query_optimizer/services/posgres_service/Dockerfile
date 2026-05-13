FROM python:3.10-slim
WORKDIR app/
COPY requirements.txt .
RUN pip install -r requirements.txt
# Копируем общую библиотеку и код сервиса
COPY common/* common/
COPY ./services/posgres_service/main.py .
COPY ./services/posgres_service/collector.py .
COPY ./services/posgres_service/balancer.py .
COPY ./services/posgres_service/models/model_predict/ml_models/err_regressor.py                ml_models/
COPY ./services/posgres_service/models/model_predict/ml_models/pg_cascade_strategy.py          ml_models/
COPY ./services/posgres_service/models/model_predict/ml_models/pg_classification_mlcascade.py  ml_models/
COPY ./services/posgres_service/models/model_predict/ml_models/models/impala_cascade_20260509_182026.joblib ml_models/models/
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
