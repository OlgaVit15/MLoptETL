FROM python:3.10-slim
WORKDIR app/
COPY requirements.txt .
RUN pip install -r requirements.txt
# Копируем общую библиотеку и код сервиса
COPY common/* common/
COPY ./services/impala_service/main.py .
COPY ./services/impala_service/collector.py .
COPY ./services/impala_service/balancer.py .
COPY ./services/impala_service/models/model_predict/ml_models/mem_regressor.py                ml_models/
COPY ./services/impala_service/models/model_predict/ml_models/mem_strategy_cascade.py          ml_models/
COPY ./services/impala_service/models/model_predict/ml_models/strategy_classifier.py  ml_models/
COPY ./services/impala_service/models/model_predict/models/impala_cascade_20260510_090620.joblib ml_models/models/
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]
