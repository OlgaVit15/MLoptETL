FROM python:3.10-slim

WORKDIR /app

COPY api/model_ml/impala/model_predict/mem_regressor.py        model_ml/impala/model_predict/
COPY api/model_ml/impala/model_predict/mem_strategy_cascade.py model_ml/impala/model_predict/
COPY api/model_ml/impala/model_predict/strategy_classifier.py  model_ml/impala/model_predict/
COPY api/model_ml/impala/model_predict/models/impala_cascade_20260425_175212.joblib model_ml/impala/model_predict/models/

COPY api/balancer/collect_service.py balancer/
COPY api/balancer/impala_balancer.py balancer/
COPY api/app.py .

RUN pip install --no-cache-dir pandas numpy scikit-learn joblib fastapi uvicorn requests sqlalchemy catboost

EXPOSE 4040

# Запуск с 4 воркерами для параллельной обработки запросов
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "4040", "--workers", "4"]

