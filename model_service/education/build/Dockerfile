FROM python:3.10-slim

WORKDIR /app

# # Копируем файл зависимостей
# COPY requirements.txt .
#
# # Установка зависимостей из requirements.txt
# RUN pip install --no-cache-dir -r requirements.txt
RUN pip install pandas numpy scikit-learn joblib fastapi uvicorn
# Копируем код и модель
COPY model_ml/ensemble2.py model_ml/
COPY model_ml/impala_cascade_ensemble_model.joblib model_ml/
COPY app.py .

EXPOSE 4040

# Запуск с 4 воркерами для параллельной обработки запросов
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "4040", "--workers", "4"]
