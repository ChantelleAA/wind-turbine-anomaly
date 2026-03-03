FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY mlflow/ ./mlflow/

EXPOSE 8080

CMD ["python", "src/serve/inference_api.py"]