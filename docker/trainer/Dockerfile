FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY training/ training/

RUN mkdir -p models

CMD ["python", "training/train.py", "--registered-model-name", "mlops-fastapi-classifier"]