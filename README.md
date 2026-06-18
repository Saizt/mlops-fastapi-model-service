# MLOps Model Service

A small production-style ML service for practicing end-to-end MLOps workflows.

The model is intentionally simple. The goal is not model performance, but the production workflow around a model:

- repeatable training
- MLflow experiment tracking
- FastAPI model serving
- automated tests
- Dockerized runtime
- Docker Compose service orchestration
- Prometheus metrics
- Grafana dashboards
- reproducible monitoring configuration

---

## 1. Architecture

```text
training/train.py
  -> trains a simple sklearn model
  -> saves models/model.joblib
  -> logs run to MLflow

FastAPI app
  -> loads models/model.joblib
  -> serves /health, /predict, /metrics

Prometheus
  -> scrapes FastAPI /metrics
  -> stores metrics over time

Grafana
  -> queries Prometheus
  -> displays provisioned dashboard
```

---

## 2. Project structure

```text
mlops-model-service/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI app, endpoints, Prometheus metrics
│   └── model_loader.py      # Loads models/model.joblib
│
├── training/
│   └── train.py             # Training script with MLflow tracking
│
├── tests/
│   ├── __init__.py
│   └── test_api.py          # API tests
│
├── models/
│   └── model.joblib         # Local model artifact, ignored by Git
│
├── monitoring/
│   ├── prometheus/
│   │   └── prometheus.yml   # Prometheus scrape config
│   └── grafana/
│       ├── provisioning/
│       │   ├── datasources/
│       │   │   └── prometheus.yml
│       │   └── dashboards/
│       │       └── dashboards.yml
│       └── dashboards/
│           └── mlops-model-service.json
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .dockerignore
├── .gitignore
└── README.md
```

---

## 3. Key files

### `training/train.py`

Trains a simple scikit-learn model and logs the run to MLflow.

It:

1. loads the built-in breast cancer dataset
2. splits data into train/test
3. trains a `Pipeline(StandardScaler + LogisticRegression)`
4. calculates accuracy
5. saves the local model to `models/model.joblib`
6. logs params, metrics, artifact, model signature, and input example to MLflow

---

### `app/main.py`

Defines the FastAPI application.

Endpoints:

- `GET /health` — health check
- `POST /predict` — prediction endpoint
- `GET /metrics` — Prometheus metrics endpoint

Custom metrics:

- `prediction_requests_total`
- `prediction_errors_total`
- `prediction_latency_seconds`

---

### `app/model_loader.py`

Loads the saved model from:

```text
models/model.joblib
```

The API loads the model once during startup and reuses it for prediction requests.

---

### `tests/test_api.py`

Tests the API contract:

- `/health` returns `200`
- `/predict` works with valid input
- `/predict` rejects missing features
- `/predict` rejects wrong feature length

---

### `docker-compose.yml`

Runs the local stack:

- API
- Prometheus
- Grafana

This is the main entry point for the full local production-like environment.

---

## 4. Local Python setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Verify Python is from `.venv`:

```bash
which python
```

Expected output should contain:

```text
.venv/bin/python
```

---

## 5. Train the model

Default training run:

```bash
python training/train.py
```

Training with custom parameters:

```bash
python training/train.py --test-size 0.25 --max-iter 500
```

This creates or overwrites:

```text
models/model.joblib
```

The model file is ignored by Git. In real production, model artifacts usually live in artifact storage or a model registry.

---

## 6. MLflow

The training script logs each run to MLflow.

Start MLflow UI:

```bash
mlflow ui
```

Open:

```text
http://127.0.0.1:5000
```

Local MLflow runs are stored in:

```text
mlruns/
```

This folder is ignored by Git.

---

## 7. Run the API locally

Start FastAPI without Docker:

```bash
uvicorn app.main:app --reload
```

Useful URLs:

```text
Health check:
http://127.0.0.1:8000/health

Swagger docs:
http://127.0.0.1:8000/docs

Metrics:
http://127.0.0.1:8000/metrics
```

---

## 8. Example prediction request

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "features": [
      17.99, 10.38, 122.8, 1001.0, 0.1184,
      0.2776, 0.3001, 0.1471, 0.2419, 0.07871,
      1.095, 0.9053, 8.589, 153.4, 0.006399,
      0.04904, 0.05373, 0.01587, 0.03003, 0.006193,
      25.38, 17.33, 184.6, 2019.0, 0.1622,
      0.6656, 0.7119, 0.2654, 0.4601, 0.1189
    ]
  }'
```

Expected response shape:

```json
{
  "prediction": 0,
  "probability": 0.99
}
```

The exact probability may vary.

---

## 9. Run tests

```bash
python -m pytest
```

Expected:

```text
4 passed
```

---

## 10. Docker

Build the API image manually:

```bash
docker build -t mlops-model-service:local .
```

Run the API container manually:

```bash
docker run --rm -p 8000:8000 mlops-model-service:local
```

Open:

```text
http://127.0.0.1:8000/health
```

Stop with `Control + C`.

---

## 11. Docker Compose stack

Start API, Prometheus, and Grafana:

```bash
docker compose up --build
```

Start in background:

```bash
docker compose up -d --build
```

Stop everything:

```bash
docker compose down
```

Show running services:

```bash
docker compose ps
```

Follow logs:

```bash
docker compose logs -f
```

---

## 12. Local service URLs

When Docker Compose is running:

```text
FastAPI:
http://127.0.0.1:8000

Prometheus:
http://127.0.0.1:9090

Prometheus targets:
http://127.0.0.1:9090/targets

Grafana:
http://127.0.0.1:3000
```

Grafana credentials for local development:

```text
username: admin
password: admin
```

---

## 13. Monitoring flow

```text
FastAPI app
  exposes /metrics
        ↓
Prometheus
  scrapes http://api:8000/metrics every 5 seconds
        ↓
Grafana
  queries Prometheus and displays dashboards
```

Important Docker networking detail:

From your laptop, the API is available at:

```text
http://127.0.0.1:8000
```

From inside Docker Compose, Prometheus reaches the API at:

```text
http://api:8000
```

`api` is the Docker Compose service name.

---

## 14. Prometheus config

Prometheus config lives here:

```text
monitoring/prometheus/prometheus.yml
```

Current config:

```yaml
global:
  scrape_interval: 5s

scrape_configs:
  - job_name: "mlops-model-service-api"
    metrics_path: /metrics
    static_configs:
      - targets: ["api:8000"]
```

Meaning:

- scrape metrics every 5 seconds
- call `/metrics`
- target the API service inside Docker Compose

---

## 15. Grafana provisioning

Grafana is configured from files, not manual UI clicks.

Docker Compose mounts:

```yaml
./monitoring/grafana/provisioning:/etc/grafana/provisioning
./monitoring/grafana/dashboards:/var/lib/grafana/dashboards
```

This means Grafana reads local configuration files when it starts.

### Datasource

```text
monitoring/grafana/provisioning/datasources/prometheus.yml
```

Creates the Prometheus datasource automatically.

### Dashboard provider

```text
monitoring/grafana/provisioning/dashboards/dashboards.yml
```

Tells Grafana to load dashboards from:

```text
/var/lib/grafana/dashboards
```

### Dashboard JSON

```text
monitoring/grafana/dashboards/mlops-model-service.json
```

Defines the actual dashboard panels and PromQL queries.

---

## 16. Useful PromQL queries

Total prediction requests:

```promql
prediction_requests_total
```

Prediction requests per second:

```promql
rate(prediction_requests_total[1m])
```

Average prediction latency:

```promql
rate(prediction_latency_seconds_sum[1m])
/
rate(prediction_latency_seconds_count[1m])
```

p95 prediction latency:

```promql
histogram_quantile(0.95, rate(prediction_latency_seconds_bucket[5m]))
```

Prediction errors:

```promql
prediction_errors_total
```

Prediction error rate:

```promql
rate(prediction_errors_total[1m])
```

---

## 17. Generate test traffic

Run this while Docker Compose is running:

```bash
for i in {1..30}; do
  curl -s -X POST "http://127.0.0.1:8000/predict" \
    -H "Content-Type: application/json" \
    -d '{
      "features": [
        17.99, 10.38, 122.8, 1001.0, 0.1184,
        0.2776, 0.3001, 0.1471, 0.2419, 0.07871,
        1.095, 0.9053, 8.589, 153.4, 0.006399,
        0.04904, 0.05373, 0.01587, 0.03003, 0.006193,
        25.38, 17.33, 184.6, 2019.0, 0.1622,
        0.6656, 0.7119, 0.2654, 0.4601, 0.1189
      ]
    }' > /dev/null
done
```

Then refresh Grafana.

Rate-based panels may need 5-10 seconds before showing values.

---

## 18. Debugging

### API does not start

```bash
docker compose ps
docker compose logs api
```

Check that the model exists:

```bash
ls -lh models
```

If missing:

```bash
python training/train.py
docker compose up --build
```

---

### API health check fails

```bash
curl http://127.0.0.1:8000/health
```

Expected:

```json
{"status":"ok"}
```

If it fails:

```bash
docker compose logs api
```

---

### Prometheus has no metrics

Check API metrics:

```bash
curl http://127.0.0.1:8000/metrics
```

Check Prometheus targets:

```text
http://127.0.0.1:9090/targets
```

Target should be `UP`.

If target is `DOWN`, check:

```bash
docker compose logs prometheus
```

Also verify `monitoring/prometheus/prometheus.yml` uses:

```yaml
targets: ["api:8000"]
```

not:

```yaml
targets: ["127.0.0.1:8000"]
```

---

### Grafana dashboard is missing

Check Grafana logs:

```bash
docker compose logs grafana
```

Check mounted files inside the Grafana container:

```bash
docker exec -it mlops-model-service-grafana sh
```

Inside the container:

```bash
ls /etc/grafana/provisioning
ls /etc/grafana/provisioning/datasources
ls /etc/grafana/provisioning/dashboards
ls /var/lib/grafana/dashboards
```

If files are missing, the Docker Compose volume mounts are wrong.

---

### Grafana panels are empty

Check Prometheus directly:

```text
http://127.0.0.1:9090
```

Query:

```promql
prediction_requests_total
```

If Prometheus has data but Grafana does not, check:

- Grafana datasource
- dashboard query
- selected time range
- dashboard refresh interval

If Prometheus does not have data, check:

- FastAPI `/metrics`
- Prometheus scrape config
- Docker networking

---

## 19. Common commands

```bash
# install dependencies
pip install -r requirements.txt

# train model
python training/train.py

# run MLflow UI
mlflow ui

# run API locally
uvicorn app.main:app --reload

# run tests
python -m pytest

# build Docker image
docker build -t mlops-model-service:local .

# run full stack
docker compose up --build

# run full stack in background
docker compose up -d --build

# stop full stack
docker compose down

# follow logs
docker compose logs -f
```

---

## 20. Current limitations

This is a learning project, not a full production deployment.

Current limitations:

- model artifact is copied into the Docker image
- MLflow runs locally, not as a remote tracking server
- no model registry workflow yet
- no CI/CD yet
- no cloud deployment yet
- no authentication
- no persistent Grafana database
- no alerting rules yet

These limitations are intentional for now. The project is being built incrementally.

---

## 21. Next planned steps

Planned improvements:

1. Add GitHub Actions CI.
2. Run tests automatically on push.
3. Add linting and formatting.
4. Build Docker image in CI.
5. Add MLflow tracking server as a service.
6. Add model registry workflow.
7. Add AWS deployment.
8. Add alerting rules.
