# MLOps Model Service

A small production-style ML service for practicing end-to-end MLOps workflows.

The model is intentionally simple. The goal is not model performance. The goal is the production workflow around a model:

- repeatable training
- MLflow experiment tracking
- MLflow Tracking Server
- MLflow Model Registry
- FastAPI model serving
- automated tests
- GitHub Actions CI
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
  -> saves models/model.joblib locally
  -> logs params, metrics, artifacts, model signature, and input example to MLflow
  -> can optionally register a model version in MLflow Model Registry

MLflow Tracking Server
  -> stores tracking metadata in SQLite
  -> stores artifacts in a local artifact directory
  -> exposes experiment tracking UI and model registry UI

FastAPI app
  -> default mode: loads models/model.joblib
  -> registry mode: loads a registered MLflow model version
  -> serves /health, /predict, /metrics

Prometheus
  -> scrapes FastAPI /metrics
  -> stores metrics over time

Grafana
  -> queries Prometheus
  -> displays provisioned dashboard

GitHub Actions CI
  -> installs dependencies
  -> starts MLflow
  -> trains and registers the model once
  -> runs tests
  -> builds Docker Compose services
  -> verifies API, MLflow, Prometheus, and Grafana
```

---

## 2. Project structure

```text
mlops-model-service/
├── app/
│   ├── __init__.py
│   ├── main.py                  # FastAPI app, endpoints, Prometheus metrics
│   └── model_loader.py          # Loads local model or MLflow Registry model
│
├── docker/
│   ├── api/
│   │   └── Dockerfile           # FastAPI API image
│   └── mlflow/
│       └── Dockerfile           # MLflow Tracking Server image
│
├── training/
│   └── train.py                 # Training script with MLflow tracking/registration
│
├── tests/
│   ├── __init__.py
│   └── test_api.py              # API tests
│
├── models/
│   └── model.joblib             # Local model artifact, ignored by Git
│
├── monitoring/
│   ├── prometheus/
│   │   └── prometheus.yml       # Prometheus scrape config
│   └── grafana/
│       ├── provisioning/
│       │   ├── datasources/
│       │   │   └── prometheus.yml
│       │   └── dashboards/
│       │       └── dashboards.yml
│       └── dashboards/
│           └── mlops-model-service.json
│
├── .github/
│   └── workflows/
│       └── ci.yml               # GitHub Actions CI workflow
│
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
7. optionally registers the model in MLflow Model Registry

Example with default local MLflow tracking:

```bash
python training/train.py
```

Example with MLflow Tracking Server and Model Registry:

```bash
MLFLOW_TRACKING_URI=http://127.0.0.1:5001 python training/train.py \
  --registered-model-name mlops-fastapi-classifier
```

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

The model is loaded once during application startup and reused for prediction requests.

---

### `app/model_loader.py`

Loads the model in one of two modes.

Default local mode:

```text
models/model.joblib
```

MLflow Registry mode:

```text
models:/mlops-fastapi-classifier/1
```

The mode is controlled by environment variables:

```text
MLFLOW_TRACKING_URI=http://mlflow:5000
MLFLOW_MODEL_URI=models:/mlops-fastapi-classifier/1
```

If `MLFLOW_MODEL_URI` is not set, the API loads the local model file.

If `MLFLOW_MODEL_URI` is set, the API loads the model from MLflow Registry. If the registered model version does not exist, the API fails fast instead of silently falling back to the local model.

---

### `docker-compose.yml`

Runs the local production-like stack:

- FastAPI API
- MLflow Tracking Server
- Prometheus
- Grafana

Important Docker networking detail:

From your laptop, MLflow is available at:

```text
http://127.0.0.1:5001
```

From inside Docker Compose, the API reaches MLflow at:

```text
http://mlflow:5000
```

`mlflow` is the Docker Compose service name.

---

### `.github/workflows/ci.yml`

Runs CI on push and pull request.

Current CI workflow:

1. checks out the repository
2. installs Python dependencies
3. validates Docker Compose config
4. builds and starts MLflow
5. trains and registers the model once
6. runs API tests
7. builds Docker Compose services
8. starts the full stack
9. checks API health
10. checks API prediction
11. checks Prometheus health
12. checks Grafana health
13. prints Docker Compose logs on failure
14. stops Docker Compose

This validates the important production-style path:

```text
training -> MLflow Registry -> Dockerized API -> prediction endpoint
```

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

## 5. Train the model locally

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

The model file is ignored by Git.

---

## 6. MLflow local tracking mode

If no `MLFLOW_TRACKING_URI` is set, MLflow uses local file-based tracking.

Run:

```bash
python training/train.py
```

Then start local MLflow UI:

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

## 7. MLflow Tracking Server mode

The Docker Compose stack includes an MLflow Tracking Server.

Start only MLflow:

```bash
docker compose up -d --build mlflow
```

Open:

```text
http://127.0.0.1:5001
```

Train and register a model version:

```bash
MLFLOW_TRACKING_URI=http://127.0.0.1:5001 python training/train.py \
  --registered-model-name mlops-fastapi-classifier
```

Then open MLflow UI and go to:

```text
Model training -> Models -> mlops-fastapi-classifier
```

You should see a registered model version.

---

## 8. MLflow storage layout

The Dockerized MLflow server uses:

```text
mlflow-data/      # SQLite backend database for metadata
mlartifacts/      # artifact storage for model files and run artifacts
```

These folders are ignored by Git.

Metadata includes:

- experiments
- runs
- params
- metrics
- tags
- registered models
- model versions

Artifacts include:

- model files
- MLmodel metadata
- input examples
- environment files
- logged artifacts

---

## 9. Run the API locally without Docker

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

This uses the local model file by default:

```text
models/model.joblib
```

---

## 10. Run the API locally with an MLflow Registry model

Start MLflow and make sure a registered model version exists:

```bash
docker compose up -d --build mlflow

MLFLOW_TRACKING_URI=http://127.0.0.1:5001 python training/train.py \
  --registered-model-name mlops-fastapi-classifier
```

Then start the API locally with registry loading:

```bash
MLFLOW_TRACKING_URI=http://127.0.0.1:5001 \
MLFLOW_MODEL_URI=models:/mlops-fastapi-classifier/1 \
uvicorn app.main:app --reload
```

This tells the API to load model version `1` from MLflow Registry.

---

## 11. Run the full Docker Compose stack

For the current production-style workflow, first start MLflow and register the model:

```bash
docker compose down

docker compose up -d --build mlflow

MLFLOW_TRACKING_URI=http://127.0.0.1:5001 python training/train.py \
  --registered-model-name mlops-fastapi-classifier
```

Then start the full stack:

```bash
docker compose up -d --build
```

Check services:

```bash
docker compose ps
```

Useful URLs:

```text
FastAPI:
http://127.0.0.1:8000

MLflow:
http://127.0.0.1:5001

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

Stop everything:

```bash
docker compose down
```

---

## 12. Example prediction request

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

## 13. Run tests

```bash
python -m pytest
```

Expected:

```text
4 passed
```

---

## 14. Docker commands

Build the API image manually:

```bash
docker build -f docker/api/Dockerfile -t mlops-model-service:local .
```

Run the API container manually with local model loading:

```bash
docker run --rm -p 8000:8000 mlops-model-service:local
```

The full stack should usually be run with Docker Compose instead of manual Docker commands.

---

## 15. Monitoring flow

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

## 16. Prometheus config

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

## 17. Grafana provisioning

Grafana is configured from files, not manual UI clicks.

Docker Compose mounts:

```yaml
./monitoring/grafana/provisioning:/etc/grafana/provisioning
./monitoring/grafana/dashboards:/var/lib/grafana/dashboards
```

This means Grafana reads local configuration files when it starts.

Datasource:

```text
monitoring/grafana/provisioning/datasources/prometheus.yml
```

Dashboard provider:

```text
monitoring/grafana/provisioning/dashboards/dashboards.yml
```

Dashboard JSON:

```text
monitoring/grafana/dashboards/mlops-model-service.json
```

---

## 18. Useful PromQL queries

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

## 19. Generate test traffic

Run this while the API is running:

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

## 20. Debugging

### API does not start

```bash
docker compose ps
docker compose logs api
```

Common causes:

- registered model version does not exist
- `MLFLOW_TRACKING_URI` points to the wrong address
- MLflow server is not ready
- model dependency versions differ between training and serving

For Docker Compose, the API should use:

```text
MLFLOW_TRACKING_URI=http://mlflow:5000
```

not:

```text
MLFLOW_TRACKING_URI=http://127.0.0.1:5001
```

Inside Docker, `127.0.0.1` means the API container itself, not your laptop.

---

### MLflow returns 403

If MLflow returns:

```text
Invalid Host header - possible DNS rebinding attack detected
```

make sure the MLflow server is started with an allowed hosts setting suitable for local Docker Compose development.

For this local learning project, the MLflow Dockerfile uses:

```text
--allowed-hosts *
```

This is acceptable for local development, but should be restricted in a real production deployment.

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

## 21. Common commands

```bash
# install dependencies
pip install -r requirements.txt

# train model locally
python training/train.py

# start MLflow server
 docker compose up -d --build mlflow

# train and register model in MLflow server
MLFLOW_TRACKING_URI=http://127.0.0.1:5001 python training/train.py \
  --registered-model-name mlops-fastapi-classifier

# run API locally
uvicorn app.main:app --reload

# run tests
python -m pytest

# build API Docker image
 docker build -f docker/api/Dockerfile -t mlops-model-service:local .

# run full stack
 docker compose up -d --build

# stop full stack
 docker compose down

# follow logs
 docker compose logs -f
```

---

## 22. Current limitations

This is a learning project, not a full production deployment.

Current limitations:

- MLflow uses local SQLite instead of Postgres/RDS
- MLflow artifacts use local filesystem instead of S3/object storage
- the Docker Compose MLflow server uses permissive local `--allowed-hosts *`
- no authentication
- no persistent Grafana database
- no alerting rules yet
- no cloud deployment yet
- training is still run from the host machine, not from a dedicated training container
- API currently loads a fixed registered model version instead of a movable alias like `champion`

These limitations are intentional for now. The project is being built incrementally.

---

## 23. Next planned steps

Planned improvements:

1. Finish CI validation for registry-to-serving workflow.
2. Add alerting rules.
3. Consider model alias workflow, for example `champion`.
4. Consider training container.
5. Consider AWS deployment.
6. Consider replacing local SQLite/artifacts with RDS/S3-style storage.
