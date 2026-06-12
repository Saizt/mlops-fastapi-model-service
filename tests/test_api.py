from fastapi.testclient import TestClient

from app.main import app


VALID_FEATURES = [
    17.99, 10.38, 122.8, 1001.0, 0.1184,
    0.2776, 0.3001, 0.1471, 0.2419, 0.07871,
    1.095, 0.9053, 8.589, 153.4, 0.006399,
    0.04904, 0.05373, 0.01587, 0.03003, 0.006193,
    25.38, 17.33, 184.6, 2019.0, 0.1622,
    0.6656, 0.7119, 0.2654, 0.4601, 0.1189,
]


def test_health_check():
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_success():
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={"features": VALID_FEATURES},
        )

    assert response.status_code == 200

    body = response.json()

    assert "prediction" in body
    assert "probability" in body

    assert isinstance(body["prediction"], int)
    assert isinstance(body["probability"], float)

    assert body["prediction"] in [0, 1]
    assert 0.0 <= body["probability"] <= 1.0


def test_predict_rejects_wrong_feature_length():
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={"features": [1.0, 2.0, 3.0]},
        )

    assert response.status_code == 422


def test_predict_rejects_missing_features():
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={},
        )

    assert response.status_code == 422