from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "model.joblib"

EXPERIMENT_NAME = "mlops-model-service"


def train_model() -> float:
    """
    Train a simple classifier, save it locally, and log the run to MLflow.

    Returns:
        Test accuracy.
    """
    X, y = load_breast_cancer(return_X_y=True)

    test_size = 0.2
    random_state = 42
    max_iter = 1000

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=max_iter, random_state=random_state)),
        ]
    )

    # regular way
    # model.fit(X_train, y_train)
    # predictions = model.predict(X_test)
    # accuracy = accuracy_score(y_test, predictions)

    # MODEL_DIR.mkdir(parents=True, exist_ok=True)
    # joblib.dump(model, MODEL_PATH)

    # print(f"Model saved to: {MODEL_PATH}")
    # print(f"Test accuracy: {accuracy:.4f}")
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    with mlflow.start_run(run_name="logistic-regression-baseline"):
        mlflow.log_param("dataset", "sklearn_breast_cancer")
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("max_iter", max_iter)

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        mlflow.log_metric("accuracy", accuracy)

        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, MODEL_PATH)

        mlflow.log_artifact(str(MODEL_PATH), artifact_path="model_artifact")
        mlflow.sklearn.log_model(model, name="model")

        print(f"Model saved to: {MODEL_PATH}")
        print(f"Test accuracy: {accuracy:.4f}")

    return accuracy

if __name__ == "__main__":
    train_model()