import os
from pathlib import Path

import mlflow


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENT_NAME = "MindMirror Emotion Classification"
TRACKING_URI_ENV_VAR = "MLFLOW_TRACKING_URI"
DEFAULT_TRACKING_DB = PROJECT_ROOT / "mlflow_new.db"


def get_tracking_uri() -> str:
    configured_uri = os.environ.get(TRACKING_URI_ENV_VAR, "").strip()
    if configured_uri:
        return configured_uri

    return f"sqlite:///{DEFAULT_TRACKING_DB.as_posix()}"


def configure_mlflow(create_experiment: bool = True):
    mlflow.set_tracking_uri(get_tracking_uri())

    if create_experiment:
        return mlflow.set_experiment(EXPERIMENT_NAME)

    return mlflow.get_experiment_by_name(EXPERIMENT_NAME)
