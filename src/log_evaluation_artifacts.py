from pathlib import Path
import tempfile
from uuid import uuid4

import matplotlib.pyplot as plt
import mlflow
from mlflow.tracking import MlflowClient
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)
from tensorflow.keras.models import load_model

from mlflow_config import configure_mlflow
from preprocessing import (
    DATA_PATH,
    load_data,
    preprocess_images,
    split_data,
)


ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT_DIR / "models"
MODELS = {
    "Baseline CNN": MODEL_DIR / "emotion_model.keras",
    "Augmented CNN": MODEL_DIR / "emotion_model_augmented.keras",
    "Class Weighted CNN": MODEL_DIR / "emotion_model_weighted.keras",
}
EMOTION_NAMES = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral",
]


def find_model_run(runs, model_name: str, model_filename: str):
    matches = []

    for run in runs:
        tags = run.data.tags
        run_type = tags.get("run_type")

        if tags.get("model_file") == model_filename:
            if run_type == "existing_model_evaluation":
                priority = 0
            elif run_type == "training":
                priority = 1
            else:
                continue
        elif (
            run.info.run_name == model_name
            and "model_file" not in tags
            and run_type is None
        ):
            priority = 2
        else:
            continue

        matches.append((priority, run.info.start_time, run.info.run_id, run))

    if not matches:
        return None

    return min(matches, key=lambda item: (item[0], -item[1], item[2]))[3]


def main() -> None:
    available_models = [
        (model_name, model_path)
        for model_name, model_path in MODELS.items()
        if model_path.is_file()
    ]
    for model_name, model_path in MODELS.items():
        if not model_path.is_file():
            print(f"Skipping {model_name}: model file not found at {model_path}")

    if not available_models:
        print("No model files were found; no evaluation artifacts were created.")
        return

    experiment = configure_mlflow(create_experiment=False)
    if experiment is None:
        print(
            "MLflow experiment not found; no experiment was created and "
            "no artifacts were logged."
        )
        return

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        output_format="list",
    )
    targets = []
    for model_name, model_path in available_models:
        run = find_model_run(runs, model_name, model_path.name)
        if run is None:
            print(
                f"Skipping {model_name}: no matching training or "
                "existing-model evaluation run was found."
            )
        else:
            targets.append((model_name, model_path, run))

    if not targets:
        print("No matching MLflow runs were found; no artifacts were created.")
        return

    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"FER2013 dataset not found at {DATA_PATH}. "
            "Place the dataset there before creating evaluation artifacts."
        )

    print("\nLoading dataset...")
    X, y = load_data()
    X = preprocess_images(X)
    _, _, X_test, _, _, y_test = split_data(X, y)

    client = MlflowClient()

    for model_name, model_path, run in targets:
        print(f"\nEvaluating {model_name} for MLflow run {run.info.run_id}")
        model = load_model(model_path)
        predictions = model.predict(
            X_test,
            batch_size=64,
            verbose=1,
        )
        y_pred = np.argmax(predictions, axis=1)
        report = classification_report(
            y_test,
            y_pred,
            labels=list(range(len(EMOTION_NAMES))),
            target_names=EMOTION_NAMES,
            digits=4,
            zero_division=0,
            output_dict=False,
        )
        if not isinstance(report, str):
            raise TypeError("Expected classification_report to return text.")
        matrix = confusion_matrix(
            y_test,
            y_pred,
            labels=list(range(len(EMOTION_NAMES))),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            report_path = temp_path / "classification_report.txt"
            report_path.write_text(report, encoding="utf-8")

            matrix_path = temp_path / "confusion_matrix.png"
            plt.figure(figsize=(8, 7))
            plt.imshow(matrix)
            plt.title(f"{model_name} - Confusion Matrix")
            plt.colorbar()
            plt.xticks(
                range(len(EMOTION_NAMES)),
                EMOTION_NAMES,
                rotation=45,
                ha="right",
            )
            plt.yticks(range(len(EMOTION_NAMES)), EMOTION_NAMES)
            plt.xlabel("Predicted Label")
            plt.ylabel("True Label")
            for row in range(matrix.shape[0]):
                for column in range(matrix.shape[1]):
                    plt.text(
                        column,
                        row,
                        matrix[row, column],
                        ha="center",
                        va="center",
                    )
            plt.tight_layout()
            plt.savefig(matrix_path, dpi=150)
            plt.close()

            artifact_path = f"evaluation/{uuid4().hex}"
            client.log_artifact(
                run.info.run_id,
                str(report_path),
                artifact_path=artifact_path,
            )
            client.log_artifact(
                run.info.run_id,
                str(matrix_path),
                artifact_path=artifact_path,
            )

        print(
            f"Logged classification report and confusion matrix under "
            f"{artifact_path} for run {run.info.run_id}"
        )


if __name__ == "__main__":
    main()
