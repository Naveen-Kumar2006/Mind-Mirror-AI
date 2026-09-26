from pathlib import Path
import tempfile

import mlflow
import mlflow.tensorflow
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)

from tensorflow.keras.models import load_model

from preprocessing import (
    load_data,
    preprocess_images,
    split_data,
)


# ============================================================
# PATHS
# ============================================================

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


# ============================================================
# MLflow
# ============================================================

mlflow.set_tracking_uri(
    f"sqlite:///{ROOT_DIR / 'mlflow_new.db'}"
)

mlflow.set_experiment(
    "MindMirror Emotion Classification"
)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading dataset...")

X, y = load_data()

X = preprocess_images(X)

(
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test,
) = split_data(X, y)


# ============================================================
# EVALUATE EACH MODEL
# ============================================================

for model_name, model_path in MODELS.items():

    print("\n==========================================")
    print(f"EVALUATING: {model_name}")
    print("==========================================")

    if not model_path.exists():
        print(f"Model not found: {model_path}")
        continue

    model = load_model(model_path)

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    predictions = model.predict(
        X_test,
        batch_size=64,
        verbose=1,
    )

    y_pred = np.argmax(predictions, axis=1)

    # --------------------------------------------------------
    # Classification Report
    # --------------------------------------------------------

    report = classification_report(
        y_test,
        y_pred,
        target_names=EMOTION_NAMES,
        digits=4,
        zero_division=0,
    )

    print("\nClassification Report:")
    print(report)

    # --------------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_test,
        y_pred,
    )

    # --------------------------------------------------------
    # Temporary artifact directory
    # --------------------------------------------------------

    with tempfile.TemporaryDirectory() as temp_dir:

        temp_dir = Path(temp_dir)

        # ====================================================
        # Classification Report
        # ====================================================

        report_path = (
            temp_dir / "classification_report.txt"
        )

        report_path.write_text(
            report,
            encoding="utf-8",
        )

        # ====================================================
        # Confusion Matrix
        # ====================================================

        cm_path = (
            temp_dir / "confusion_matrix.png"
        )

        plt.figure(figsize=(8, 7))

        plt.imshow(cm)

        plt.title(
            f"{model_name} - Confusion Matrix"
        )

        plt.colorbar()

        plt.xticks(
            range(len(EMOTION_NAMES)),
            EMOTION_NAMES,
            rotation=45,
            ha="right",
        )

        plt.yticks(
            range(len(EMOTION_NAMES)),
            EMOTION_NAMES,
        )

        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")

        # Write values inside cells
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(
                    j,
                    i,
                    cm[i, j],
                    ha="center",
                    va="center",
                )

        plt.tight_layout()

        plt.savefig(
            cm_path,
            dpi=150,
        )

        plt.close()

        # ====================================================
        # FIND EXISTING MLflow RUN
        # ====================================================

        experiment = mlflow.get_experiment_by_name(
            "MindMirror Emotion Classification"
        )

        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"tags.mlflow.runName = '{model_name}'",
        )

        if runs.empty:
            print(
                f"\nWARNING: No existing MLflow run found "
                f"for {model_name}"
            )
            continue

        run_id = runs.iloc[0]["run_id"]

        print(f"\nMLflow Run ID: {run_id}")

        # ====================================================
        # LOG ARTIFACTS TO EXISTING RUN
        # ====================================================

        with mlflow.start_run(
            run_id=run_id
        ):

            mlflow.log_artifact(
                str(report_path),
                artifact_path="evaluation",
            )

            mlflow.log_artifact(
                str(cm_path),
                artifact_path="evaluation",
            )

        print(
            f"Artifacts logged successfully for "
            f"{model_name}"
        )


print("\n==========================================")
print("EVALUATION ARTIFACTS COMPLETE")
print("==========================================")