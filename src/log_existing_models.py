from pathlib import Path

import mlflow
import mlflow.tensorflow
import tensorflow as tf
import numpy as np

from sklearn.metrics import accuracy_score

from preprocessing import (
    load_data,
    preprocess_images,
    split_data,
    encode_labels,
)


# ============================================================
# PATHS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

AUGMENTED_MODEL = (
    ROOT_DIR / "models" / "emotion_model_augmented.keras"
)


# ============================================================
# MLFLOW
# ============================================================

mlflow.set_tracking_uri(
    "sqlite:///C:/projects/MindMirror AI/mlflow_new.db"
)

mlflow.set_experiment(
    "MindMirror Emotion Classification"
)


# ============================================================
# LOAD TEST DATA
# ============================================================

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

(
    y_train,
    y_val,
    y_test,
) = encode_labels(
    y_train,
    y_val,
    y_test
)

print("\nTest data:", X_test.shape)


# ============================================================
# FUNCTION TO LOG MODEL
# ============================================================

def log_model(model_path, run_name):

    print("\n================================")
    print(run_name)
    print("================================")

    # Load existing model
    model = tf.keras.models.load_model(model_path)

    # Predictions
    predictions = model.predict(
        X_test,
        verbose=1
    )

    predictions = np.argmax(
        predictions,
        axis=1
    )

    actual = np.argmax(
        y_test,
        axis=1
    )

    # Accuracy
    accuracy = accuracy_score(
        actual,
        predictions
    )

    # Loss
    loss = model.evaluate(
        X_test,
        y_test,
        verbose=0
    )[0]

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Loss     : {loss:.4f}")

    # ========================================================
    # MLFLOW RUN
    # ========================================================

    with mlflow.start_run(
        run_name=run_name
    ):

        # Parameters
        mlflow.log_param(
            "model_file",
            model_path.name
        )

        mlflow.log_param(
            "model_type",
            run_name
        )

        mlflow.log_param(
            "total_parameters",
            model.count_params()
        )

        # Metrics
        mlflow.log_metric(
            "test_accuracy",
            float(accuracy)
        )

        mlflow.log_metric(
            "test_loss",
            float(loss)
        )

        # Model artifact
        mlflow.tensorflow.log_model(
            model,
            artifact_path="model"
        )

        print(
            "MLflow run:",
            mlflow.active_run().info.run_id
        )


# ============================================================
# LOG AUGMENTED CNN
# ============================================================

log_model(
    AUGMENTED_MODEL,
    "Augmented CNN"
)


# ============================================================
# DONE
# ============================================================

print("\n================================")
print("AUGMENTED MODEL LOGGED")
print("================================")