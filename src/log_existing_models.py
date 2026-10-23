from pathlib import Path

import mlflow
import mlflow.tensorflow
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)
from tensorflow.keras.models import load_model

from mlflow_config import configure_mlflow
from preprocessing import (
    DATA_PATH,
    encode_labels,
    load_data,
    preprocess_images,
    split_data,
)


ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS = {
    "Baseline CNN": ROOT_DIR / "models" / "emotion_model.keras",
    "Augmented CNN": ROOT_DIR / "models" / "emotion_model_augmented.keras",
    "Class Weighted CNN": ROOT_DIR / "models" / "emotion_model_weighted.keras",
}


def main() -> None:
    available_models = [
        (model_name, model_path)
        for model_name, model_path in MODELS.items()
        if model_path.is_file()
    ]

    if not available_models:
        print("No model files were found; nothing was evaluated or logged.")
        return

    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"FER2013 dataset not found at {DATA_PATH}. "
            "Place the dataset there before evaluating existing models."
        )

    X, y = load_data()
    X = preprocess_images(X)
    _, _, X_test, y_train, y_val, y_test = split_data(X, y)
    y_test_labels = y_test
    _, _, y_test = encode_labels(
        y_train,
        y_val,
        y_test,
    )

    configure_mlflow()

    for model_name, model_path in available_models:
        print(f"\nEvaluating {model_name} from {model_path}")
        model = load_model(model_path)

        predictions = model.predict(
            X_test,
            batch_size=64,
            verbose=1,
        )
        y_pred = np.argmax(predictions, axis=1)
        test_loss, _ = model.evaluate(
            X_test,
            y_test,
            batch_size=64,
            verbose=0,
        )
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test_labels,
            y_pred,
            average="macro",
            zero_division=0,
        )
        test_accuracy = accuracy_score(y_test_labels, y_pred)

        with mlflow.start_run(
            run_name=f"{model_name} - Existing Model Evaluation",
            tags={
                "run_type": "existing_model_evaluation",
                "model_file": model_path.name,
            },
        ) as run:
            mlflow.log_params({
                "model_file": model_path.name,
                "model_type": model_name,
                "total_parameters": model.count_params(),
                "evaluation_dataset": "FER2013",
                "evaluation_samples": len(y_test_labels),
                "test_split_fraction": 0.10,
                "random_state": 42,
            })
            mlflow.log_metrics({
                "test_accuracy": float(test_accuracy),
                "test_loss": float(test_loss),
                "test_macro_precision": float(precision),
                "test_macro_recall": float(recall),
                "test_macro_f1": float(f1),
            })
            mlflow.tensorflow.log_model(
                model,
                name="emotion_model",
            )

            print(
                f"Logged {model_name}: "
                f"accuracy={test_accuracy:.4f}, loss={test_loss:.4f}, "
                f"macro_f1={f1:.4f}; run_id={run.info.run_id}"
            )


if __name__ == "__main__":
    main()
