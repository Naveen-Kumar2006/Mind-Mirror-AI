from pathlib import Path

import numpy as np
import mlflow
import mlflow.tensorflow

from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    BatchNormalization,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
)
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight

from preprocessing import (
    load_data,
    preprocess_images,
    split_data,
    encode_labels,
)
from mlflow_config import configure_mlflow, EXPERIMENT_NAME


ROOT_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT_DIR / "models" / "emotion_model_weighted.keras"

configure_mlflow()


print("\nLoading dataset...")

X, y = load_data()

X = preprocess_images(X)

(
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test
) = split_data(X, y)


classes = np.unique(y_train)

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)

class_weights = dict(
    zip(classes, class_weights)
)

print("\n==========================================")
print("CLASS WEIGHTS")
print("==========================================")

emotion_names = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral"
]

for class_id, weight in class_weights.items():
    print(
        f"{emotion_names[class_id]:10s} : {weight:.4f}"
    )


(
    y_train,
    y_val,
    y_test
) = encode_labels(
    y_train,
    y_val,
    y_test
)

model = Sequential(
    [
        Input(shape=(48, 48, 1)),

        # Block 1
        Conv2D(
            32,
            (3, 3),
            activation="relu",
            padding="same"
        ),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        # Block 2
        Conv2D(
            64,
            (3, 3),
            activation="relu",
            padding="same"
        ),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        # Block 3
        Conv2D(
            128,
            (3, 3),
            activation="relu",
            padding="same"
        ),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        # Block 4
        Conv2D(
            256,
            (3, 3),
            activation="relu",
            padding="same"
        ),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        # Classifier
        Flatten(),

        Dense(256, activation="relu"),
        Dropout(0.5),

        Dense(128, activation="relu"),
        Dropout(0.3),

        Dense(7, activation="softmax"),
    ]
)


learning_rate = 0.00025
max_epochs = 50
batch_size = 64

model.compile(
    optimizer=Adam(learning_rate=learning_rate),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)



model.summary()


early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=7,
    restore_best_weights=True
)

model_checkpoint = ModelCheckpoint(
    str(MODEL_PATH),
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1
)


with mlflow.start_run(
    run_name="Class Weighted CNN",
    tags={
        "run_type": "training",
        "model_file": MODEL_PATH.name,
    },
) as run:


    mlflow.log_params({
        "model_type": "CNN",
        "training_type": "class_weighted",
        "epochs": max_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "optimizer": "Adam",
        "loss": "categorical_crossentropy",
        "num_classes": 7,
        "input_shape": "48x48x1",
        "conv_blocks": 4,
        "dropout_1": 0.5,
        "dropout_2": 0.3,
    })

    # Log class weights
    for class_id, weight in class_weights.items():
        mlflow.log_param(
            f"class_weight_{emotion_names[class_id]}",
            float(weight)
        )

    print("\n==========================================")
    print("STARTING CLASS-WEIGHTED CNN TRAINING")
    print("==========================================")

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=max_epochs,
        batch_size=batch_size,
        class_weight=class_weights,
        callbacks=[
            early_stopping,
            model_checkpoint,
            reduce_lr
        ],
        verbose=1
    )

    for epoch in range(len(history.history["loss"])):
        epoch_metrics = {
            f"train_{metric_name}": float(values[epoch])
            for metric_name, values in history.history.items()
            if not metric_name.startswith("val_")
        }
        epoch_metrics.update({
            f"validation_{metric_name[4:]}": float(values[epoch])
            for metric_name, values in history.history.items()
            if metric_name.startswith("val_")
        })
        mlflow.log_metrics(epoch_metrics, step=epoch)

    best_val_accuracy = max(
        history.history["val_accuracy"]
    )

    best_val_loss = min(
        history.history["val_loss"]
    )

    final_train_accuracy = (
        history.history["accuracy"][-1]
    )

    final_train_loss = (
        history.history["loss"][-1]
    )

    mlflow.log_metrics({
        "best_val_accuracy": float(best_val_accuracy),
        "best_val_loss": float(best_val_loss),
        "final_train_accuracy": float(final_train_accuracy),
        "final_train_loss": float(final_train_loss),
    })

    print("\n==========================================")
    print("FINAL TEST RESULTS")
    print("==========================================")

    test_loss, test_accuracy = model.evaluate(
        X_test,
        y_test,
        batch_size=64,
        verbose=1
    )

    print(f"\nTest Loss     : {test_loss:.4f}")
    print(f"Test Accuracy : {test_accuracy:.4f}")
    print(f"Test Accuracy : {test_accuracy * 100:.2f}%")

    mlflow.log_metrics({
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
    })

    mlflow.tensorflow.log_model(
        model,
        name="emotion_model"
    )
    print(f"MLflow run ID: {run.info.run_id}")


print("\n==========================================")
print("TRAINING COMPLETE")
print("==========================================")

print("\nBest model saved to:")
print(MODEL_PATH)

print("\nMLflow experiment:")
print(EXPERIMENT_NAME)