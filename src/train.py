from pathlib import Path

import numpy as np
import tensorflow as tf
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


# ============================================================
# PATHS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT_DIR / "models" / "emotion_model_weighted.keras"


# ============================================================
# MLFLOW
# ============================================================

mlflow.set_experiment("MindMirror Emotion Classification")


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
    y_test
) = split_data(X, y)


# ============================================================
# CLASS WEIGHTS
# ============================================================

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


# ============================================================
# ENCODE LABELS
# ============================================================

(
    y_train,
    y_val,
    y_test
) = encode_labels(
    y_train,
    y_val,
    y_test
)


# ============================================================
# BUILD CNN
# ============================================================

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


# ============================================================
# COMPILE
# ============================================================

learning_rate = 0.00025

model.compile(
    optimizer=Adam(learning_rate=learning_rate),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


# ============================================================
# MODEL SUMMARY
# ============================================================

model.summary()


# ============================================================
# CALLBACKS
# ============================================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=7,
    restore_best_weights=True
)

model_checkpoint = ModelCheckpoint(
    MODEL_PATH,
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


# ============================================================
# MLFLOW RUN
# ============================================================

with mlflow.start_run(run_name="Class Weighted CNN"):

    # --------------------------------------------------------
    # LOG PARAMETERS
    # --------------------------------------------------------

    mlflow.log_params({
        "model_type": "CNN",
        "training_type": "class_weighted",
        "epochs": 50,
        "batch_size": 64,
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

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    print("\n==========================================")
    print("STARTING CLASS-WEIGHTED CNN TRAINING")
    print("==========================================")

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=64,
        class_weight=class_weights,
        callbacks=[
            early_stopping,
            model_checkpoint,
            reduce_lr
        ],
        verbose=1
    )

    # --------------------------------------------------------
    # LOG TRAINING METRICS
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # FINAL TEST
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # LOG TEST METRICS
    # --------------------------------------------------------

    mlflow.log_metrics({
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
    })

    # --------------------------------------------------------
    # LOG MODEL
    # --------------------------------------------------------

    mlflow.tensorflow.log_model(
        model,
        name="emotion_model"
    )


# ============================================================
# TRAINING COMPLETE
# ============================================================

print("\n==========================================")
print("TRAINING COMPLETE")
print("==========================================")

print("\nBest model saved to:")
print(MODEL_PATH)

print("\nMLflow experiment:")
print("MindMirror Emotion Classification")