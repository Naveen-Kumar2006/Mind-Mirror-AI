from pathlib import Path

import numpy as np
import tensorflow as tf
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

model.compile(
    optimizer=Adam(learning_rate=0.00025),
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
# TRAIN
# ============================================================

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


# ============================================================
# FINAL TEST
# ============================================================

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

print("\n==========================================")
print("TRAINING COMPLETE")
print("==========================================")

print("\nBest model saved to:")
print(MODEL_PATH)