from pathlib import Path

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    BatchNormalization,
    Flatten,
    Dense,
    Dropout
)
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)

from preprocessing import (
    load_data,
    preprocess_images,
    split_data,
    encode_labels
)


# ==========================================
# Paths
# ==========================================

ROOT_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = ROOT_DIR / "models"

MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "emotion_model.keras"


# ==========================================
# 1. Load and Preprocess Dataset
# ==========================================

X, y = load_data()

X = preprocess_images(X)


# ==========================================
# 2. Split Dataset
# ==========================================

(
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test
) = split_data(X, y)


# ==========================================
# 3. Encode Labels
# ==========================================

(
    y_train,
    y_val,
    y_test
) = encode_labels(
    y_train,
    y_val,
    y_test
)


# ==========================================
# 4. Build CNN Model
# ==========================================

model = Sequential()


# ------------------------------------------
# Block 1
# ------------------------------------------

model.add(
    Conv2D(
        32,
        (3, 3),
        activation="relu",
        input_shape=(48, 48, 1)
    )
)

model.add(BatchNormalization())

model.add(
    MaxPooling2D((2, 2))
)


# ------------------------------------------
# Block 2
# ------------------------------------------

model.add(
    Conv2D(
        64,
        (3, 3),
        activation="relu"
    )
)

model.add(BatchNormalization())

model.add(
    MaxPooling2D((2, 2))
)


# ------------------------------------------
# Block 3
# ------------------------------------------

model.add(
    Conv2D(
        128,
        (3, 3),
        activation="relu"
    )
)

model.add(BatchNormalization())

model.add(
    MaxPooling2D((2, 2))
)


# ------------------------------------------
# Block 4
# ------------------------------------------

model.add(
    Conv2D(
        256,
        (3, 3),
        activation="relu"
    )
)

model.add(BatchNormalization())

model.add(
    MaxPooling2D((2, 2))
)


# ==========================================
# 5. Fully Connected Layers
# ==========================================

model.add(Flatten())

model.add(
    Dense(
        256,
        activation="relu"
    )
)

model.add(
    Dropout(0.5)
)

model.add(
    Dense(
        128,
        activation="relu"
    )
)

model.add(
    Dropout(0.3)
)


# ==========================================
# 6. Output Layer
# ==========================================

model.add(
    Dense(
        7,
        activation="softmax"
    )
)


# ==========================================
# 7. Model Summary
# ==========================================

model.summary()


# ==========================================
# 8. Compile Model
# ==========================================

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


# ==========================================
# 9. Callbacks
# ==========================================

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=7,
    restore_best_weights=True,
    verbose=1
)


checkpoint = ModelCheckpoint(
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


# ==========================================
# 10. Train Model
# ==========================================

history = model.fit(
    X_train,
    y_train,

    validation_data=(
        X_val,
        y_val
    ),

    epochs=50,

    batch_size=64,

    callbacks=[
        early_stop,
        checkpoint,
        reduce_lr
    ]
)


# ==========================================
# 11. Load Best Model
# ==========================================

best_model = load_model(
    MODEL_PATH
)


# ==========================================
# 12. Final Evaluation
# ==========================================

test_loss, test_accuracy = best_model.evaluate(
    X_test,
    y_test,
    verbose=1
)


print("\n==========================================")
print("FINAL TEST RESULTS")
print("==========================================")

print(
    f"Test Loss     : {test_loss:.4f}"
)

print(
    f"Test Accuracy : {test_accuracy:.4f}"
)