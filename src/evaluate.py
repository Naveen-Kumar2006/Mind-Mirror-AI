from pathlib import Path

import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from preprocessing import (
    load_data,
    preprocess_images,
    split_data
)

ROOT_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT_DIR / "models" / "emotion_model_weighted.keras"

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

print("\nLoading trained model...")

model = load_model(MODEL_PATH)

print("Model loaded successfully!")

print("\nMaking predictions...")

predictions = model.predict(
    X_test,
    batch_size=64,
    verbose=1
)

y_pred = np.argmax(
    predictions,
    axis=1
)

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n==========================================")
print("MODEL EVALUATION")
print("==========================================")

print(f"\nAccuracy : {accuracy:.4f}")
print(f"Accuracy : {accuracy * 100:.2f}%")

emotion_names = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral"
]

print("\n==========================================")
print("CLASSIFICATION REPORT")
print("==========================================")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=emotion_names,
        digits=4
    )
)


cm = confusion_matrix(
    y_test,
    y_pred
)

print("\n==========================================")
print("CONFUSION MATRIX")
print("==========================================")

print(cm)