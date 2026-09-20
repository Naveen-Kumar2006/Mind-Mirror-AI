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


# ==========================================
# Paths
# ==========================================

ROOT_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT_DIR / "models" / "emotion_model_weighted.keras"
# ==========================================
# 1. Load Dataset
# ==========================================

X, y = load_data()


# ==========================================
# 2. Preprocess Images
# ==========================================

X = preprocess_images(X)


# ==========================================
# 3. Split Dataset
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
# 4. Load Trained Model
# ==========================================

print("\nLoading trained model...")

model = load_model(MODEL_PATH)

print("Model loaded successfully!")


# ==========================================
# 5. Make Predictions
# ==========================================

print("\nMaking predictions...")

predictions = model.predict(
    X_test,
    batch_size=64,
    verbose=1
)


# ==========================================
# 6. Convert Predictions to Class Labels
# ==========================================

y_pred = np.argmax(
    predictions,
    axis=1
)


# ==========================================
# 7. Accuracy
# ==========================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n==========================================")
print("MODEL EVALUATION")
print("==========================================")

print(f"\nAccuracy : {accuracy:.4f}")
print(f"Accuracy : {accuracy * 100:.2f}%")


# ==========================================
# 8. Classification Report
# ==========================================

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


# ==========================================
# 9. Confusion Matrix
# ==========================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\n==========================================")
print("CONFUSION MATRIX")
print("==========================================")

print(cm)