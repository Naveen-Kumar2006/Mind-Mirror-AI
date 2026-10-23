from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import load_model

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

from preprocessing import (
    load_data,
    preprocess_images,
    split_data,
)


# =========================================================
# CONFIGURATION
# =========================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT_DIR / "models" / "emotion_model_weighted.keras"

REPORT_DIR = ROOT_DIR / "reports"

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

EMOTION_NAMES = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral",
]


# =========================================================
# LOAD AND PREPROCESS DATA
# =========================================================

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

print("\nDataset loaded successfully!")
print(f"Test samples: {len(y_test)}")


# =========================================================
# LOAD TRAINED MODEL
# =========================================================

print("\nLoading trained model...")

model = load_model(MODEL_PATH)

print("Model loaded successfully!")


# =========================================================
# MAKE PREDICTIONS
# =========================================================

print("\nMaking predictions...")

probabilities = model.predict(
    X_test,
    batch_size=64,
    verbose=1,
)

y_pred = np.argmax(
    probabilities,
    axis=1,
)


# =========================================================
# CALCULATE METRICS
# =========================================================

accuracy = accuracy_score(
    y_test,
    y_pred,
)

precision = precision_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0,
)

recall = recall_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0,
)

f1 = f1_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0,
)

report = classification_report(
    y_test,
    y_pred,
    labels=list(range(len(EMOTION_NAMES))),
    target_names=EMOTION_NAMES,
    digits=4,
    zero_division=0,
)

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=list(range(len(EMOTION_NAMES))),
)


# =========================================================
# DISPLAY RESULTS
# =========================================================

print("\n" + "=" * 50)
print("MODEL EVALUATION")
print("=" * 50)

print(f"Test Accuracy : {accuracy:.4f} ({accuracy:.2%})")
print(f"Macro Precision: {precision:.4f}")
print(f"Macro Recall   : {recall:.4f}")
print(f"Macro F1-score : {f1:.4f}")

print("\n" + "=" * 50)
print("CLASSIFICATION REPORT")
print("=" * 50)

print(report)


# =========================================================
# SAVE CLASSIFICATION REPORT
# =========================================================

report_path = REPORT_DIR / "classification_report.txt"

with open(
    report_path,
    "w",
    encoding="utf-8",
) as file:

    file.write("MINDMIRROR AI - MODEL EVALUATION\n")
    file.write("=" * 50 + "\n\n")

    file.write(f"Test samples: {len(y_test)}\n")
    file.write(f"Accuracy: {accuracy:.4f} ({accuracy:.2%})\n")
    file.write(f"Macro Precision: {precision:.4f}\n")
    file.write(f"Macro Recall: {recall:.4f}\n")
    file.write(f"Macro F1-score: {f1:.4f}\n\n")

    file.write("CLASSIFICATION REPORT\n")
    file.write("=" * 50 + "\n")
    file.write(report)

print(f"\nClassification report saved to:\n{report_path}")


# =========================================================
# PLOT CONFUSION MATRIX
# =========================================================

plt.figure(figsize=(10, 8))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=EMOTION_NAMES,
    yticklabels=EMOTION_NAMES,
    cbar=True,
)

plt.title("MindMirror AI - Confusion Matrix")
plt.xlabel("Predicted Expression")
plt.ylabel("Actual Expression")

plt.tight_layout()

cm_path = REPORT_DIR / "confusion_matrix.png"

plt.savefig(
    cm_path,
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print(f"Confusion matrix saved to:\n{cm_path}")


# =========================================================
# SAVE SUMMARY METRICS
# =========================================================

metrics_path = REPORT_DIR / "metrics.txt"

with open(
    metrics_path,
    "w",
    encoding="utf-8",
) as file:

    file.write(f"Test Accuracy: {accuracy:.4f}\n")
    file.write(f"Macro Precision: {precision:.4f}\n")
    file.write(f"Macro Recall: {recall:.4f}\n")
    file.write(f"Macro F1-score: {f1:.4f}\n")
    file.write(f"Test Samples: {len(y_test)}\n")

print(f"Summary metrics saved to:\n{metrics_path}")

print("\nEvaluation completed successfully!")
