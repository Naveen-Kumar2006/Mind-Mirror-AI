from pathlib import Path
import io

import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from tensorflow.keras.models import load_model


# ============================================================
# PATHS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent

MODEL_PATH = ROOT_DIR / "models" / "emotion_model.keras"


# ============================================================
# EMOTION LABELS
# ============================================================

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
# LOAD MODEL
# ============================================================

print("Loading emotion recognition model...")

model = load_model(MODEL_PATH)

print("Model loaded successfully!")


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="MindMirror AI",
    description="Facial Emotion Recognition API using CNN",
    version="1.0.0",
)


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():
    return {
        "message": "MindMirror AI Emotion Recognition API",
        "status": "running",
        "model": "Baseline CNN",
    }


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # --------------------------------------------------------
    # Check file type
    # --------------------------------------------------------

    if file.content_type not in [
        "image/jpeg",
        "image/png",
        "image/jpg",
    ]:
        raise HTTPException(
            status_code=400,
            detail="Please upload a JPG or PNG image."
        )

    try:

        # ----------------------------------------------------
        # Read uploaded image
        # ----------------------------------------------------

        image_bytes = await file.read()

        image = Image.open(
            io.BytesIO(image_bytes)
        )

        # ----------------------------------------------------
        # Preprocess image
        # ----------------------------------------------------

        image = image.convert("L")

        image = image.resize(
            (48, 48)
        )

        image_array = np.array(
            image,
            dtype="float32"
        )

        # Normalize
        image_array = image_array / 255.0

        # Add channel dimension
        image_array = image_array.reshape(
            48,
            48,
            1
        )

        # Add batch dimension
        image_array = np.expand_dims(
            image_array,
            axis=0
        )

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        predictions = model.predict(
            image_array,
            verbose=0
        )

        probabilities = predictions[0]

        predicted_class = np.argmax(
            probabilities
        )

        emotion = EMOTION_NAMES[
            predicted_class
        ]

        confidence = float(
            probabilities[predicted_class]
        )

        # ----------------------------------------------------
        # Return result
        # ----------------------------------------------------

        return {
            "emotion": emotion,
            "confidence": round(
                confidence,
                4
            ),
            "probabilities": {
                EMOTION_NAMES[i]: round(
                    float(probabilities[i]),
                    4
                )
                for i in range(7)
            }
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "Baseline CNN"
    }