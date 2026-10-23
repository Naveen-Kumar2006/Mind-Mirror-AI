from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model

ROOT_DIR = Path(__file__).resolve().parent.parent

FACE_MODEL_PATH = ROOT_DIR / "models" / "yolov11n-face.pt"
EMOTION_MODEL_PATH = ROOT_DIR / "models" / "emotion_model.keras"

print("Loading YOLO face detector...")
face_model = YOLO(FACE_MODEL_PATH)

print("Loading MindMirror emotion model...")
emotion_model = load_model(EMOTION_MODEL_PATH)

print("Both models loaded successfully!")



EMOTION_NAMES = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral",
]

cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Could not open laptop camera.")
    exit()

print("Webcam started.")
print("Press Q to quit.")

while True:

    ret, frame = cap.read()

    if not ret:
        print("Could not read webcam frame.")
        break

    # YOLO face detection
    results = face_model.predict(
        source=frame,
        conf=0.5,
        verbose=False
    )

    for result in results:

        for box in result.boxes:

            # Face coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Make sure coordinates stay inside image
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)

            # Crop face
            face = frame[y1:y2, x1:x2]

            if face.size == 0:
                continue


            face_gray = cv2.cvtColor(
                face,
                cv2.COLOR_BGR2GRAY
            )

            face_gray = cv2.resize(
                face_gray,
                (48, 48)
            )

            face_array = face_gray.astype("float32") / 255.0

            face_array = face_array.reshape(
                1, 48, 48, 1
            )


            predictions = emotion_model.predict(
                face_array,
                verbose=0
            )

            probabilities = predictions[0]

            predicted_class = np.argmax(probabilities)

            emotion = EMOTION_NAMES[predicted_class]

            confidence = float(
                probabilities[predicted_class]
            )


            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            label = f"{emotion} {confidence:.2f}"

            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

    # Show webcam
    cv2.imshow(
        "MindMirror AI - Emotion Detection",
        frame
    )

    # Quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()