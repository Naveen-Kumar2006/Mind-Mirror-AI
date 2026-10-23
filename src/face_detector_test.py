from pathlib import Path
from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT_DIR / "models" / "yolov11n-face.pt"

print("Loading YOLO face detector...")

model = YOLO(MODEL_PATH)

print("Model loaded successfully!")
print(model.names)