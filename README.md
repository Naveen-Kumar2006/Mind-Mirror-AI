MindMirror AI — Project Documentation
1. Project Overview
MindMirror AI is a CNN-based facial emotion recognition system trained on the FER2013 dataset. The system classifies facial expressions into seven emotion categories: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral.
The project implements an end-to-end machine learning workflow, including image preprocessing, CNN model training, model evaluation, MLflow experiment tracking, and FastAPI-based model serving.
Users can upload a facial image through the FastAPI REST API, and the system returns the predicted emotion along with its confidence score.
2. Project Objectives
The main objectives of MindMirror AI are:
Build a facial emotion classification system using deep learning.
Train a CNN using the FER2013 dataset.
Perform image preprocessing and normalization.
Evaluate model performance using classification metrics.
Track machine learning experiments using MLflow.
Compare different CNN training approaches.
Deploy the trained model through a FastAPI REST API.
Provide real-time emotion predictions from uploaded images.
3. Emotion Classes
Label	Emotion
0	Angry
1	Disgust
2	Fear
3	Happy
4	Sad
5	Surprise
6	Neutral
4. Dataset
The project uses the FER2013 (Facial Expression Recognition 2013) dataset.
The images are grayscale facial images with a resolution of 48 × 48 pixels.
The dataset contains 35,887 images.
Dataset Split
Dataset	Images
Training	28,709
Validation	3,589
Testing	3,589
The dataset is split using stratified sampling to maintain the distribution of emotion classes.
5. Image Preprocessing
The raw FER2013 pixel data is converted into image arrays before being provided to the CNN.
The preprocessing pipeline consists of:
Reading pixel values from the dataset.
Converting the pixel strings into numerical arrays.
Reshaping each image into `48 × 48`.
Normalizing pixel values using `/255.0`.
Adding the grayscale channel dimension.
Encoding the emotion labels into seven classes.
The final model input shape is:
`(48, 48, 1)`
6. CNN Architecture
The baseline model uses multiple convolutional blocks followed by fully connected layers.
```text
Input: 48 × 48 × 1
        ↓
Conv2D - 32 filters
        ↓
Batch Normalization
        ↓
MaxPooling
        ↓
Conv2D - 64 filters
        ↓
Batch Normalization
        ↓
MaxPooling
        ↓
Conv2D - 128 filters
        ↓
Batch Normalization
        ↓
MaxPooling
        ↓
Conv2D - 256 filters
        ↓
Batch Normalization
        ↓
MaxPooling
        ↓
Flatten
        ↓
Dense - 256
        ↓
Dropout
        ↓
Dense - 128
        ↓
Dropout
        ↓
Dense - 7
        ↓
Softmax
```
Training Configuration
Optimizer: Adam
Loss Function: Categorical Crossentropy
Metric: Accuracy
Batch Size: 64
Maximum Epochs: 50
Early Stopping
Model Checkpoint
ReduceLROnPlateau
7. Model Evaluation
The baseline CNN achieved a test accuracy of:
59.68%
The model was evaluated using:
Accuracy
Precision
Recall
F1-score
Classification report
Confusion matrix
Class-wise F1 Score
Emotion	F1 Score
Angry	50.16%
Disgust	48.19%
Fear	40.90%
Happy	80.52%
Sad	48.01%
Surprise	71.99%
Neutral	55.12%
The evaluation artifacts are tracked using MLflow.
8. MLflow Experiment Tracking
MLflow is used to track the machine learning experiments and evaluation results.
The project contains experiments for:
Baseline CNN
Augmented CNN
Class Weighted CNN
MLflow is used to track:
Training parameters
Training metrics
Validation metrics
Test metrics
Model artifacts
Classification reports
Confusion matrices
This provides a centralized way to compare different model experiments and maintain experiment history.
9. FastAPI Model Serving
The trained Baseline CNN is served through a FastAPI REST API.
The API provides an endpoint for facial emotion prediction.
Prediction Endpoint
```text
POST /predict
```
The endpoint accepts an image file.
Prediction Workflow
```text
Image Upload
     ↓
FastAPI
     ↓
Read Image
     ↓
Convert to Grayscale
     ↓
Resize to 48 × 48
     ↓
Normalize Pixel Values
     ↓
CNN Model
     ↓
Emotion Probabilities
     ↓
Select Highest Probability
     ↓
Return Prediction
```
Example Response
```json
{
    "emotion": "Happy",
    "confidence": 0.8234,
    "probabilities": {
        "Angry": 0.0213,
        "Disgust": 0.0012,
        "Fear": 0.0345,
        "Happy": 0.8234,
        "Sad": 0.0456,
        "Surprise": 0.0521,
        "Neutral": 0.0219
    }
}
```
10. API Documentation
FastAPI provides an interactive Swagger UI for testing the API.
Start the application using:
```bash
uvicorn api:app --reload
```
The Swagger documentation is available at:
`http://127.0.0.1:8000/docs`
Users can upload a facial image through the `/predict` endpoint and directly test the trained model.
11. Technology Stack
Programming Language
Python
Machine Learning & Deep Learning
TensorFlow
Keras
Scikit-learn
NumPy
Pandas
Computer Vision
Pillow
OpenCV
Experiment Tracking
MLflow
API Development
FastAPI
Uvicorn
Pydantic
Development & Version Control
VS Code
Git
GitHub
12. Project Structure
```text
MindMirror AI/
│
├── data/
│   └── fer2013.csv
│
├── models/
│   ├── emotion_model.keras
│   ├── emotion_model_augmented.keras
│   └── emotion_model_weighted.keras
│
├── notebooks/
│   └── emotion_training.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── log_existing_models.py
│   └── log_evaluation_artifacts.py
│
├── api.py
├── requirements.txt
├── README.md
├── pyproject.toml
├── uv.lock
└── .gitignore
```
13. Installation
Clone the repository:
```bash
git clone https://github.com/Naveen-Kumar2006/Mind-Mirror-AI.git
```
Navigate to the project directory:
```bash
cd Mind-Mirror-AI
```
Create a virtual environment:
```bash
python -m venv .venv
```
Activate the virtual environment on Windows:
```powershell
.venv\Scripts\Activate.ps1
```
Install the required dependencies:
```bash
pip install -r requirements.txt
```
14. Running the Application
Start the FastAPI server:
```bash
uvicorn api:app --reload
```
Open the Swagger UI:
`http://127.0.0.1:8000/docs`
Select:
`POST /predict`
Upload a facial image and execute the request.
The API returns the predicted emotion and confidence score.
15. End-to-End Architecture
```text
                 FER2013 Dataset
                        ↓
                Data Preprocessing
                        ↓
                  CNN Training
                        ↓
                Model Evaluation
                       │
                       ▼
              Trained CNN Model
                       │
                       ▼
                 FastAPI API
                       │
                       ▼
              Emotion Prediction
