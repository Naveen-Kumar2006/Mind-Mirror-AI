# 🧠 MindMirror AI

### Emotion Recognition using Deep Learning, MLflow & FastAPI

MindMirror AI is a deep learning-based facial emotion recognition system that classifies human facial expressions into **7 different emotion categories** using a Convolutional Neural Network (CNN).

The project uses the **FER2013 dataset** and follows an end-to-end machine learning workflow including data preprocessing, CNN training, class-weighted learning, experiment tracking with MLflow, model evaluation, and API-based inference using FastAPI.

---

## 🎯 Project Objective

The goal of MindMirror AI is to build a facial emotion recognition system capable of identifying emotions from facial images.

The model classifies images into:

- 😠 Angry
- 🤢 Disgust
- 😨 Fear
- 😊 Happy
- 😐 Neutral
- 😢 Sad
- 😲 Surprise

---

## 🚀 Key Features

- Facial emotion classification using CNN
- FER2013 dataset preprocessing
- Grayscale 48×48 image processing
- Image normalization
- Stratified dataset splitting
- Class-weighted CNN training
- Overfitting monitoring
- Model checkpointing
- Early stopping
- Learning-rate reduction
- MLflow experiment tracking
- Model evaluation
- FastAPI prediction API
- Modular project structure

---

## 🧠 Machine Learning Pipeline

```text
                FER2013 Dataset
                       │
                       ▼
              Data Preprocessing
                       │
                       ▼
             Image Normalization
                  48 × 48
                       │
                       ▼
              Train / Validation
                    / Test
                       │
                       ▼
             Class Weight Calculation
                       │
                       ▼
                CNN Training
                       │
                       ▼
          ┌─────────────────────────┐
          │      MLflow Tracking    │
          │                         │
          │ Parameters              │
          │ Metrics                 │
          │ Training Results        │
          │ Model Artifacts         │
          └─────────────────────────┘
                       │
                       ▼
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
