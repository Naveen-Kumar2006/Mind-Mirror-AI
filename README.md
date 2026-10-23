# 🧠 MindMirror AI — Facial Expression Recognition

An end-to-end Deep Learning application that predicts facial expressions from uploaded images using a Convolutional Neural Network (CNN). The application provides a web interface for image uploads, predictions, model confidence, and emotion probability visualization.

🌐 **Live Demo:** [Try MindMirror AI](https://mind-mirror-ai-havrwb3wgxf3xqbqmc6yan.streamlit.app/)

🚀 **Backend API:** [FastAPI Backend](https://mind-mirror-ai.onrender.com)

---

## 📌 Project Overview

MindMirror AI demonstrates the development and deployment of a computer vision application, from model training and evaluation to backend API development and frontend integration.

Users can upload a facial image and receive a predicted facial expression along with the model's confidence score and probability distribution.

> **Note:** The model predicts facial-expression categories. Its predictions do not establish a person's actual emotional state.

## ✨ Features

* 🧠 **CNN-Based Prediction:** Predicts facial-expression categories using a trained deep learning model.
* 🖼️ **Image Upload:** Supports JPG, JPEG, and PNG images.
* ⚡ **REST API:** Processes prediction requests through FastAPI.
* 📊 **Confidence Score:** Displays the model's confidence for its prediction.
* 📈 **Probability Visualization:** Uses an interactive Altair chart to display emotion probabilities.
* 🌐 **Live Deployment:** Streamlit frontend and Render-hosted backend.
* 🛡️ **Error Handling:** Handles API connection failures, request timeouts, and invalid responses.

## 🛠️ Tech Stack

| Category             | Technologies                      |
| -------------------- | --------------------------------- |
| Programming Language | Python                            |
| Deep Learning        | TensorFlow, Keras                 |
| Computer Vision      | Pillow, image preprocessing       |
| Backend              | FastAPI, Uvicorn                  |
| Frontend             | Streamlit                         |
| Data Visualization   | Pandas, Altair                    |
| HTTP Requests        | Requests                          |
| Deployment           | Streamlit Community Cloud, Render |
| Version Control      | Git, GitHub                       |

## 🏗️ System Architecture

```text
             User
              |
              v
    +---------------------+
    | Streamlit Frontend  |
    | Image Upload        |
    +---------------------+
              |
              | HTTP POST /predict
              v
    +---------------------+
    | FastAPI Backend     |
    | Image Processing    |
    +---------------------+
              |
              v
    +---------------------+
    | Trained CNN Model   |
    | Expression Predict. |
    +---------------------+
              |
              v
    +---------------------+
    | Prediction Results  |
    | Confidence          |
    | Probabilities       |
    +---------------------+
              |
              v
    +---------------------+
    | Streamlit Dashboard |
    | Result + Chart      |
    +---------------------+
```

## 🔄 How It Works

1. The user uploads a facial image through the Streamlit interface.
2. The frontend sends the image to the FastAPI `/predict` endpoint.
3. The backend preprocesses the image to match the model's expected input format.
4. The trained CNN generates prediction scores.
5. The backend returns the predicted expression, confidence score, and probability distribution.
6. The frontend displays the prediction and an interactive probability chart.

## 🚀 Live Application

**Frontend:** [Open MindMirror AI](https://mind-mirror-ai-havrwb3wgxf3xqbqmc6yan.streamlit.app/)

**Backend Health Check:** [Check API Status](https://mind-mirror-ai.onrender.com/health)

To try the application:

1. Open the live frontend.
2. Upload a facial image.
3. Click **Analyze Emotion**.
4. View the predicted expression and confidence score.
5. Explore the emotion probability chart.

*Note: The free backend hosting plan may introduce a delay when the service wakes up after inactivity.*

## 📁 Project Structure

```text
Mind-Mirror-AI/
│
├── app.py
├── streamlit_app.py
├── requirements.txt
├── README.md
│
├── models/
│   └── [Trained model files]
│
├── src/
│   ├── train.py
│   ├── evaluate.py
│   ├── preprocessing.py
│   └── [Additional project modules]
│
└── [Additional configuration and project files]
```

*The structure above is illustrative. Update it to match the files actually committed to your repository.*

## 📊 Model Evaluation

The project includes model training and evaluation modules.

Important evaluation metrics for this classification task include:

* **Accuracy:** Overall proportion of correct predictions.
* **Precision:** Proportion of positive predictions that are correct for a given class.
* **Recall:** Proportion of actual class instances correctly identified.
* **F1-score:** Harmonic mean of precision and recall.
* **Confusion Matrix:** Shows correct and incorrect predictions across classes.

Model performance should be reported using results from the final evaluation on held-out test data.

## 🔧 Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/Naveen-Kumar2006/Mind-Mirror-AI.git
cd Mind-Mirror-AI
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the FastAPI backend

Use the actual backend module and application object defined in your repository. For example, if your FastAPI application is exposed as `app` in `app.py`:

```bash
uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

### 5. Start the Streamlit frontend

Open a second terminal in the project directory.

Set the backend URL for local development:

```powershell
$env:API_URL = "http://127.0.0.1:8000"
```

Start Streamlit:

```bash
streamlit run streamlit_app.py
```

If your backend module has a different filename, adjust the Uvicorn command accordingly.

## 🎯 Learning Outcomes

This project demonstrates practical experience with:

* Building and using a CNN for image classification.
* Preprocessing images for deep learning inference.
* Developing REST APIs with FastAPI.
* Integrating a Python frontend with a machine learning backend.
* Visualizing classification probabilities.
* Deploying an end-to-end ML application.
* Managing source code using Git and GitHub.

## 🔮 Future Improvements

* Improve model generalization and reduce overfitting.
* Evaluate performance across individual expression classes.
* Add a confusion matrix and per-class classification report.
* Improve robustness to lighting, pose, and image-quality variations.
* Add automated tests for preprocessing and API predictions.
* Optimize inference latency and deployment reliability.

## 👨‍💻 Author

**Naveen Kumar T**

B.Tech Artificial Intelligence and Data Science
Aspiring Machine Learning Engineer

GitHub: [Naveen-Kumar2006](https://github.com/Naveen-Kumar2006)

---

⭐ If you find this project interesting, consider starring the repository!
