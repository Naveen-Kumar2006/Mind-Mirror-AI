import os

import altair as alt
import pandas as pd
import requests
import streamlit as st
from PIL import Image


API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8000"
).rstrip("/")

EMOTION_NAMES = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral",
]

st.set_page_config(
    page_title="MindMirror AI",
    page_icon="🧠",
    layout="centered",
)

st.title("🧠 MindMirror AI")
st.write("Upload a facial image to predict its expression using a CNN model.")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded image",
            use_container_width=True,
        )

        if st.button("Analyze Emotion", type="primary"):
            with st.spinner("Analyzing facial expression..."):
                try:
                    uploaded_file.seek(0)

                    response = requests.post(
                        f"{API_URL}/predict",
                        files={
                            "file": (
                                uploaded_file.name,
                                uploaded_file.getvalue(),
                                uploaded_file.type or "image/jpeg",
                            )
                        },
                        timeout=120,
                    )

                    response.raise_for_status()
                    result = response.json()

                    emotion = result["emotion"]
                    confidence = float(result["confidence"])
                    probabilities = result["probabilities"]

                    st.subheader("Prediction")
                    st.success(f"Predicted expression: {emotion}")
                    st.metric(
                        "Confidence",
                        f"{confidence:.2%}",
                    )

                    chart_data = pd.DataFrame(
                        {
                            "Emotion": list(probabilities.keys()),
                            "Probability": [
                                float(value)
                                for value in probabilities.values()
                            ],
                        }
                    )

                    chart = (
                        alt.Chart(chart_data)
                        .mark_bar()
                        .encode(
                            x=alt.X(
                                "Emotion:N",
                                sort="-y",
                                title="Emotion",
                            ),
                            y=alt.Y(
                                "Probability:Q",
                                title="Probability",
                                scale=alt.Scale(domain=[0, 1]),
                            ),
                            tooltip=[
                                "Emotion",
                                alt.Tooltip(
                                    "Probability:Q",
                                    format=".2%",
                                ),
                            ],
                        )
                        .properties(
                            title="Emotion Probability Distribution",
                            height=350,
                        )
                    )

                    st.altair_chart(
                        chart,
                        use_container_width=True,
                    )

                    st.caption(
                        "Predictions describe facial-expression patterns, "
                        "not a person's actual emotional state."
                    )

                except requests.exceptions.ConnectionError:
                    st.error(
                        "Cannot connect to the FastAPI backend. "
                        "Make sure the API server is running."
                    )

                except requests.exceptions.Timeout:
                    st.error(
                        "The prediction request timed out. "
                        "Please try again."
                    )

                except requests.exceptions.HTTPError as error:
                    st.error(
                        f"The API returned an error: {error.response.text}"
                    )

                except (KeyError, TypeError, ValueError) as error:
                    st.error(
                        f"Unexpected API response: {error}"
                    )

    except Exception:
        st.error("Unable to open this image. Please upload a valid JPG or PNG.")
