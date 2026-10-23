```python
import os

import altair as alt
import pandas as pd
import requests
import streamlit as st
from PIL import Image, UnidentifiedImageError


# =========================================================
# CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="MindMirror AI",
    page_icon="🧠",
    layout="centered",
)

# Priority:
# 1. Streamlit Secrets (deployed application)
# 2. Environment variable
# 3. Localhost (local development)

try:
    API_URL = st.secrets.get(
        "API_URL",
        os.getenv("API_URL", "http://127.0.0.1:8000"),
    ).rstrip("/")
except Exception:
    API_URL = os.getenv(
        "API_URL",
        "http://127.0.0.1:8000",
    ).rstrip("/")


# =========================================================
# HEADER
# =========================================================

st.title("🧠 MindMirror AI")

st.write(
    "Upload a facial image to predict its expression "
    "using a Convolutional Neural Network (CNN)."
)

st.caption(
    "This application identifies facial-expression patterns. "
    "It cannot determine a person's actual emotional state."
)


# =========================================================
# BACKEND STATUS
# =========================================================

with st.expander("Backend configuration"):
    st.write(f"API endpoint: `{API_URL}`")


# =========================================================
# IMAGE UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "Choose a facial image",
    type=["jpg", "jpeg", "png"],
)


# =========================================================
# EMOTION PREDICTION
# =========================================================

if uploaded_file is not None:

    try:
        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded image",
            use_container_width=True,
        )

        if st.button("Analyze Emotion", type="primary"):

            with st.spinner(
                "Analyzing facial expression... "
                "The server may take a moment to wake up."
            ):

                try:
                    # Send the uploaded image to FastAPI
                    response = requests.post(
                        f"{API_URL}/predict",
                        files={
                            "file": (
                                uploaded_file.name,
                                uploaded_file.getvalue(),
                                uploaded_file.type or "image/jpeg",
                            )
                        },
                        timeout=180,
                    )

                    response.raise_for_status()

                    result = response.json()

                    # -----------------------------------------
                    # EXTRACT PREDICTION RESULTS
                    # -----------------------------------------

                    emotion = result["emotion"]
                    confidence = float(result["confidence"])
                    probabilities = result["probabilities"]

                    if not isinstance(probabilities, dict):
                        raise ValueError(
                            "The API returned an invalid probabilities format."
                        )

                    # -----------------------------------------
                    # DISPLAY PREDICTION
                    # -----------------------------------------

                    st.subheader("🎯 Prediction")

                    st.success(
                        f"Predicted expression: {emotion}"
                    )

                    st.metric(
                        label="Model Confidence",
                        value=f"{confidence:.2%}",
                    )

                    # -----------------------------------------
                    # PROBABILITY CHART
                    # -----------------------------------------

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
                                title="Facial Expression",
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
                        "Probabilities represent the model's outputs, "
                        "not verified emotional states."
                    )

                # ---------------------------------------------
                # ERROR HANDLING
                # ---------------------------------------------

                except requests.exceptions.ConnectionError:
                    st.error(
                        "Cannot connect to the FastAPI backend. "
                        "Check API_URL in Streamlit Secrets and "
                        "verify that the Render backend is available."
                    )

                except requests.exceptions.Timeout:
                    st.error(
                        "The prediction request timed out. "
                        "The server may be waking up or processing "
                        "the image. Please try again."
                    )

                except requests.exceptions.HTTPError as error:
                    status_code = (
                        error.response.status_code
                        if error.response is not None
                        else "Unknown"
                    )

                    st.error(
                        f"The API returned HTTP {status_code}. "
                        "Check the backend logs for details."
                    )

                    if error.response is not None:
                        st.code(error.response.text)

                except requests.exceptions.RequestException as error:
                    st.error(f"Request failed: {error}")

                except (
                    KeyError,
                    TypeError,
                    ValueError,
                ) as error:
                    st.error(
                        f"Unexpected API response: {error}"
                    )

    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
    ):
        st.error(
            "Unable to open this image. "
            "Please upload a valid JPG or PNG image."
        )
```
