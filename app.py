from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from src.predict import predict_image


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Brain Tumor MRI Classifier",
    page_icon="🧠",
    layout="centered",
)

ROOT = Path(__file__).resolve().parent


# ============================================================
# HEADER
# ============================================================

st.title("🧠 Brain Tumor MRI Image Classification")

st.write(
    "An educational deep-learning application that classifies "
    "uploaded MRI images into one of four dataset classes."
)

st.warning(
    "Educational and research use only. This application is not "
    "a medical device and cannot diagnose or rule out a brain tumor. "
    "Do not use its predictions for medical decisions. Consult a "
    "qualified healthcare professional for medical interpretation."
)


# ============================================================
# MODEL SELECTION
# ============================================================

model_choice = st.selectbox(
    "Select a classification model",
    ["MobileNetV2", "Custom CNN"],
)

if model_choice == "MobileNetV2":
    st.caption(
        "MobileNetV2 achieved 87.40% accuracy on this project's "
        "246-image test set. This result does not guarantee "
        "performance on other images."
    )
else:
    st.caption(
        "Custom CNN achieved 70.33% accuracy on this project's "
        "246-image test set. This result does not guarantee "
        "performance on other images."
    )


# ============================================================
# IMAGE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload a brain MRI image",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
)

if uploaded_file is not None:

    try:
        image = Image.open(uploaded_file).convert("RGB")

        st.subheader("Uploaded MRI Image")

        st.image(
            image,
            caption="Image selected for classification",
            use_container_width=True,
        )

        if st.button("Classify Image", type="primary"):

            with st.spinner("Classifying image..."):

                label, confidence, scores = predict_image(
                    image,
                    model_choice,
                )

            st.divider()

            st.subheader("Classification Result")

            st.success(
                f"Predicted dataset class: "
                f"{label.replace('_', ' ').title()}"
            )

            st.metric(
                "Highest model output score",
                f"{confidence:.2%}",
            )

            st.subheader("Scores for Each Dataset Class")

            score_df = pd.DataFrame(
                {
                    "Class": [
                        name.replace("_", " ").title()
                        for name in scores
                    ],
                    "Score": list(scores.values()),
                }
            )

            score_df = score_df.set_index("Class")

            st.bar_chart(score_df)

            st.caption(
                "These are model output scores, not validated "
                "clinical probabilities. They should not be used "
                "to make medical decisions."
            )

    except Exception as error:
        st.error(f"Unable to process the image: {error}")


# ============================================================
# PROJECT INFORMATION
# ============================================================

st.divider()

with st.expander("About this project"):

    st.write(
        """
        **Project:** Brain Tumor MRI Image Classification

        **Models:**
        - Custom Convolutional Neural Network (CNN)
        - MobileNetV2 transfer learning

        **Dataset classes:**
        - Glioma
        - Meningioma
        - No tumor
        - Pituitary

        **Image size:** 160 × 160 pixels

        **Purpose:** Educational image classification and model
        comparison.
        """
    )

st.caption(
    "This application is an educational demonstration, "
    "not a clinically validated diagnostic system."
)