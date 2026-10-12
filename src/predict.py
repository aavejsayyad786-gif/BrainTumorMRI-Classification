from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

try:
    from .data_utils import MODEL_DIR, CLASS_NAMES, IMG_SIZE
except ImportError:
    from data_utils import MODEL_DIR, CLASS_NAMES, IMG_SIZE


# ============================================================
# LOAD AND CACHE MODELS
# ============================================================

@st.cache_resource
def load_model(model_choice="MobileNetV2"):
    """Load a trained model and cache it for repeated predictions."""

    if model_choice == "MobileNetV2":
        filename = "mobilenetv2.keras"
    elif model_choice == "Custom CNN":
        filename = "custom_cnn.keras"
    else:
        raise ValueError(f"Unsupported model: {model_choice}")

    model_path = MODEL_DIR / filename

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}. Train the model first."
        )

    return tf.keras.models.load_model(model_path)


# ============================================================
# PREDICT IMAGE
# ============================================================

def predict_image(image, model_choice="MobileNetV2"):
    """
    Predict the dataset class of a PIL image.

    Returns:
        class_name: Predicted dataset class.
        confidence: Highest model output score.
        all_scores: Model output scores for all classes.
    """

    model = load_model(model_choice)

    # Prepare the uploaded image.
    image = image.convert("RGB").resize(
        (IMG_SIZE[0], IMG_SIZE[1])
    )

    arr = np.asarray(image, dtype=np.float32)

    # Custom CNN uses Rescaling(1/255) inside the model.
    # MobileNetV2 uses preprocess_input inside the model.
    # Therefore, pass 0-255 pixel values to both models.
    batch = np.expand_dims(arr, axis=0)

    predictions = model.predict(batch, verbose=0)[0]

    class_index = int(np.argmax(predictions))

    scores = {
        name: float(predictions[index])
        for index, name in enumerate(CLASS_NAMES)
    }

    return (
        CLASS_NAMES[class_index],
        float(predictions[class_index]),
        scores,
    )