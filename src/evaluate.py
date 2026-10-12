import os

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["TF_NUM_INTRAOP_THREADS"] = "2"
os.environ["TF_NUM_INTEROP_THREADS"] = "2"

import json
from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

TEST_DIR = BASE_DIR / "dataset" / "test"
MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = (160, 160)
BATCH_SIZE = 8

MODEL_PATHS = {
    "Custom CNN": MODEL_DIR / "custom_cnn.keras",
    "MobileNetV2": MODEL_DIR / "mobilenetv2.keras",
}

CLASS_NAMES_PATH = MODEL_DIR / "class_names.json"

# ============================================================
# LOAD CLASS NAMES
# ============================================================

with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
    class_names = json.load(file)

print("=" * 65)
print("BRAIN TUMOR MRI - MODEL EVALUATION")
print("=" * 65)
print("Classes:", class_names)

# ============================================================
# LOAD TEST DATASET
# ============================================================

test_ds = tf.keras.utils.image_dataset_from_directory(
    str(TEST_DIR),
    labels="inferred",
    label_mode="int",
    class_names=class_names,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

# Collect true labels once for both models.
y_true = np.concatenate(
    [labels.numpy() for _, labels in test_ds]
)

# ============================================================
# EVALUATE MODELS
# ============================================================

comparison = {}

for model_name, model_path in MODEL_PATHS.items():

    print("\n" + "=" * 65)
    print(f"EVALUATING: {model_name}")
    print("=" * 65)

    if not model_path.exists():
        print(f"Model not found: {model_path}")
        print("Skipping this model.")
        continue

    model = tf.keras.models.load_model(model_path)

    print("Model loaded:", model_path)
    print("Input shape:", model.input_shape)

    probabilities = model.predict(test_ds, verbose=1)
    y_pred = np.argmax(probabilities, axis=1)

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    accuracy = accuracy_score(y_true, y_pred)

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    print(f"\nTest accuracy: {accuracy * 100:.2f}%")

    print("\nClassification report:")

    print(
        classification_report(
            y_true,
            y_pred,
            labels=list(range(len(class_names))),
            target_names=class_names,
            digits=4,
            zero_division=0,
        )
    )

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names,
    )

    fig, ax = plt.subplots(figsize=(8, 6))

    display.plot(
        ax=ax,
        cmap="Blues",
        values_format="d",
        xticks_rotation=45,
    )

    plt.title(f"{model_name} - Test Confusion Matrix")
    plt.tight_layout()

    safe_name = model_name.lower().replace(" ", "_")

    confusion_path = (
        RESULTS_DIR / f"{safe_name}_confusion_matrix.png"
    )

    plt.savefig(confusion_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("Confusion matrix saved:", confusion_path)

    # --------------------------------------------------------
    # SAVE METRICS
    # --------------------------------------------------------

    comparison[model_name] = {
        "test_samples": int(len(y_true)),
        "test_accuracy": float(accuracy),
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall": float(report["macro avg"]["recall"]),
        "macro_f1_score": float(report["macro avg"]["f1-score"]),
        "weighted_precision": float(
            report["weighted avg"]["precision"]
        ),
        "weighted_recall": float(
            report["weighted avg"]["recall"]
        ),
        "weighted_f1_score": float(
            report["weighted avg"]["f1-score"]
        ),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }

    del model

# ============================================================
# SAVE COMPARISON RESULTS
# ============================================================

comparison_path = RESULTS_DIR / "model_comparison.json"

with open(comparison_path, "w", encoding="utf-8") as file:
    json.dump(comparison, file, indent=4)

print("\n" + "=" * 65)
print("MODEL COMPARISON")
print("=" * 65)

print(
    f"{'Model':<20}"
    f"{'Accuracy':>12}"
    f"{'Macro F1':>12}"
)

for model_name, metrics in comparison.items():
    print(
        f"{model_name:<20}"
        f"{metrics['test_accuracy'] * 100:>11.2f}%"
        f"{metrics['macro_f1_score'] * 100:>11.2f}%"
    )

print("\nComparison saved to:", comparison_path)
print("\nEVALUATION COMPLETED")