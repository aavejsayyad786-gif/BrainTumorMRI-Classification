import os

# ============================================================
# 1. CPU CONFIGURATION
# ============================================================

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["TF_NUM_INTRAOP_THREADS"] = "2"
os.environ["TF_NUM_INTEROP_THREADS"] = "2"

import json
import traceback
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)

# ============================================================
# 2. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = BASE_DIR / "dataset"
TRAIN_DIR = DATASET_DIR / "train"
VALID_DIR = DATASET_DIR / "valid"

MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "custom_cnn.keras"
CLASS_NAMES_PATH = MODEL_DIR / "class_names.json"
HISTORY_PATH = MODEL_DIR / "custom_cnn_history.json"

IMAGE_SIZE = (160, 160)
BATCH_SIZE = 8
EPOCHS = 15
SEED = 42

tf.keras.utils.set_random_seed(SEED)

try:
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
except RuntimeError:
    print("TensorFlow threading was already initialized.")

print("=" * 60)
print("BRAIN TUMOR MRI CLASSIFICATION - CUSTOM CNN")
print("=" * 60)
print(f"Project directory: {BASE_DIR}")
print(f"Image size: {IMAGE_SIZE}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Epochs: {EPOCHS}")
print(f"TensorFlow version: {tf.__version__}")
print("=" * 60)

# ============================================================
# 3. CHECK DATASET DIRECTORIES
# ============================================================

if not TRAIN_DIR.exists():
    raise FileNotFoundError(
        f"Training directory not found: {TRAIN_DIR}"
    )

if not VALID_DIR.exists():
    raise FileNotFoundError(
        f"Validation directory not found: {VALID_DIR}"
    )

# ============================================================
# 4. LOAD TRAINING DATA
# ============================================================

print("\nLoading training dataset...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    str(TRAIN_DIR),
    labels="inferred",
    label_mode="int",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED,
    interpolation="bilinear",
)

class_names = train_ds.class_names
num_classes = len(class_names)

if num_classes < 2:
    raise ValueError("At least two classes are required.")

print("Classes:", class_names)
print("Number of classes:", num_classes)

with open(CLASS_NAMES_PATH, "w", encoding="utf-8") as file:
    json.dump(class_names, file, indent=4)

# ============================================================
# 5. LOAD VALIDATION DATA
# ============================================================

print("\nLoading validation dataset...")

valid_ds = tf.keras.utils.image_dataset_from_directory(
    str(VALID_DIR),
    labels="inferred",
    label_mode="int",
    class_names=class_names,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
    interpolation="bilinear",
)

# Limit prefetching to help control memory consumption.
train_ds = train_ds.prefetch(1)
valid_ds = valid_ds.prefetch(1)

# ============================================================
# 6. DATA AUGMENTATION
# ============================================================

data_augmentation = models.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.10),
    ],
    name="data_augmentation",
)

# ============================================================
# 7. BUILD CUSTOM CNN MODEL
# ============================================================

print("\nBuilding custom CNN model...")

inputs = layers.Input(
    shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
    name="input_image",
)

x = data_augmentation(inputs)

# Normalize pixel values to the range 0-1.
x = layers.Rescaling(
    1.0 / 255.0,
    name="normalization",
)(x)

# Convolution block 1
x = layers.Conv2D(
    16,
    kernel_size=3,
    padding="same",
    activation="relu",
    name="conv2d_1",
)(x)

x = layers.MaxPooling2D(
    pool_size=2,
    name="maxpool_1",
)(x)

# Convolution block 2
x = layers.Conv2D(
    32,
    kernel_size=3,
    padding="same",
    activation="relu",
    name="conv2d_2",
)(x)

x = layers.MaxPooling2D(
    pool_size=2,
    name="maxpool_2",
)(x)

# Convolution block 3
x = layers.Conv2D(
    64,
    kernel_size=3,
    padding="same",
    activation="relu",
    name="conv2d_3",
)(x)

x = layers.MaxPooling2D(
    pool_size=2,
    name="maxpool_3",
)(x)

# Classification head
x = layers.GlobalAveragePooling2D(
    name="global_average_pooling",
)(x)

x = layers.Dense(
    64,
    activation="relu",
    name="dense_features",
)(x)

x = layers.Dropout(
    0.30,
    name="dropout",
)(x)

outputs = layers.Dense(
    num_classes,
    activation="softmax",
    name="class_predictions",
)(x)

model = models.Model(
    inputs=inputs,
    outputs=outputs,
    name="BrainTumor_CustomCNN",
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# ============================================================
# 8. TRAINING CALLBACKS
# ============================================================

callbacks = [
    ModelCheckpoint(
        filepath=str(MODEL_PATH),
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    ),

    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=0.00001,
        verbose=1,
    ),

    EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True,
        verbose=1,
    ),
]

# ============================================================
# 9. TRAIN MODEL
# ============================================================

print("\nStarting custom CNN training...")
print(f"Maximum epochs: {EPOCHS}")
print("Best model will be saved automatically.")

try:
    history = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

    # Save the final model using the restored best weights.
    model.save(str(MODEL_PATH))

    # Save training history for later analysis.
    history_data = {
        key: [float(value) for value in values]
        for key, values in history.history.items()
    }

    with open(HISTORY_PATH, "w", encoding="utf-8") as file:
        json.dump(history_data, file, indent=4)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print(f"Model saved: {MODEL_PATH}")
    print(f"Class names saved: {CLASS_NAMES_PATH}")
    print(f"Training history saved: {HISTORY_PATH}")

    if "accuracy" in history.history:
        print(
            f"Final training accuracy: "
            f"{history.history['accuracy'][-1]:.4f}"
        )

    if "val_accuracy" in history.history:
        print(
            f"Final validation accuracy: "
            f"{history.history['val_accuracy'][-1]:.4f}"
        )

    if "val_loss" in history.history:
        print(
            f"Final validation loss: "
            f"{history.history['val_loss'][-1]:.4f}"
        )

except Exception:
    print("\nTRAINING FAILED")
    traceback.print_exc()
    raise