import os

# Limit CPU thread usage on Windows.
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["TF_NUM_INTRAOP_THREADS"] = "2"
os.environ["TF_NUM_INTEROP_THREADS"] = "2"

import json
import traceback
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, models

# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = BASE_DIR / "dataset"
TRAIN_DIR = DATASET_DIR / "train"
VALID_DIR = DATASET_DIR / "valid"

MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "mobilenetv2.keras"
FINAL_MODEL_PATH = MODEL_DIR / "mobilenetv2_final.keras"
CLASS_NAMES_PATH = MODEL_DIR / "class_names.json"

IMAGE_SIZE = (160, 160)
BATCH_SIZE = 8

INITIAL_EPOCHS = 8
FINE_TUNE_EPOCHS = 4

SEED = 42

tf.keras.utils.set_random_seed(SEED)

try:
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
except RuntimeError:
    pass

print("=" * 60)
print("BRAIN TUMOR MRI - MOBILENETV2 TRANSFER LEARNING")
print("=" * 60)
print("TensorFlow:", tf.__version__)
print("Image size:", IMAGE_SIZE)
print("Batch size:", BATCH_SIZE)

# ============================================================
# 2. CHECK DATASET
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
# 3. LOAD DATASETS
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
)

class_names = train_ds.class_names
num_classes = len(class_names)

print("Classes:", class_names)

print("\nLoading validation dataset...")

valid_ds = tf.keras.utils.image_dataset_from_directory(
    str(VALID_DIR),
    labels="inferred",
    label_mode="int",
    class_names=class_names,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

with open(CLASS_NAMES_PATH, "w", encoding="utf-8") as file:
    json.dump(class_names, file, indent=4)

train_ds = train_ds.prefetch(1)
valid_ds = valid_ds.prefetch(1)

# ============================================================
# 4. BUILD MOBILENETV2 MODEL
# ============================================================

def build_model():
    print("\nLoading MobileNetV2 ImageNet weights...")

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
        include_top=False,
        weights="imagenet",
    )

    # Freeze the pretrained feature extractor initially.
    base_model.trainable = False

    inputs = layers.Input(
        shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
        name="input_image",
    )

    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.03)(x)
    x = layers.RandomZoom(0.06)(x)

    # MobileNetV2 requires preprocessing to approximately [-1, 1].
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

    x = base_model(x, training=False)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.30)(x)

    x = layers.Dense(
        64,
        activation="relu",
        name="dense_features",
    )(x)

    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        name="class_predictions",
    )(x)

    model = models.Model(
        inputs,
        outputs,
        name="BrainTumor_MobileNetV2",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model, base_model


# ============================================================
# 5. CALLBACKS
# ============================================================

def create_callbacks(history_filename):
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_PATH),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(
            str(RESULTS_DIR / history_filename),
        ),
    ]


# ============================================================
# 6. TRAINING
# ============================================================

def main():
    model, base_model = build_model()

    model.summary()

    print("\n" + "=" * 60)
    print("STAGE 1: TRAINING CLASSIFICATION HEAD")
    print("=" * 60)

    initial_history = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=INITIAL_EPOCHS,
        callbacks=create_callbacks(
            "mobilenetv2_initial_history.csv"
        ),
        verbose=1,
    )

    model.save(str(MODEL_DIR / "mobilenetv2_stage1.keras"))

    print("\n" + "=" * 60)
    print("STAGE 2: FINE-TUNING LAST 20 LAYERS")
    print("=" * 60)

    base_model.trainable = True

    # Freeze all but the last 20 layers.
    for layer in base_model.layers[:-20]:
        layer.trainable = False

    # Keep batch normalization layers frozen for stability.
    for layer in base_model.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    # Recompile after changing trainable layers.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    fine_history = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=FINE_TUNE_EPOCHS,
        callbacks=create_callbacks(
            "mobilenetv2_finetune_history.csv"
        ),
        verbose=1,
    )

    # Save the final model.
    model.save(str(FINAL_MODEL_PATH))

    # Save both training histories.
    history_data = {
        "initial_training": {
            key: [float(v) for v in values]
            for key, values in initial_history.history.items()
        },
        "fine_tuning": {
            key: [float(v) for v in values]
            for key, values in fine_history.history.items()
        },
    }

    history_path = RESULTS_DIR / "mobilenetv2_history.json"

    with open(history_path, "w", encoding="utf-8") as file:
        json.dump(history_data, file, indent=4)

    print("\n" + "=" * 60)
    print("MOBILENETV2 TRAINING COMPLETED")
    print("=" * 60)

    print("Best checkpoint:", MODEL_PATH)
    print("Final model:", FINAL_MODEL_PATH)
    print("Class names:", CLASS_NAMES_PATH)
    print("Training history:", history_path)

    if fine_history.history.get("val_accuracy"):
        print(
            "Final fine-tuning validation accuracy:",
            f"{fine_history.history['val_accuracy'][-1]:.4f}",
        )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\nMOBILENETV2 TRAINING FAILED")
        traceback.print_exc()
        raise