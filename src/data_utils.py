from pathlib import Path
import tensorflow as tf

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_DIR / "dataset"
MODEL_DIR = PROJECT_DIR / "models"
OUTPUT_DIR = PROJECT_DIR / "outputs"
CLASS_NAMES = ["glioma", "meningioma", "no_tumor", "pituitary"]
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def load_datasets(preprocess="rescale", batch_size=BATCH_SIZE):
    """Load fixed train/valid/test folders. preprocess='rescale' or 'mobilenet'."""
    datasets = {}
    for split, shuffle in [("train", True), ("valid", False), ("test", False)]:
        folder = DATASET_DIR / split
        if not folder.is_dir():
            raise FileNotFoundError(f"Missing dataset folder: {folder}")
        ds = tf.keras.utils.image_dataset_from_directory(
            str(folder), labels="inferred", label_mode="int",
            class_names=CLASS_NAMES, image_size=IMG_SIZE, batch_size=batch_size,
            shuffle=shuffle, seed=SEED)
        if preprocess == "rescale":
            ds = ds.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y), num_parallel_calls=tf.data.AUTOTUNE)
        elif preprocess == "mobilenet":
            ds = ds.map(lambda x, y: (tf.keras.applications.mobilenet_v2.preprocess_input(tf.cast(x, tf.float32)), y), num_parallel_calls=tf.data.AUTOTUNE)
        elif preprocess == "none":
            ds = ds.map(lambda x, y: (tf.cast(x, tf.float32), y), num_parallel_calls=tf.data.AUTOTUNE)
        else:
            raise ValueError("preprocess must be 'rescale', 'mobilenet', or 'none'")
        datasets[split] = ds.prefetch(tf.data.AUTOTUNE)
    return datasets["train"], datasets["valid"], datasets["test"]

def ensure_output_dirs():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def count_images(split):
    root = DATASET_DIR / split
    return {name: sum(1 for p in (root/name).rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS) for name in CLASS_NAMES}
