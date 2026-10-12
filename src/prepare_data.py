try:
    from .data_utils import load_datasets, CLASS_NAMES
except ImportError:
    from data_utils import load_datasets, CLASS_NAMES
import matplotlib.pyplot as plt

def main():
    train_ds, valid_ds, test_ds = load_datasets("rescale")
    print("Classes:", CLASS_NAMES)
    print("Training batches:", train_ds.cardinality().numpy())
    print("Validation batches:", valid_ds.cardinality().numpy())
    print("Testing batches:", test_ds.cardinality().numpy())
    images, labels = next(iter(train_ds))
    plt.figure(figsize=(10, 8))
    for i in range(min(9, len(images))):
        plt.subplot(3, 3, i+1)
        plt.imshow(images[i].numpy())
        plt.title(CLASS_NAMES[int(labels[i].numpy())])
        plt.axis("off")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
