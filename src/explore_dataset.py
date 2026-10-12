from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_DIR / "dataset"
SPLITS = ("train", "valid", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def main():
    print("\nBRAIN TUMOR MRI DATASET CHECK")
    print("=" * 52)
    grand_total = 0
    reference_classes = None
    for split in SPLITS:
        root = DATASET_DIR / split
        print(f"\n{split.upper()}")
        if not root.is_dir():
            print(f"ERROR: Missing directory: {root}")
            continue
        classes = sorted(p.name for p in root.iterdir() if p.is_dir())
        if reference_classes is None:
            reference_classes = classes
        elif classes != reference_classes:
            print(f"WARNING: Class folders differ. Found: {classes}")
        split_total = 0
        for name in classes:
            folder = root / name
            count = sum(1 for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
            print(f"  {name:15s} {count:5d}")
            split_total += count
        print(f"  {'TOTAL':15s} {split_total:5d}")
        grand_total += split_total
    print("\n" + "=" * 52)
    print(f"TOTAL IMAGES: {grand_total}")
    print(f"CLASSES: {reference_classes}")
    print("Expected totals for the supplied dataset: train=1695, valid=502, test=246.")

if __name__ == "__main__":
    main()
