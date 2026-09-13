from pathlib import Path
import shutil
import hashlib


ROOT = Path(__file__).resolve().parents[1]

RAW_IMAGES = ROOT / "data" / "raw" / "images"
RAW_LABELS = ROOT / "data" / "raw" / "labels"

TRAIN_FILE = ROOT / "data" / "raw" / "train_files.txt"
VAL_FILE = ROOT / "data" / "raw" / "val_files.txt"

OUTPUT_ROOT = ROOT / "data" / "processed" / "sh17"


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def load_split(path):
    """Load image stems from a SH17 split file."""
    return {
        Path(line.strip()).stem
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def build_image_index():
    """Create image-stem -> image-path mapping."""
    index = {}

    for image_path in RAW_IMAGES.iterdir():
        if (
            image_path.is_file()
            and image_path.suffix.lower() in IMAGE_EXTENSIONS
        ):
            index[image_path.stem] = image_path

    return index


def sha256(path):
    """Return SHA-256 checksum for a file."""
    digest = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def prepare_split(split_name, image_ids, image_index):

    output_images = OUTPUT_ROOT / split_name / "images"
    output_labels = OUTPUT_ROOT / split_name / "labels"

    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)

    copied = 0
    missing_images = []
    missing_labels = []

    for image_id in sorted(image_ids):

        image_path = image_index.get(image_id)
        label_path = RAW_LABELS / f"{image_id}.txt"

        if image_path is None:
            missing_images.append(image_id)
            continue

        if not label_path.exists():
            missing_labels.append(image_id)
            continue

        destination_image = output_images / image_path.name
        destination_label = output_labels / label_path.name

        shutil.copy2(image_path, destination_image)
        shutil.copy2(label_path, destination_label)

        copied += 1

    return copied, missing_images, missing_labels


def verify_split(split_name):

    image_dir = OUTPUT_ROOT / split_name / "images"
    label_dir = OUTPUT_ROOT / split_name / "labels"

    images = {
        p.stem
        for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    }

    labels = {
        p.stem
        for p in label_dir.glob("*.txt")
    }

    missing_labels = images - labels
    missing_images = labels - images

    return (
        len(images),
        len(labels),
        missing_labels,
        missing_images,
    )


def main():

    print("=" * 70)
    print("PPE-SENTINEL DATASET PREPARATION")
    print("=" * 70)

    train_ids = load_split(TRAIN_FILE)
    val_ids = load_split(VAL_FILE)

    print(f"Train IDs: {len(train_ids)}")
    print(f"Validation IDs: {len(val_ids)}")

    overlap = train_ids & val_ids

    if overlap:
        raise RuntimeError(
            f"Train/validation overlap detected: {len(overlap)}"
        )

    print("Train/validation overlap: 0")

    image_index = build_image_index()

    print(f"Indexed images: {len(image_index)}")

    train_result = prepare_split(
        "train",
        train_ids,
        image_index,
    )

    val_result = prepare_split(
        "val",
        val_ids,
        image_index,
    )

    train_copied, train_missing_images, train_missing_labels = train_result
    val_copied, val_missing_images, val_missing_labels = val_result

    print()
    print("=== COPY RESULTS ===")

    print(f"Train copied: {train_copied}")
    print(f"Validation copied: {val_copied}")

    if train_missing_images:
        print(
            f"Train missing images: {len(train_missing_images)}"
        )

    if train_missing_labels:
        print(
            f"Train missing labels: {len(train_missing_labels)}"
        )

    if val_missing_images:
        print(
            f"Validation missing images: {len(val_missing_images)}"
        )

    if val_missing_labels:
        print(
            f"Validation missing labels: {len(val_missing_labels)}"
        )

    if (
        train_missing_images
        or train_missing_labels
        or val_missing_images
        or val_missing_labels
    ):
        raise RuntimeError(
            "Dataset preparation encountered missing files."
        )

    print()
    print("=== OUTPUT VERIFICATION ===")

    for split in ["train", "val"]:

        image_count, label_count, missing_labels, missing_images = (
            verify_split(split)
        )

        print(f"\n{split.upper()}")
        print(f"Images: {image_count}")
        print(f"Labels: {label_count}")
        print(f"Images without labels: {len(missing_labels)}")
        print(f"Labels without images: {len(missing_images)}")

        if missing_labels or missing_images:
            raise RuntimeError(
                f"Verification failed for {split}."
            )

    print()
    print("Dataset preparation completed successfully.")

    print()
    print(f"Processed dataset: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()