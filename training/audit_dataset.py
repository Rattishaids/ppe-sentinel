from pathlib import Path
from collections import Counter
import json
import xml.etree.ElementTree as ET


PROJECT_ROOT = Path(__file__).resolve().parents[1]

IMAGE_DIR = PROJECT_ROOT / "data" / "raw" / "images"
YOLO_LABEL_DIR = PROJECT_ROOT / "data" / "raw" / "labels"
VOC_DIR = PROJECT_ROOT / "data" / "raw" / "voc_labels"

TRAIN_FILE = PROJECT_ROOT / "data" / "raw" / "train_files.txt"
VAL_FILE = PROJECT_ROOT / "data" / "raw" / "val_files.txt"


CLASS_NAMES = [
    "person",
    "ear",
    "ear-mufs",
    "face",
    "face-guard",
    "face-mask-medical",
    "foot",
    "tools",
    "glasses",
    "gloves",
    "helmet",
    "hands",
    "head",
    "medical-suit",
    "shoes",
    "safety-suit",
    "safety-vest",
]


def load_split_file(path):
    """Load image identifiers from train/validation split file."""
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")

    return {
        Path(line.strip()).stem
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def get_image_dimensions(image_path):
    """
    Read image dimensions without requiring an image library.
    Uses Pillow if available.
    """
    try:
        from PIL import Image

        with Image.open(image_path) as img:
            return img.width, img.height

    except ImportError:
        return None, None


def audit_images():
    images = [
        p for p in IMAGE_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]

    labels = [
        p for p in YOLO_LABEL_DIR.iterdir()
        if p.is_file() and p.suffix.lower() == ".txt"
    ]

    image_stems = {p.stem for p in images}
    label_stems = {p.stem for p in labels}

    missing_labels = sorted(image_stems - label_stems)
    missing_images = sorted(label_stems - image_stems)

    print("\n=== DATASET INTEGRITY ===")
    print(f"Images: {len(images)}")
    print(f"YOLO labels: {len(labels)}")
    print(f"Images missing labels: {len(missing_labels)}")
    print(f"Labels missing images: {len(missing_images)}")

    return images, labels


def audit_splits():
    train_ids = load_split_file(TRAIN_FILE)
    val_ids = load_split_file(VAL_FILE)

    overlap = train_ids & val_ids

    print("\n=== SPLIT INTEGRITY ===")
    print(f"Train images: {len(train_ids)}")
    print(f"Validation images: {len(val_ids)}")
    print(f"Total split images: {len(train_ids | val_ids)}")
    print(f"Train/validation overlap: {len(overlap)}")

    return train_ids, val_ids


def audit_voc_instances():
    class_counts = Counter()
    image_class_counts = Counter()
    object_sizes = Counter()

    total_objects = 0

    for xml_path in VOC_DIR.glob("*.xml"):

        try:
            root = ET.parse(xml_path).getroot()
        except Exception as exc:
            print(f"WARNING: Could not parse {xml_path.name}: {exc}")
            continue

        image_classes = set()

        size_element = root.find("size")

        if size_element is not None:
            try:
                image_width = float(size_element.findtext("width"))
                image_height = float(size_element.findtext("height"))
            except (TypeError, ValueError):
                image_width = None
                image_height = None
        else:
            image_width = None
            image_height = None

        for obj in root.findall("object"):

            name_element = obj.find("name")

            if name_element is None or name_element.text is None:
                continue

            class_name = name_element.text.strip()

            class_counts[class_name] += 1
            image_classes.add(class_name)
            total_objects += 1

            bbox = obj.find("bndbox")

            if bbox is not None and image_width and image_height:

                try:
                    xmin = float(bbox.findtext("xmin"))
                    ymin = float(bbox.findtext("ymin"))
                    xmax = float(bbox.findtext("xmax"))
                    ymax = float(bbox.findtext("ymax"))

                    box_width = max(0.0, xmax - xmin)
                    box_height = max(0.0, ymax - ymin)

                    area_ratio = (
                        box_width * box_height
                    ) / (image_width * image_height)

                    if area_ratio < 0.01:
                        size = "small_<1%"
                    elif area_ratio < 0.05:
                        size = "medium_1-5%"
                    else:
                        size = "large_>5%"

                    object_sizes[(class_name, size)] += 1

                except (TypeError, ValueError):
                    pass

        for class_name in image_classes:
            image_class_counts[class_name] += 1

    print("\n=== INSTANCE DISTRIBUTION ===")
    print(f"Total annotated objects: {total_objects}")

    print("\nClass                     Instances")
    print("------------------------------------")

    for class_name, count in class_counts.most_common():
        print(f"{class_name:<25} {count:>8}")

    print("\n=== IMAGE-LEVEL CLASS FREQUENCY ===")
    print("\nClass                     Images")
    print("--------------------------------")

    for class_name, count in image_class_counts.most_common():
        print(f"{class_name:<25} {count:>8}")

    print("\n=== OBJECT SIZE DISTRIBUTION ===")
    print("\nClass                     Small      Medium     Large")
    print("------------------------------------------------------")

    for class_name in CLASS_NAMES:

        small = object_sizes[(class_name, "small_<1%")]
        medium = object_sizes[(class_name, "medium_1-5%")]
        large = object_sizes[(class_name, "large_>5%")]

        if small + medium + large > 0:
            print(
                f"{class_name:<25}"
                f"{small:>8}"
                f"{medium:>12}"
                f"{large:>11}"
            )

    return class_counts


def audit_yolo_class_ids():
    observed_ids = Counter()
    invalid_ids = []

    for label_path in YOLO_LABEL_DIR.glob("*.txt"):

        try:
            lines = label_path.read_text(
                encoding="utf-8"
            ).splitlines()
        except Exception as exc:
            print(f"WARNING: Could not read {label_path.name}: {exc}")
            continue

        for line_number, line in enumerate(lines, start=1):

            line = line.strip()

            if not line:
                continue

            parts = line.split()

            try:
                class_id = int(parts[0])
            except (ValueError, IndexError):
                invalid_ids.append(
                    f"{label_path.name}:{line_number}"
                )
                continue

            observed_ids[class_id] += 1

    print("\n=== YOLO CLASS IDS ===")
    print("\nID    Class                    Instances")
    print("----------------------------------------")

    for class_id in sorted(observed_ids):

        if 0 <= class_id < len(CLASS_NAMES):
            name = CLASS_NAMES[class_id]
        else:
            name = "INVALID_ID"

        print(
            f"{class_id:<5}"
            f"{name:<25}"
            f"{observed_ids[class_id]:>8}"
        )

    print(f"\nInvalid label lines: {len(invalid_ids)}")

    return observed_ids


def main():

    print("=" * 60)
    print("PPE-SENTINEL DATASET AUDIT")
    print("=" * 60)

    audit_images()

    train_ids, val_ids = audit_splits()

    class_counts = audit_voc_instances()

    observed_ids = audit_yolo_class_ids()

    print("\n=== FINAL SANITY CHECK ===")

    print(
        f"VOC annotated instances: "
        f"{sum(class_counts.values())}"
    )

    print(
        f"YOLO annotated instances: "
        f"{sum(observed_ids.values())}"
    )

    print(
        f"Expected SH17 classes: "
        f"{len(CLASS_NAMES)}"
    )

    print("\nAudit complete.")


if __name__ == "__main__":
    main()