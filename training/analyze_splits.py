from pathlib import Path
from collections import Counter
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]

VOC_DIR = ROOT / "data" / "raw" / "voc_labels"
TRAIN_FILE = ROOT / "data" / "raw" / "train_files.txt"
VAL_FILE = ROOT / "data" / "raw" / "val_files.txt"


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


def load_split(path):
    return {
        Path(x.strip()).stem
        for x in path.read_text(encoding="utf-8").splitlines()
        if x.strip()
    }


def count_objects(image_ids):

    counts = Counter()
    images_with_class = Counter()

    for image_id in image_ids:

        xml_path = VOC_DIR / f"{image_id}.xml"

        if not xml_path.exists():
            continue

        root = ET.parse(xml_path).getroot()

        classes_in_image = set()

        for obj in root.findall("object"):

            name = obj.findtext("name")

            if not name:
                continue

            name = name.strip()

            counts[name] += 1
            classes_in_image.add(name)

        for name in classes_in_image:
            images_with_class[name] += 1

    return counts, images_with_class


def main():

    train_ids = load_split(TRAIN_FILE)
    val_ids = load_split(VAL_FILE)

    train_counts, train_images = count_objects(train_ids)
    val_counts, val_images = count_objects(val_ids)

    print("=" * 75)
    print("PPE-SENTINEL TRAIN / VALIDATION CLASS DISTRIBUTION")
    print("=" * 75)

    print()
    print(
        f"{'Class':<25}"
        f"{'Train Obj':>12}"
        f"{'Val Obj':>12}"
        f"{'Total':>12}"
        f"{'Val %':>10}"
    )

    print("-" * 75)

    for cls in CLASS_NAMES:

        train = train_counts[cls]
        val = val_counts[cls]
        total = train + val

        val_pct = (val / total * 100) if total else 0

        print(
            f"{cls:<25}"
            f"{train:>12}"
            f"{val:>12}"
            f"{total:>12}"
            f"{val_pct:>9.2f}%"
        )

    print()
    print("IMAGE-LEVEL DISTRIBUTION")
    print("-" * 75)

    print(
        f"{'Class':<25}"
        f"{'Train Images':>15}"
        f"{'Val Images':>15}"
        f"{'Total Images':>15}"
    )

    print("-" * 75)

    for cls in CLASS_NAMES:

        train = train_images[cls]
        val = val_images[cls]

        print(
            f"{cls:<25}"
            f"{train:>15}"
            f"{val:>15}"
            f"{train + val:>15}"
        )


if __name__ == "__main__":
    main()