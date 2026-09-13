from pathlib import Path
from collections import Counter
import shutil
import os


ROOT = Path(__file__).resolve().parents[1]

SOURCE_IMAGES = (
    ROOT
    / "data"
    / "processed"
    / "sh17"
    / "train"
    / "images"
)

SOURCE_LABELS = (
    ROOT
    / "data"
    / "processed"
    / "sh17"
    / "train"
    / "labels"
)

OUTPUT_ROOT = (
    ROOT
    / "data"
    / "processed"
    / "sh17_ppe_balanced"
)

OUTPUT_IMAGES = OUTPUT_ROOT / "images"
OUTPUT_LABELS = OUTPUT_ROOT / "labels"


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
}


CLASS_NAMES = {
    0: "person",
    1: "ear",
    2: "ear-mufs",
    3: "face",
    4: "face-guard",
    5: "face-mask-medical",
    6: "foot",
    7: "tools",
    8: "glasses",
    9: "gloves",
    10: "helmet",
    11: "hands",
    12: "head",
    13: "medical-suit",
    14: "shoes",
    15: "safety-suit",
    16: "safety-vest",
}


PPE_CLASSES = {
    "gloves",
    "helmet",
    "safety-vest",
}


WEIGHTS = {
    "gloves": 1.35,
    "helmet": 1.85,
    "safety-vest": 2.35,
}


def get_ppe_classes(label_path):

    classes = set()

    for line in label_path.read_text(
        encoding="utf-8"
    ).splitlines():

        parts = line.strip().split()

        if len(parts) < 5:
            continue

        class_id = int(parts[0])

        if class_id in CLASS_NAMES:

            class_name = CLASS_NAMES[
                class_id
            ]

            if class_name in PPE_CLASSES:
                classes.add(class_name)

    return classes


def create_link_or_copy(source, destination):

    if destination.exists():
        return

    try:

        os.link(
            source,
            destination,
        )

        return "hardlink"

    except OSError:

        shutil.copy2(
            source,
            destination,
        )

        return "copy"


def main():

    print("=" * 72)
    print(
        "PPE-SENTINEL PPE-AWARE DATASET BUILDER"
    )
    print("=" * 72)

    OUTPUT_IMAGES.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_LABELS.mkdir(
        parents=True,
        exist_ok=True,
    )

    images = sorted(
        image
        for image in SOURCE_IMAGES.iterdir()
        if image.is_file()
        and image.suffix.lower()
        in IMAGE_EXTENSIONS
    )

    print()
    print(
        f"Source training images: {len(images)}"
    )

    records = []

    for image in images:

        label_path = (
            SOURCE_LABELS
            / f"{image.stem}.txt"
        )

        if not label_path.exists():
            raise RuntimeError(
                f"Missing label: {label_path}"
            )

        ppe_classes = get_ppe_classes(
            label_path
        )

        if not ppe_classes:

            multiplier = 1

        else:

            multiplier = max(
                WEIGHTS[class_name]
                for class_name in ppe_classes
            )

        records.append(
            (
                image,
                label_path,
                ppe_classes,
                multiplier,
            )
        )

    # ------------------------------------------------------------
    # Convert fractional weights into deterministic integer
    # replication factors.
    #
    # The values below deliberately use modest replication:
    #
    # normal image       -> 1
    # gloves             -> 1
    # helmet             -> 2
    # safety vest        -> 2
    #
    # This avoids excessive oversampling.
    # ------------------------------------------------------------

    def replication_factor(
        ppe_classes
    ):

        if "safety-vest" in ppe_classes:
            return 2

        if "helmet" in ppe_classes:
            return 2

        if "gloves" in ppe_classes:
            return 1

        return 1

    total_entries = 0

    ppe_entry_counts = Counter()

    link_count = 0
    copy_count = 0

    for image, label, ppe_classes, multiplier in records:

        repetitions = replication_factor(
            ppe_classes
        )

        for repeat_index in range(
            repetitions
        ):

            # Unique filenames are required because
            # the same source image may appear multiple
            # times in the training directory.
            #
            # First copy keeps the original name.
            # Additional exposures receive a suffix.

            if repeat_index == 0:

                image_name = image.name
                label_name = label.name

            else:

                image_name = (
                    f"{image.stem}"
                    f"__ppe_repeat{repeat_index}"
                    f"{image.suffix}"
                )

                label_name = (
                    f"{label.stem}"
                    f"__ppe_repeat{repeat_index}"
                    f".txt"
                )

            destination_image = (
                OUTPUT_IMAGES
                / image_name
            )

            destination_label = (
                OUTPUT_LABELS
                / label_name
            )

            method = create_link_or_copy(
                image,
                destination_image,
            )

            create_link_or_copy(
                label,
                destination_label,
            )

            if method == "hardlink":
                link_count += 1
            else:
                copy_count += 1

            total_entries += 1

            for class_name in ppe_classes:
                ppe_entry_counts[
                    class_name
                ] += 1

    # ------------------------------------------------------------
    # Write dataset YAML.
    # ------------------------------------------------------------

    dataset_yaml = (
        ROOT
        / "configs"
        / "dataset_ppe_balanced.yaml"
    )

    yaml_text = f"""path: {OUTPUT_ROOT.as_posix()}

train: images

val: ../sh17/val/images

names:
  0: person
  1: ear
  2: ear-mufs
  3: face
  4: face-guard
  5: face-mask-medical
  6: foot
  7: tools
  8: glasses
  9: gloves
  10: helmet
  11: hands
  12: head
  13: medical-suit
  14: shoes
  15: safety-suit
  16: safety-vest
"""

    dataset_yaml.write_text(
        yaml_text,
        encoding="utf-8",
    )

    print()
    print(
        "=== DATASET RESULT ==="
    )

    print(
        f"Original entries : {len(records)}"
    )

    print(
        f"Balanced entries : {total_entries}"
    )

    print()
    print(
        "PPE exposure:"
    )

    for class_name in [
        "gloves",
        "helmet",
        "safety-vest",
    ]:

        original = sum(
            1
            for _, _, classes, _ in records
            if class_name in classes
        )

        sampled = ppe_entry_counts[
            class_name
        ]

        multiplier = (
            sampled / original
            if original
            else 0
        )

        print(
            f"{class_name:<15}"
            f"{original:>6} -> "
            f"{sampled:>6} "
            f"({multiplier:.2f}x)"
        )

    print()
    print(
        f"Hard links created: {link_count}"
    )

    print(
        f"Physical copies created: {copy_count}"
    )

    print()
    print(
        f"Dataset YAML:"
    )

    print(
        dataset_yaml
    )

    print()
    print(
        "Validation remains the original SH17 validation set."
    )


if __name__ == "__main__":
    main()
