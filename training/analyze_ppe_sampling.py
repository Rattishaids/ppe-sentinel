from pathlib import Path
from collections import Counter
import json
import random


ROOT = Path(__file__).resolve().parents[1]

TRAIN_IMAGES = (
    ROOT
    / "data"
    / "processed"
    / "sh17"
    / "train"
    / "images"
)

TRAIN_LABELS = (
    ROOT
    / "data"
    / "processed"
    / "sh17"
    / "train"
    / "labels"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "metrics"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "ppe_sampling_analysis.json"
)


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
    "helmet",
    "gloves",
    "safety-vest",
}


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
}


def load_image_classes():

    records = []

    images = sorted(
        image
        for image in TRAIN_IMAGES.iterdir()
        if image.is_file()
        and image.suffix.lower()
        in IMAGE_EXTENSIONS
    )

    for image in images:

        label_path = (
            TRAIN_LABELS
            / f"{image.stem}.txt"
        )

        if not label_path.exists():
            continue

        classes = []

        for line in label_path.read_text(
            encoding="utf-8"
        ).splitlines():

            parts = line.strip().split()

            if len(parts) < 5:
                continue

            class_id = int(parts[0])

            if class_id in CLASS_NAMES:
                classes.append(
                    CLASS_NAMES[class_id]
                )

        records.append(
            {
                "image": image.name,
                "classes": sorted(
                    set(classes)
                ),
                "ppe_classes": sorted(
                    {
                        c
                        for c in set(classes)
                        if c in PPE_CLASSES
                    }
                ),
            }
        )

    return records


def main():

    print("=" * 72)
    print(
        "PPE-SENTINEL TRAINING SAMPLING ANALYSIS"
    )
    print("=" * 72)

    records = load_image_classes()

    print()
    print(
        f"Training images: {len(records)}"
    )

    # ------------------------------------------------------------
    # Original image-level distribution
    # ------------------------------------------------------------

    image_counts = Counter()

    instance_counts = Counter()

    for record in records:

        for class_name in record[
            "classes"
        ]:

            image_counts[
                class_name
            ] += 1

    for record in records:

        label_path = (
            TRAIN_LABELS
            / Path(
                record["image"]
            ).with_suffix(".txt")
        )

        for line in label_path.read_text(
            encoding="utf-8"
        ).splitlines():

            parts = line.strip().split()

            if len(parts) < 5:
                continue

            class_id = int(parts[0])

            if class_id in CLASS_NAMES:

                instance_counts[
                    CLASS_NAMES[class_id]
                ] += 1

    # ------------------------------------------------------------
    # Sampling weights
    #
    # We deliberately use modest weights.
    #
    # safety-vest receives the strongest boost because
    # it has the fewest training images.
    # ------------------------------------------------------------

    class_weights = {
        "helmet": 1.75,
        "gloves": 1.35,
        "safety-vest": 2.50,
    }

    def image_weight(record):

        weight = 1.0

        for class_name in record[
            "ppe_classes"
        ]:

            weight = max(
                weight,
                class_weights[
                    class_name
                ],
            )

        return weight

    weighted_records = []

    for record in records:

        weighted_records.append(
            {
                **record,
                "sampling_weight": image_weight(
                    record
                ),
            }
        )

    # ------------------------------------------------------------
    # Simulate one effective epoch.
    #
    # IMPORTANT:
    # This does NOT modify the dataset.
    # It only tells us what the sampler would expose.
    # ------------------------------------------------------------

    seed = 0

    rng = random.Random(seed)

    population = list(
        range(
            len(weighted_records)
        )
    )

    weights = [
        record["sampling_weight"]
        for record in weighted_records
    ]

    sampled_indices = rng.choices(
        population,
        weights=weights,
        k=len(records),
    )

    sampled_image_counts = Counter()

    sampled_ppe_image_counts = Counter()

    for index in sampled_indices:

        record = weighted_records[
            index
        ]

        for class_name in record[
            "classes"
        ]:

            sampled_image_counts[
                class_name
            ] += 1

        for class_name in record[
            "ppe_classes"
        ]:

            sampled_ppe_image_counts[
                class_name
            ] += 1

    # ------------------------------------------------------------
    # Effective exposure ratios
    # ------------------------------------------------------------

    exposure = {}

    for class_name in sorted(
        PPE_CLASSES
    ):

        original = image_counts[
            class_name
        ]

        sampled = sampled_ppe_image_counts[
            class_name
        ]

        exposure[class_name] = {
            "original_images": original,
            "sampled_exposures": sampled,
            "original_fraction": (
                original
                / len(records)
            ),
            "sampled_fraction": (
                sampled
                / len(records)
            ),
            "exposure_multiplier": (
                sampled / original
                if original
                else 0.0
            ),
        }

    result = {
        "seed": seed,
        "training_images": len(
            records
        ),
        "class_weights": class_weights,
        "image_level_counts": dict(
            image_counts
        ),
        "instance_counts": dict(
            instance_counts
        ),
        "ppe_exposure": exposure,
        "sampling_method": (
            "weighted_with_replacement"
        ),
        "effective_epoch_size": len(
            records
        ),
    }

    OUTPUT_PATH.write_text(
        json.dumps(
            result,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ------------------------------------------------------------
    # Console report
    # ------------------------------------------------------------

    print()
    print(
        "ORIGINAL PPE IMAGE DISTRIBUTION"
    )

    print(
        "-" * 72
    )

    for class_name in sorted(
        PPE_CLASSES
    ):

        count = image_counts[
            class_name
        ]

        fraction = (
            count
            / len(records)
            * 100
        )

        print(
            f"{class_name:<15}"
            f"{count:>6} images "
            f"({fraction:>5.2f}%)"
        )

    print()
    print(
        "INSTANCE DISTRIBUTION"
    )

    print(
        "-" * 72
    )

    for class_name in sorted(
        PPE_CLASSES
    ):

        print(
            f"{class_name:<15}"
            f"{instance_counts[class_name]:>7}"
        )

    print()
    print(
        "SIMULATED PPE-AWARE EXPOSURE"
    )

    print(
        "-" * 72
    )

    print(
        f"{'Class':<15}"
        f"{'Original':>12}"
        f"{'Sampled':>12}"
        f"{'Multiplier':>14}"
    )

    print(
        "-" * 55
    )

    for class_name in sorted(
        PPE_CLASSES
    ):

        item = exposure[
            class_name
        ]

        print(
            f"{class_name:<15}"
            f"{item['original_images']:>12}"
            f"{item['sampled_exposures']:>12}"
            f"{item['exposure_multiplier']:>13.2f}x"
        )

    print()
    print(
        "Sampling manifest simulation saved:"
    )

    print(
        OUTPUT_PATH
    )

    print()
    print(
        "No dataset files were modified."
    )


if __name__ == "__main__":
    main()
