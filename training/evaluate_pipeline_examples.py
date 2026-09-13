from pathlib import Path
from typing import Any

from inference.association import associate_workers
from inference.detector import PPEDetector
from inference.pipeline import build_worker_evidence


ROOT = Path(__file__).resolve().parents[1]

IMAGE_DIR = (
    ROOT
    / "data"
    / "processed"
    / "sh17"
    / "val"
    / "images"
)

LABEL_DIR = (
    ROOT
    / "data"
    / "processed"
    / "sh17"
    / "val"
    / "labels"
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


def ground_truth_classes(
    image_path: Path,
) -> set[str]:

    label_path = (
        LABEL_DIR
        / f"{image_path.stem}.txt"
    )

    classes = set()

    if not label_path.exists():
        return classes

    with label_path.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:

            parts = line.strip().split()

            if not parts:
                continue

            class_id = int(parts[0])

            if class_id in CLASS_NAMES:
                classes.add(
                    CLASS_NAMES[class_id]
                )

    return classes


def find_first(
    required_classes: set[str],
) -> Path | None:

    for image_path in sorted(
        IMAGE_DIR.glob("*")
    ):

        classes = ground_truth_classes(
            image_path
        )

        if required_classes.issubset(classes):
            return image_path

    return None


def print_pipeline_result(
    category: str,
    image_path: Path,
    detector: PPEDetector,
) -> None:

    print()
    print("=" * 70)
    print(category)
    print("=" * 70)

    print(
        f"Image: {image_path.name}"
    )

    truth = ground_truth_classes(
        image_path
    )

    print(
        "Ground-truth classes:"
    )

    print(
        ", ".join(sorted(truth))
    )

    detections = detector.predict(
        image_path
    )

    print()
    print(
        f"Model detections: {len(detections)}"
    )

    for detection in detections:

        print(
            f"  {detection['class_name']:18s}"
            f" {detection['confidence']:.3f}"
        )

    workers = associate_workers(
        detections
    )

    print()
    print(
        f"Workers detected: {len(workers)}"
    )

    for worker in workers:

        evidence = build_worker_evidence(
            worker,
            detections,
        )

        print()
        print(
            f"Worker {worker['worker_id']}"
        )

        for ppe_type, item in evidence[
            "ppe"
        ].items():

            print(
                f"  {ppe_type:15s}"
                f" {item['state']:8s}"
                f" confidence={item['confidence']:.3f}"
                f" source={item['source']}"
            )


def main():

    print("=" * 70)
    print("PPE-SENTINEL REAL VALIDATION PIPELINE")
    print("=" * 70)

    detector = PPEDetector()

    examples = [
        (
            "PERSON + HELMET",
            {"person", "helmet"},
        ),
        (
            "HANDS + GLOVES",
            {"hands", "gloves"},
        ),
        (
            "PERSON + SAFETY VEST",
            {"person", "safety-vest"},
        ),
        (
            "MULTIPLE PEOPLE",
            {"person"},
        ),
    ]

    used = set()

    for category, required in examples:

        image_path = find_first(
            required
        )

        if image_path is None:
            print(
                f"\nNo example found for {category}"
            )
            continue

        if image_path in used:
            continue

        used.add(image_path)

        print_pipeline_result(
            category,
            image_path,
            detector,
        )


if __name__ == "__main__":
    main()