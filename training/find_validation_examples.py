from pathlib import Path
from collections import defaultdict


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


def classes_in_image(label_path: Path) -> set[str]:

    classes = set()

    with label_path.open("r", encoding="utf-8") as file:

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


def find_examples():

    examples = defaultdict(list)

    image_paths = sorted(
        IMAGE_DIR.glob("*")
    )

    for image_path in image_paths:

        label_path = (
            LABEL_DIR
            / f"{image_path.stem}.txt"
        )

        if not label_path.exists():
            continue

        classes = classes_in_image(
            label_path
        )

        # --------------------------------------------------------
        # Useful combinations
        # --------------------------------------------------------

        if {
            "person",
            "helmet",
        }.issubset(classes):

            examples["person_helmet"].append(
                image_path
            )

        if {
            "person",
            "safety-vest",
        }.issubset(classes):

            examples["person_vest"].append(
                image_path
            )

        if {
            "hands",
            "gloves",
        }.issubset(classes):

            examples["hands_gloves"].append(
                image_path
            )

        if classes.intersection(
            {"person"}
        ):

            examples["person"].append(
                image_path
            )

        # Multiple people are approximated from
        # the number of person annotations.
        person_count = 0

        with label_path.open(
            "r",
            encoding="utf-8",
        ) as file:

            for line in file:

                parts = line.strip().split()

                if parts and int(parts[0]) == 0:
                    person_count += 1

        if person_count >= 2:

            examples["multiple_people"].append(
                image_path
            )

    print("=" * 70)
    print("SH17 VALIDATION EXAMPLE FINDER")
    print("=" * 70)

    for category, paths in examples.items():

        print()
        print(
            f"{category}: {len(paths)} images"
        )

        for path in paths[:5]:
            print(
                f"  {path.name}"
            )


if __name__ == "__main__":
    find_examples()