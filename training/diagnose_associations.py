from pathlib import Path

from inference.association import (
    association_score,
    associate_workers,
)
from inference.detector import PPEDetector


ROOT = Path(__file__).resolve().parents[1]

IMAGE_DIR = (
    ROOT
    / "data"
    / "processed"
    / "sh17"
    / "val"
    / "images"
)


TEST_IMAGES = [
    "pexels-photo-10246146.jpeg",
    "pexels-photo-10383595.jpeg",
    "pexels-photo-11105404.jpeg",
    "pexels-photo-10040011.jpeg",
]


def print_relationship(
    subject,
    reference,
    label,
):

    geometry = association_score(
        subject,
        reference,
    )

    print()
    print(f"{label}")
    print("-" * 60)

    print(
        f"Subject:  "
        f"{subject['class_name']} "
        f"{subject['confidence']:.3f}"
    )

    print(
        f"Reference: "
        f"{reference['class_name']} "
        f"{reference['confidence']:.3f}"
    )

    print(
        f"IoU:             "
        f"{geometry['iou']:.4f}"
    )

    print(
        f"Containment:     "
        f"{geometry['containment']:.4f}"
    )

    print(
        f"Center inside:   "
        f"{geometry['center_inside']}"
    )

    print(
        f"Relative dx:     "
        f"{geometry['relative_position']['dx']:.4f}"
    )

    print(
        f"Relative dy:     "
        f"{geometry['relative_position']['dy']:.4f}"
    )


def find_class(
    detections,
    class_name,
):

    return [
        detection
        for detection in detections
        if detection["class_name"] == class_name
    ]


def main():

    detector = PPEDetector()

    print("=" * 70)
    print("PPE-SENTINEL ASSOCIATION DIAGNOSTICS")
    print("=" * 70)

    for image_name in TEST_IMAGES:

        image_path = IMAGE_DIR / image_name

        if not image_path.exists():
            continue

        print()
        print("=" * 70)
        print(f"IMAGE: {image_name}")
        print("=" * 70)

        detections = detector.predict(
            image_path
        )

        heads = find_class(
            detections,
            "head",
        )

        helmets = find_class(
            detections,
            "helmet",
        )

        persons = find_class(
            detections,
            "person",
        )

        vests = find_class(
            detections,
            "safety-vest",
        )

        hands = find_class(
            detections,
            "hands",
        )

        gloves = find_class(
            detections,
            "gloves",
        )

        print(
            f"persons={len(persons)}, "
            f"heads={len(heads)}, "
            f"helmets={len(helmets)}, "
            f"vests={len(vests)}, "
            f"hands={len(hands)}, "
            f"gloves={len(gloves)}"
        )

        # --------------------------------------------------------
        # Helmet ↔ head
        # --------------------------------------------------------

        for helmet in helmets:

            for head in heads:

                print_relationship(
                    helmet,
                    head,
                    "HELMET → HEAD",
                )

        # --------------------------------------------------------
        # Vest ↔ person
        # --------------------------------------------------------

        for vest in vests:

            for person in persons:

                print_relationship(
                    vest,
                    person,
                    "VEST → PERSON",
                )

        # --------------------------------------------------------
        # Gloves ↔ hands
        # --------------------------------------------------------

        for glove in gloves:

            for hand in hands:

                print_relationship(
                    glove,
                    hand,
                    "GLOVE → HAND",
                )


if __name__ == "__main__":
    main()