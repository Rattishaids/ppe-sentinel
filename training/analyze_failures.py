from pathlib import Path
from collections import Counter
import json
import math

from inference.detector import PPEDetector


ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    ROOT
    / "outputs"
    / "training"
    / "rtdetr_l_batch4_50e"
    / "weights"
    / "best.pt"
)

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

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "failures"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

REPORT_PATH = (
    OUTPUT_DIR
    / "failure_candidates.json"
)

IOU_THRESHOLD = 0.50

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

SMALL_OBJECT_CLASSES = {
    "helmet",
    "gloves",
    "glasses",
    "shoes",
    "foot",
    "ear-mufs",
}


def xywh_to_xyxy(
    x_center,
    y_center,
    width,
    height,
    image_width,
    image_height,
):
    x1 = (
        x_center - width / 2
    ) * image_width

    y1 = (
        y_center - height / 2
    ) * image_height

    x2 = (
        x_center + width / 2
    ) * image_width

    y2 = (
        y_center + height / 2
    ) * image_height

    return {
        "x1": max(0.0, x1),
        "y1": max(0.0, y1),
        "x2": min(float(image_width), x2),
        "y2": min(float(image_height), y2),
    }


def bbox_area(bbox):
    return max(
        0.0,
        bbox["x2"] - bbox["x1"],
    ) * max(
        0.0,
        bbox["y2"] - bbox["y1"],
    )


def intersection_over_union(
    a,
    b,
):
    x1 = max(
        a["x1"],
        b["x1"],
    )

    y1 = max(
        a["y1"],
        b["y1"],
    )

    x2 = min(
        a["x2"],
        b["x2"],
    )

    y2 = min(
        a["y2"],
        b["y2"],
    )

    intersection = max(
        0.0,
        x2 - x1,
    ) * max(
        0.0,
        y2 - y1,
    )

    union = (
        bbox_area(a)
        + bbox_area(b)
        - intersection
    )

    if union <= 0:
        return 0.0

    return intersection / union


def center_distance(
    a,
    b,
):
    ax = (
        a["x1"] + a["x2"]
    ) / 2

    ay = (
        a["y1"] + a["y2"]
    ) / 2

    bx = (
        b["x1"] + b["x2"]
    ) / 2

    by = (
        b["y1"] + b["y2"]
    ) / 2

    return math.sqrt(
        (ax - bx) ** 2
        + (ay - by) ** 2
    )


def load_ground_truth(
    image_path,
):
    label_path = (
        LABEL_DIR
        / f"{image_path.stem}.txt"
    )

    if not label_path.exists():
        return []

    try:
        import cv2

        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            return []

        image_height, image_width = (
            image.shape[:2]
        )

    except Exception:
        return []

    ground_truth = []

    lines = label_path.read_text(
        encoding="utf-8"
    ).splitlines()

    for index, line in enumerate(lines):

        parts = line.strip().split()

        if len(parts) < 5:
            continue

        class_id = int(parts[0])

        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])

        class_name = CLASS_NAMES.get(
            class_id,
            f"class_{class_id}",
        )

        bbox = xywh_to_xyxy(
            x_center,
            y_center,
            width,
            height,
            image_width,
            image_height,
        )

        ground_truth.append(
            {
                "gt_id": index,
                "class_id": class_id,
                "class_name": class_name,
                "bbox": bbox,
                "area_ratio": (
                    bbox_area(bbox)
                    / (
                        image_width
                        * image_height
                    )
                ),
            }
        )

    return ground_truth


def find_best_match(
    prediction,
    ground_truth,
    used_gt,
):
    best_index = None
    best_iou = 0.0

    for index, gt in enumerate(
        ground_truth
    ):

        if index in used_gt:
            continue

        iou = intersection_over_union(
            prediction["bbox"],
            gt["bbox"],
        )

        if iou > best_iou:
            best_iou = iou
            best_index = index

    return best_index, best_iou


def analyze_image(
    detector,
    image_path,
):
    predictions = detector.predict(
        image_path
    )

    ground_truth = load_ground_truth(
        image_path
    )

    used_gt = set()

    true_positives = []
    false_positives = []
    false_negatives = []
    class_confusions = []

    # ============================================================
    # Match predictions to ground truth
    # ============================================================

    for prediction in predictions:

        gt_index, best_iou = (
            find_best_match(
                prediction,
                ground_truth,
                used_gt,
            )
        )

        if (
            gt_index is None
            or best_iou < IOU_THRESHOLD
        ):

            false_positives.append(
                {
                    "prediction": prediction,
                    "best_iou": round(
                        best_iou,
                        4,
                    ),
                }
            )

            continue

        gt = ground_truth[
            gt_index
        ]

        used_gt.add(
            gt_index
        )

        if (
            prediction[
                "class_name"
            ]
            == gt["class_name"]
        ):

            true_positives.append(
                {
                    "prediction": prediction,
                    "ground_truth": gt,
                    "iou": round(
                        best_iou,
                        4,
                    ),
                }
            )

        else:

            class_confusions.append(
                {
                    "prediction": prediction,
                    "ground_truth": gt,
                    "iou": round(
                        best_iou,
                        4,
                    ),
                }
            )

    # ============================================================
    # Remaining GT objects = false negatives
    # ============================================================

    for index, gt in enumerate(
        ground_truth
    ):

        if index not in used_gt:

            false_negatives.append(
                gt
            )

    # ============================================================
    # Failure categories
    # ============================================================

    reasons = []

    ppe_false_negatives = [
        gt
        for gt in false_negatives
        if gt["class_name"]
        in PPE_CLASSES
    ]

    ppe_false_positives = [
        item
        for item in false_positives
        if item["prediction"][
            "class_name"
        ] in PPE_CLASSES
    ]

    small_object_failures = [
        gt
        for gt in false_negatives
        if gt["class_name"]
        in SMALL_OBJECT_CLASSES
        and gt["area_ratio"] < 0.01
    ]

    if false_negatives:
        reasons.append(
            "false_negative"
        )

    if false_positives:
        reasons.append(
            "false_positive"
        )

    if class_confusions:
        reasons.append(
            "class_confusion"
        )

    if ppe_false_negatives:
        reasons.append(
            "ppe_false_negative"
        )

    if ppe_false_positives:
        reasons.append(
            "ppe_false_positive"
        )

    if small_object_failures:
        reasons.append(
            "small_object_failure"
        )

    person_count = sum(
        1
        for gt in ground_truth
        if gt["class_name"]
        == "person"
    )

    if person_count >= 2:
        reasons.append(
            "multiple_worker_scene"
        )

    safety_suit_vest_confusions = [
        item
        for item in class_confusions
        if {
            item["prediction"][
                "class_name"
            ],
            item["ground_truth"][
                "class_name"
            ],
        }
        == {
            "safety-suit",
            "safety-vest",
        }
    ]

    if safety_suit_vest_confusions:
        reasons.append(
            "safety_suit_safety_vest_confusion"
        )

    weak_ppe_predictions = [
        item["prediction"]
        for item in predictions
        if item["class_name"]
        in PPE_CLASSES
        and float(
            item["confidence"]
        ) < 0.50
    ]

    if weak_ppe_predictions:
        reasons.append(
            "weak_ppe_confidence"
        )

    # ============================================================
    # Severity score
    # ============================================================

    severity = 0

    severity += (
        len(false_negatives)
        * 5
    )

    severity += (
        len(class_confusions)
        * 6
    )

    severity += (
        len(ppe_false_negatives)
        * 10
    )

    severity += (
        len(ppe_false_positives)
        * 8
    )

    severity += (
        len(small_object_failures)
        * 8
    )

    severity += (
        len(safety_suit_vest_confusions)
        * 12
    )

    severity += (
        len(false_positives)
        * 2
    )

    return {
        "image": str(
            image_path.relative_to(
                ROOT
            )
        ),
        "ground_truth_count": len(
            ground_truth
        ),
        "prediction_count": len(
            predictions
        ),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "class_confusions": (
            class_confusions
        ),
        "ppe_false_negatives": (
            ppe_false_negatives
        ),
        "ppe_false_positives": (
            ppe_false_positives
        ),
        "small_object_failures": (
            small_object_failures
        ),
        "safety_suit_vest_confusions": (
            safety_suit_vest_confusions
        ),
        "weak_ppe_predictions": (
            weak_ppe_predictions
        ),
        "person_count": person_count,
        "candidate_reasons": list(
            dict.fromkeys(reasons)
        ),
        "severity_score": severity,
    }


def main():

    print("=" * 70)
    print(
        "PPE-SENTINEL GROUND-TRUTH FAILURE MINING"
    )
    print("=" * 70)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    if not IMAGE_DIR.exists():
        raise FileNotFoundError(
            f"Image directory not found: {IMAGE_DIR}"
        )

    if not LABEL_DIR.exists():
        raise FileNotFoundError(
            f"Label directory not found: {LABEL_DIR}"
        )

    images = sorted(
        image
        for image in IMAGE_DIR.iterdir()
        if image.is_file()
        and image.suffix.lower()
        in {
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".webp",
        }
    )

    print()
    print(
        f"Validation images: {len(images)}"
    )

    print()
    print(
        "Loading RT-DETR..."
    )

    detector = PPEDetector(
        model_path=MODEL_PATH,
        confidence=0.25,
        image_size=640,
    )

    results = []

    for index, image_path in enumerate(
        images,
        start=1,
    ):

        if index % 50 == 0:
            print(
                f"Processed "
                f"{index}/{len(images)}"
            )

        try:

            result = analyze_image(
                detector,
                image_path,
            )

            if result[
                "candidate_reasons"
            ]:

                results.append(
                    result
                )

        except Exception as exc:

            print(
                f"ERROR: "
                f"{image_path.name}: "
                f"{exc}"
            )

    results.sort(
        key=lambda item: (
            -item[
                "severity_score"
            ],
            -len(
                item[
                    "ppe_false_negatives"
                ]
            ),
            -len(
                item[
                    "class_confusions"
                ]
            ),
            item["image"],
        )
    )

    reason_counter = Counter()

    for result in results:

        for reason in result[
            "candidate_reasons"
        ]:

            reason_counter[
                reason
            ] += 1

    report = {
        "model": str(
            MODEL_PATH.relative_to(
                ROOT
            )
        ),
        "iou_threshold": IOU_THRESHOLD,
        "validation_images": len(
            images
        ),
        "candidate_images": len(
            results
        ),
        "candidate_reason_counts": dict(
            reason_counter
        ),
        "candidates": results,
    }

    REPORT_PATH.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 70)
    print(
        "GROUND-TRUTH FAILURE MINING COMPLETE"
    )
    print("=" * 70)

    print()
    print(
        f"Images scanned: {len(images)}"
    )

    print(
        f"Actual candidate images: "
        f"{len(results)}"
    )

    print()
    print(
        "Failure categories:"
    )

    for reason, count in (
        reason_counter.most_common()
    ):

        print(
            f"  {reason}: {count}"
        )

    print()
    print(
        f"Report saved to:"
    )

    print(
        REPORT_PATH
    )

    print()
    print(
        "TOP 15 HIGH-SEVERITY CASES"
    )

    print(
        "-" * 70
    )

    for index, result in enumerate(
        results[:15],
        start=1,
    ):

        print()
        print(
            f"{index}. "
            f"{Path(result['image']).name}"
        )

        print(
            f"   Severity: "
            f"{result['severity_score']}"
        )

        print(
            f"   GT objects: "
            f"{result['ground_truth_count']}"
        )

        print(
            f"   Predictions: "
            f"{result['prediction_count']}"
        )

        print(
            "   Reasons: "
            + ", ".join(
                result[
                    "candidate_reasons"
                ]
            )
        )

        if result[
            "ppe_false_negatives"
        ]:

            print(
                "   PPE false negatives:"
            )

            for item in result[
                "ppe_false_negatives"
            ]:

                print(
                    f"      "
                    f"{item['class_name']}"
                )

        if result[
            "class_confusions"
        ]:

            print(
                "   Class confusions:"
            )

            for item in result[
                "class_confusions"
            ]:

                print(
                    "      "
                    f"GT={item['ground_truth']['class_name']} "
                    f"Pred={item['prediction']['class_name']} "
                    f"IoU={item['iou']}"
                )

        if result[
            "small_object_failures"
        ]:

            print(
                "   Small-object misses:"
            )

            for item in result[
                "small_object_failures"
            ][:5]:

                print(
                    "      "
                    f"{item['class_name']} "
                    f"area={item['area_ratio']:.6f}"
                )


if __name__ == "__main__":
    main()
