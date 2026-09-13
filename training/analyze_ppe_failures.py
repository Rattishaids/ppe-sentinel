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
    / "ppe_failure_analysis.json"
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

SMALL_PPE_CLASSES = {
    "helmet",
    "gloves",
}


def xywh_to_xyxy(
    xc,
    yc,
    w,
    h,
    image_width,
    image_height,
):
    return {
        "x1": max(
            0.0,
            (xc - w / 2) * image_width,
        ),
        "y1": max(
            0.0,
            (yc - h / 2) * image_height,
        ),
        "x2": min(
            float(image_width),
            (xc + w / 2) * image_width,
        ),
        "y2": min(
            float(image_height),
            (yc + h / 2) * image_height,
        ),
    }


def area(bbox):
    return max(
        0.0,
        bbox["x2"] - bbox["x1"],
    ) * max(
        0.0,
        bbox["y2"] - bbox["y1"],
    )


def iou(a, b):
    x1 = max(a["x1"], b["x1"])
    y1 = max(a["y1"], b["y1"])
    x2 = min(a["x2"], b["x2"])
    y2 = min(a["y2"], b["y2"])

    intersection = (
        max(0.0, x2 - x1)
        * max(0.0, y2 - y1)
    )

    union = (
        area(a)
        + area(b)
        - intersection
    )

    if union <= 0:
        return 0.0

    return intersection / union


def load_ground_truth(
    image_path,
):
    label_path = (
        LABEL_DIR
        / f"{image_path.stem}.txt"
    )

    if not label_path.exists():
        return []

    import cv2

    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        return []

    height, width = image.shape[:2]

    result = []

    for index, line in enumerate(
        label_path.read_text(
            encoding="utf-8"
        ).splitlines()
    ):

        parts = line.strip().split()

        if len(parts) < 5:
            continue

        class_id = int(parts[0])

        bbox = xywh_to_xyxy(
            float(parts[1]),
            float(parts[2]),
            float(parts[3]),
            float(parts[4]),
            width,
            height,
        )

        result.append(
            {
                "gt_id": index,
                "class_id": class_id,
                "class_name": CLASS_NAMES[
                    class_id
                ],
                "bbox": bbox,
                "area_ratio": (
                    area(bbox)
                    / (width * height)
                ),
            }
        )

    return result, width, height


def best_same_class_match(
    prediction,
    ground_truth,
    used,
):
    best = None
    best_iou = 0.0

    for index, gt in enumerate(
        ground_truth
    ):

        if index in used:
            continue

        if (
            gt["class_name"]
            != prediction["class_name"]
        ):
            continue

        value = iou(
            prediction["bbox"],
            gt["bbox"],
        )

        if value > best_iou:
            best_iou = value
            best = index

    return best, best_iou


def best_any_class_match(
    prediction,
    ground_truth,
    used,
):
    best = None
    best_iou = 0.0

    for index, gt in enumerate(
        ground_truth
    ):

        if index in used:
            continue

        value = iou(
            prediction["bbox"],
            gt["bbox"],
        )

        if value > best_iou:
            best_iou = value
            best = index

    return best, best_iou


def analyze_image(
    detector,
    image_path,
):
    loaded = load_ground_truth(
        image_path
    )

    if not loaded:
        return None

    ground_truth, width, height = loaded

    predictions = detector.predict(
        image_path
    )

    gt_ppe = [
        gt
        for gt in ground_truth
        if gt["class_name"]
        in PPE_CLASSES
    ]

    pred_ppe = [
        prediction
        for prediction in predictions
        if prediction["class_name"]
        in PPE_CLASSES
    ]

    persons = [
        gt
        for gt in ground_truth
        if gt["class_name"]
        == "person"
    ]

    used_gt = set()

    matched_ppe = []
    missed_ppe = []
    ppe_false_positives = []
    ppe_confusions = []

    # ------------------------------------------------------------
    # Match PPE predictions
    # ------------------------------------------------------------

    for prediction in pred_ppe:

        same_index, same_iou = (
            best_same_class_match(
                prediction,
                ground_truth,
                used_gt,
            )
        )

        if (
            same_index is not None
            and same_iou >= IOU_THRESHOLD
        ):

            used_gt.add(
                same_index
            )

            matched_ppe.append(
                {
                    "class": prediction[
                        "class_name"
                    ],
                    "confidence": round(
                        float(
                            prediction[
                                "confidence"
                            ]
                        ),
                        4,
                    ),
                    "iou": round(
                        same_iou,
                        4,
                    ),
                    "ground_truth": ground_truth[
                        same_index
                    ],
                }
            )

            continue

        any_index, any_iou = (
            best_any_class_match(
                prediction,
                ground_truth,
                used_gt,
            )
        )

        if (
            any_index is not None
            and any_iou >= IOU_THRESHOLD
        ):

            gt = ground_truth[
                any_index
            ]

            if (
                gt["class_name"]
                in PPE_CLASSES
            ):

                used_gt.add(
                    any_index
                )

                ppe_confusions.append(
                    {
                        "ground_truth": gt[
                            "class_name"
                        ],
                        "prediction": prediction[
                            "class_name"
                        ],
                        "confidence": round(
                            float(
                                prediction[
                                    "confidence"
                                ]
                            ),
                            4,
                        ),
                        "iou": round(
                            any_iou,
                            4,
                        ),
                    }
                )

                continue

        ppe_false_positives.append(
            {
                "class": prediction[
                    "class_name"
                ],
                "confidence": round(
                    float(
                        prediction[
                            "confidence"
                        ]
                    ),
                    4,
                ),
                "bbox": prediction[
                    "bbox"
                ],
            }
        )

    # ------------------------------------------------------------
    # Missed PPE ground truth
    # ------------------------------------------------------------

    for index, gt in enumerate(
        ground_truth
    ):

        if (
            gt["class_name"]
            in PPE_CLASSES
            and index not in used_gt
        ):

            missed_ppe.append(
                gt
            )

    # ------------------------------------------------------------
    # Failure reasons
    # ------------------------------------------------------------

    reasons = []

    if missed_ppe:
        reasons.append(
            "ppe_false_negative"
        )

    if ppe_false_positives:
        reasons.append(
            "ppe_false_positive"
        )

    if ppe_confusions:
        reasons.append(
            "ppe_class_confusion"
        )

    small_misses = [
        item
        for item in missed_ppe
        if item["class_name"]
        in SMALL_PPE_CLASSES
        and item["area_ratio"] < 0.01
    ]

    if small_misses:
        reasons.append(
            "small_ppe_failure"
        )

    weak_predictions = [
        item
        for item in pred_ppe
        if float(
            item["confidence"]
        ) < 0.50
    ]

    if weak_predictions:
        reasons.append(
            "weak_ppe_confidence"
        )

    if len(persons) >= 2:
        reasons.append(
            "multi_worker_scene"
        )

    suit_vest_confusions = [
        item
        for item in ppe_confusions
        if {
            item["ground_truth"],
            item["prediction"],
        }
        == {
            "safety-suit",
            "safety-vest",
        }
    ]

    # ------------------------------------------------------------
    # Severity designed for PPE relevance
    # ------------------------------------------------------------

    severity = 0

    severity += (
        len(missed_ppe) * 25
    )

    severity += (
        len(ppe_false_positives) * 15
    )

    severity += (
        len(ppe_confusions) * 30
    )

    severity += (
        len(small_misses) * 20
    )

    severity += (
        len(suit_vest_confusions) * 35
    )

    # Prefer cases with multiple distinct
    # PPE failure mechanisms.
    severity += (
        max(
            0,
            len(reasons) - 1,
        )
        * 5
    )

    return {
        "image": image_path.name,
        "width": width,
        "height": height,
        "person_count": len(persons),
        "ground_truth_ppe_count": len(
            gt_ppe
        ),
        "predicted_ppe_count": len(
            pred_ppe
        ),
        "matched_ppe": matched_ppe,
        "missed_ppe": missed_ppe,
        "ppe_false_positives": (
            ppe_false_positives
        ),
        "ppe_class_confusions": (
            ppe_confusions
        ),
        "small_ppe_misses": (
            small_misses
        ),
        "weak_ppe_predictions": (
            weak_predictions
        ),
        "safety_suit_vest_confusions": (
            suit_vest_confusions
        ),
        "candidate_reasons": reasons,
        "severity_score": severity,
    }


def main():

    print("=" * 72)
    print(
        "PPE-SENTINEL PPE-FOCUSED FAILURE ANALYSIS"
    )
    print("=" * 72)

    detector = PPEDetector(
        model_path=MODEL_PATH,
        confidence=0.25,
        image_size=640,
    )

    images = sorted(
        image
        for image in IMAGE_DIR.iterdir()
        if image.suffix.lower()
        in {
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".webp",
        }
    )

    results = []

    for index, image_path in enumerate(
        images,
        start=1,
    ):

        if index % 100 == 0:
            print(
                f"Processed "
                f"{index}/{len(images)}"
            )

        try:
            result = analyze_image(
                detector,
                image_path,
            )

            if (
                result is not None
                and result[
                    "candidate_reasons"
                ]
            ):
                results.append(
                    result
                )

        except Exception as exc:
            print(
                f"ERROR "
                f"{image_path.name}: "
                f"{exc}"
            )

    results.sort(
        key=lambda x: (
            -x["severity_score"],
            -len(
                x["missed_ppe"]
            ),
            -len(
                x["ppe_class_confusions"]
            ),
        )
    )

    category_counts = Counter()

    for result in results:
        for reason in result[
            "candidate_reasons"
        ]:
            category_counts[
                reason
            ] += 1

    # ------------------------------------------------------------
    # Select diverse representative cases
    # ------------------------------------------------------------

    selected = []
    selected_images = set()
    selected_categories = set()

    preferred_categories = [
        "safety_vest_failure",
        "helmet_failure",
        "gloves_failure",
        "small_ppe_failure",
        "ppe_class_confusion",
        "multi_worker_scene",
    ]

    def category_for(result):

        missed = {
            item["class_name"]
            for item in result[
                "missed_ppe"
            ]
        }

        if (
            "safety-vest"
            in missed
        ):
            return "safety_vest_failure"

        if (
            "helmet"
            in missed
        ):
            return "helmet_failure"

        if (
            "gloves"
            in missed
        ):
            return "gloves_failure"

        if result[
            "small_ppe_misses"
        ]:
            return "small_ppe_failure"

        if result[
            "ppe_class_confusions"
        ]:
            return "ppe_class_confusion"

        if (
            result["person_count"]
            >= 2
        ):
            return "multi_worker_scene"

        return None

    for preferred in preferred_categories:

        for result in results:

            category = category_for(
                result
            )

            if (
                category == preferred
                and result["image"]
                not in selected_images
            ):

                selected.append(
                    {
                        "category": category,
                        "image": result,
                    }
                )

                selected_images.add(
                    result["image"]
                )

                break

    report = {
        "model": str(
            MODEL_PATH.relative_to(
                ROOT
            )
        ),
        "iou_threshold": IOU_THRESHOLD,
        "images_scanned": len(
            images
        ),
        "ppe_candidate_images": len(
            results
        ),
        "category_counts": dict(
            category_counts
        ),
        "representative_cases": selected,
        "all_candidates": results,
    }

    REPORT_PATH.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print(
        "PPE FAILURE ANALYSIS COMPLETE"
    )
    print("=" * 72)

    print()
    print(
        f"Images scanned: {len(images)}"
    )

    print(
        f"PPE candidate images: "
        f"{len(results)}"
    )

    print()
    print(
        "PPE failure categories:"
    )

    for category, count in (
        category_counts.most_common()
    ):
        print(
            f"  {category}: {count}"
        )

    print()
    print(
        "REPRESENTATIVE CASES"
    )

    print(
        "-" * 72
    )

    for index, item in enumerate(
        selected,
        start=1,
    ):

        result = item["image"]

        print()
        print(
            f"{index}. "
            f"{item['category']}"
        )

        print(
            f"   Image: "
            f"{result['image']}"
        )

        print(
            f"   Severity: "
            f"{result['severity_score']}"
        )

        print(
            f"   Workers: "
            f"{result['person_count']}"
        )

        if result[
            "missed_ppe"
        ]:

            print(
                "   Missed PPE:"
            )

            for gt in result[
                "missed_ppe"
            ]:

                print(
                    f"      "
                    f"{gt['class_name']} "
                    f"area="
                    f"{gt['area_ratio']:.6f}"
                )

        if result[
            "ppe_class_confusions"
        ]:

            print(
                "   PPE confusions:"
            )

            for confusion in result[
                "ppe_class_confusions"
            ]:

                print(
                    "      "
                    f"GT={confusion['ground_truth']} "
                    f"Pred={confusion['prediction']} "
                    f"IoU={confusion['iou']} "
                    f"conf="
                    f"{confusion['confidence']}"
                )

        if result[
            "ppe_false_positives"
        ]:

            print(
                "   PPE false positives: "
                f"{len(result['ppe_false_positives'])}"
            )

    print()
    print(
        "Saved report:"
    )

    print(
        REPORT_PATH
    )


if __name__ == "__main__":
    main()
