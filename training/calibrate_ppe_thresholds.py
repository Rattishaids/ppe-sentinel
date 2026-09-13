from pathlib import Path
from collections import defaultdict
import json

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

OUTPUT_DIR = ROOT / "outputs" / "metrics"
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

IOU_THRESHOLD = 0.50

THRESHOLDS = [
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
]

TARGET_CLASSES = {
    "helmet",
    "gloves",
    "safety-vest",
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


def xywh_to_xyxy(
    xc,
    yc,
    w,
    h,
    image_width,
    image_height,
):
    return {
        "x1": (xc - w / 2) * image_width,
        "y1": (yc - h / 2) * image_height,
        "x2": (xc + w / 2) * image_width,
        "y2": (yc + h / 2) * image_height,
    }


def area(b):
    return max(
        0.0,
        b["x2"] - b["x1"],
    ) * max(
        0.0,
        b["y2"] - b["y1"],
    )


def iou(a, b):
    x1 = max(a["x1"], b["x1"])
    y1 = max(a["y1"], b["y1"])
    x2 = min(a["x2"], b["x2"])
    y2 = min(a["y2"], b["y2"])

    inter = max(
        0.0,
        x2 - x1,
    ) * max(
        0.0,
        y2 - y1,
    )

    union = (
        area(a)
        + area(b)
        - inter
    )

    if union <= 0:
        return 0.0

    return inter / union


def load_labels(image_path):
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

    for line in label_path.read_text(
        encoding="utf-8"
    ).splitlines():

        parts = line.split()

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
                "class_id": class_id,
                "class_name": CLASS_NAMES[
                    class_id
                ],
                "bbox": bbox,
            }
        )

    return result


def evaluate_class(
    predictions,
    ground_truth,
    threshold,
    class_name,
):
    preds = [
        p
        for p in predictions
        if p["class_name"] == class_name
        and float(p["confidence"]) >= threshold
    ]

    gts = [
        g
        for g in ground_truth
        if g["class_name"] == class_name
    ]

    matched = set()

    tp = 0
    fp = 0
    fn = 0

    for prediction in sorted(
        preds,
        key=lambda x: float(
            x["confidence"]
        ),
        reverse=True,
    ):

        best_index = None
        best_iou = 0.0

        for index, gt in enumerate(gts):

            if index in matched:
                continue

            value = iou(
                prediction["bbox"],
                gt["bbox"],
            )

            if value > best_iou:
                best_iou = value
                best_index = index

        if (
            best_index is not None
            and best_iou >= IOU_THRESHOLD
        ):

            matched.add(best_index)
            tp += 1

        else:
            fp += 1

    fn = len(gts) - len(matched)

    precision = (
        tp / (tp + fp)
        if tp + fp > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "ground_truth": len(gts),
        "predictions": len(preds),
    }


def main():

    print("=" * 72)
    print(
        "PPE-SENTINEL CONFIDENCE THRESHOLD CALIBRATION"
    )
    print("=" * 72)

    print()
    print(
        "Model:",
        MODEL_PATH,
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

    detector = PPEDetector(
        model_path=MODEL_PATH,
        confidence=0.05,
        image_size=640,
    )

    print()
    print(
        f"Scanning {len(images)} validation images..."
    )

    all_data = []

    for index, image_path in enumerate(
        images,
        start=1,
    ):

        if index % 200 == 0:
            print(
                f"Processed "
                f"{index}/{len(images)}"
            )

        try:

            predictions = detector.predict(
                image_path
            )

            ground_truth = load_labels(
                image_path
            )

            all_data.append(
                {
                    "image": image_path.name,
                    "predictions": predictions,
                    "ground_truth": ground_truth,
                }
            )

        except Exception as exc:

            print(
                f"ERROR "
                f"{image_path.name}: "
                f"{exc}"
            )

    results = {}

    for threshold in THRESHOLDS:

        threshold_key = (
            f"{threshold:.2f}"
        )

        results[
            threshold_key
        ] = {}

        for class_name in TARGET_CLASSES:

            class_predictions = []

            for item in all_data:

                class_predictions.extend(
                    [
                        p
                        for p in item[
                            "predictions"
                        ]
                        if float(
                            p["confidence"]
                        ) >= threshold
                    ]
                )

            class_ground_truth = []

            for item in all_data:

                class_ground_truth.extend(
                    [
                        g
                        for g in item[
                            "ground_truth"
                        ]
                    ]
                )

            results[
                threshold_key
            ][class_name] = evaluate_class(
                class_predictions,
                class_ground_truth,
                threshold,
                class_name,
            )

        # Overall PPE micro metrics
        total_tp = sum(
            results[
                threshold_key
            ][c]["tp"]
            for c in TARGET_CLASSES
        )

        total_fp = sum(
            results[
                threshold_key
            ][c]["fp"]
            for c in TARGET_CLASSES
        )

        total_fn = sum(
            results[
                threshold_key
            ][c]["fn"]
            for c in TARGET_CLASSES
        )

        precision = (
            total_tp
            / (total_tp + total_fp)
            if total_tp + total_fp
            else 0.0
        )

        recall = (
            total_tp
            / (total_tp + total_fn)
            if total_tp + total_fn
            else 0.0
        )

        f1 = (
            2 * precision * recall
            / (precision + recall)
            if precision + recall
            else 0.0
        )

        results[
            threshold_key
        ]["overall"] = {
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    output = {
        "model": str(
            MODEL_PATH.relative_to(ROOT)
        ),
        "iou_threshold": IOU_THRESHOLD,
        "thresholds": THRESHOLDS,
        "target_classes": sorted(
            TARGET_CLASSES
        ),
        "results": results,
    }

    output_path = (
        OUTPUT_DIR
        / "ppe_threshold_calibration.json"
    )

    output_path.write_text(
        json.dumps(
            output,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print(
        "CALIBRATION RESULTS"
    )
    print("=" * 72)

    print()

    print(
        f"{'THR':<7}"
        f"{'PREC':<10}"
        f"{'RECALL':<10}"
        f"{'F1':<10}"
        f"{'TP':<8}"
        f"{'FP':<8}"
        f"{'FN':<8}"
    )

    print("-" * 61)

    for threshold in THRESHOLDS:

        key = f"{threshold:.2f}"

        result = results[
            key
        ]["overall"]

        print(
            f"{threshold:<7.2f}"
            f"{result['precision']:<10.4f}"
            f"{result['recall']:<10.4f}"
            f"{result['f1']:<10.4f}"
            f"{result['tp']:<8}"
            f"{result['fp']:<8}"
            f"{result['fn']:<8}"
        )

    print()
    print(
        "BEST THRESHOLD BY PPE F1"
    )

    best_key = max(
        results,
        key=lambda key:
            results[key][
                "overall"
            ]["f1"],
    )

    print(
        f"Threshold: {best_key}"
    )

    print(
        f"F1: "
        f"{results[best_key]['overall']['f1']:.4f}"
    )

    print(
        f"Precision: "
        f"{results[best_key]['overall']['precision']:.4f}"
    )

    print(
        f"Recall: "
        f"{results[best_key]['overall']['recall']:.4f}"
    )

    print()
    print(
        "PER-CLASS BEST F1"
    )

    for class_name in sorted(
        TARGET_CLASSES
    ):

        best = max(
            results,
            key=lambda key:
                results[key][
                    class_name
                ]["f1"],
        )

        metrics = results[
            best
        ][class_name]

        print(
            f"{class_name:<15}"
            f"threshold={best} "
            f"precision={metrics['precision']:.4f} "
            f"recall={metrics['recall']:.4f} "
            f"F1={metrics['f1']:.4f}"
        )

    print()
    print(
        "Saved:"
    )

    print(
        output_path
    )


if __name__ == "__main__":
    main()
