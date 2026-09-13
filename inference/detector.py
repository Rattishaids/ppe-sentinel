from pathlib import Path
from typing import Any

from ultralytics import RTDETR


ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    ROOT
    / "models"
    / "ppe_sentinel_rtdetr_l_best.pt"
)


class PPEDetector:
    """
    RT-DETR inference wrapper for PPE-Sentinel.

    The detector performs visual object detection only.
    Compliance decisions are handled by the evidence and
    reasoning layers.
    """

    def __init__(
        self,
        model_path: str | Path = MODEL_PATH,
        device: int | str = 0,
        confidence: float = 0.25,
        image_size: int = 640,
    ):
        self.model_path = Path(model_path)
        self.device = device
        self.confidence = confidence
        self.image_size = image_size

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found: {self.model_path}"
            )

        self.model = RTDETR(
            str(self.model_path)
        )

    def predict(
        self,
        image_path: str | Path,
    ) -> list[dict[str, Any]]:
        """
        Run RT-DETR and return structured detections.
        """

        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        results = self.model.predict(
            source=str(image_path),
            imgsz=self.image_size,
            conf=self.confidence,
            device=self.device,
            verbose=False,
        )

        if not results:
            return []

        result = results[0]

        if result.boxes is None:
            return []

        boxes = result.boxes

        detections = []

        for index in range(len(boxes)):

            xyxy = boxes.xyxy[
                index
            ].tolist()

            class_id = int(
                boxes.cls[index].item()
            )

            confidence = float(
                boxes.conf[index].item()
            )

            class_name = self.model.names[
                class_id
            ]

            detections.append(
                {
                    "detection_id": index,
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": round(
                        confidence,
                        4,
                    ),
                    "bbox": {
                        "x1": float(xyxy[0]),
                        "y1": float(xyxy[1]),
                        "x2": float(xyxy[2]),
                        "y2": float(xyxy[3]),
                    },
                }
            )

        return detections


if __name__ == "__main__":

    detector = PPEDetector()

    validation_dir = (
        ROOT
        / "data"
        / "processed"
        / "sh17"
        / "val"
        / "images"
    )

    image_path = next(
        validation_dir.glob("*")
    )

    detections = detector.predict(
        image_path
    )

    print("=" * 70)
    print(
        "PPE-SENTINEL DETECTOR TEST"
    )
    print("=" * 70)

    print(
        f"Model: {detector.model_path}"
    )

    print(
        f"Image: {image_path.name}"
    )

    print(
        f"Detections: {len(detections)}"
    )

    for detection in detections:
        print(detection)
