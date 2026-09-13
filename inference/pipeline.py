from pathlib import Path
from typing import Any

from inference.association import (
    associate_worker_components,
    associate_workers,
)

from inference.detector import PPEDetector

from inference.evidence import (
    ASSOCIATION_THRESHOLD,
    confidence_from_association,
    visibility_gate,
)

from inference.reasoning import reason


ROOT = Path(__file__).resolve().parents[1]


def build_worker_evidence(
    worker: dict[str, Any],
    detections: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Convert worker-level spatial associations into
    deterministic PPE evidence.

    Decision policy:

        Strong association
            -> PRESENT

        Weak association
            -> UNKNOWN

        PPE not detected + visible relevant region
            -> ABSENT

        Relevant region unavailable
            -> UNKNOWN
    """

    associated = associate_worker_components(
        worker,
        detections,
    )

    components = associated["components"]

    result = {}

    # ============================================================
    # HELMET
    # ============================================================

    helmet = components["helmet"]

    helmet_visibility = visibility_gate(
        worker,
        detections,
        "helmet",
    )

    if helmet is not None:

        score = confidence_from_association(
            helmet
        )

        if (
            helmet.get("evidence_valid", False)
            and score >= ASSOCIATION_THRESHOLD
        ):

            result["helmet"] = {
                "state": "PRESENT",
                "confidence": round(
                    float(score),
                    3,
                ),
                "source": "spatial_association",
                "details": helmet,
                "visibility": helmet_visibility,
            }

        else:

            result["helmet"] = {
                "state": "UNKNOWN",
                "confidence": round(
                    float(score),
                    3,
                ),
                "source": "weak_association",
                "details": helmet,
                "visibility": helmet_visibility,
            }

    else:

        if not helmet_visibility["visible"]:

            result["helmet"] = {
                "state": "UNKNOWN",
                "confidence": round(
                    float(
                        helmet_visibility[
                            "confidence"
                        ]
                    ),
                    3,
                ),
                "source": helmet_visibility[
                    "source"
                ],
                "details": {
                    "reason": (
                        "No sufficiently reliable "
                        "head evidence was available."
                    )
                },
                "visibility": helmet_visibility,
            }

        else:

            result["helmet"] = {
                "state": "ABSENT",
                "confidence": 0.0,
                "source": "head_without_helmet",
                "details": {
                    "reason": (
                        "Head evidence is visible, "
                        "but no sufficiently strong "
                        "helmet association was detected."
                    )
                },
                "visibility": helmet_visibility,
            }

    # ============================================================
    # SAFETY VEST
    # ============================================================

    vest = components["safety_vest"]

    vest_visibility = visibility_gate(
        worker,
        detections,
        "safety_vest",
    )

    if vest is not None:

        score = confidence_from_association(
            vest
        )

        if (
            vest.get("evidence_valid", False)
            and score >= ASSOCIATION_THRESHOLD
        ):

            result["safety_vest"] = {
                "state": "PRESENT",
                "confidence": round(
                    float(score),
                    3,
                ),
                "source": "spatial_association",
                "details": vest,
                "visibility": vest_visibility,
            }

        else:

            result["safety_vest"] = {
                "state": "UNKNOWN",
                "confidence": round(
                    float(score),
                    3,
                ),
                "source": "weak_association",
                "details": vest,
                "visibility": vest_visibility,
            }

    else:

        if not vest_visibility["visible"]:

            result["safety_vest"] = {
                "state": "UNKNOWN",
                "confidence": round(
                    float(
                        vest_visibility[
                            "confidence"
                        ]
                    ),
                    3,
                ),
                "source": vest_visibility[
                    "source"
                ],
                "details": {
                    "reason": (
                        "Insufficient visual evidence "
                        "was available for vest assessment."
                    )
                },
                "visibility": vest_visibility,
            }

        else:

            result["safety_vest"] = {
                "state": "ABSENT",
                "confidence": 0.0,
                "source": "person_without_vest",
                "details": {
                    "reason": (
                        "Worker evidence is available, "
                        "but no sufficiently strong "
                        "safety-vest association was detected."
                    )
                },
                "visibility": vest_visibility,
            }

    # ============================================================
    # GLOVES
    # ============================================================

    gloves = components["gloves"]

    gloves_visibility = visibility_gate(
        worker,
        detections,
        "gloves",
    )

    if gloves is not None:

        score = confidence_from_association(
            gloves
        )

        if (
            gloves.get("evidence_valid", False)
            and score >= ASSOCIATION_THRESHOLD
        ):

            result["gloves"] = {
                "state": "PRESENT",
                "confidence": round(
                    float(score),
                    3,
                ),
                "source": "spatial_association",
                "details": gloves,
                "visibility": gloves_visibility,
            }

        else:

            result["gloves"] = {
                "state": "UNKNOWN",
                "confidence": round(
                    float(score),
                    3,
                ),
                "source": "weak_association",
                "details": gloves,
                "visibility": gloves_visibility,
            }

    else:

        if not gloves_visibility["visible"]:

            result["gloves"] = {
                "state": "UNKNOWN",
                "confidence": round(
                    float(
                        gloves_visibility[
                            "confidence"
                        ]
                    ),
                    3,
                ),
                "source": gloves_visibility[
                    "source"
                ],
                "details": {
                    "reason": (
                        "No sufficiently reliable "
                        "hand evidence was available."
                    )
                },
                "visibility": gloves_visibility,
            }

        else:

            result["gloves"] = {
                "state": "ABSENT",
                "confidence": 0.0,
                "source": "hands_without_gloves",
                "details": {
                    "reason": (
                        "Hand evidence is visible, "
                        "but no sufficiently strong "
                        "glove association was detected."
                    )
                },
                "visibility": gloves_visibility,
            }

    return {
        "worker_id": worker["worker_id"],
        "person": worker["person"],
        "ppe": result,
    }


def run_pipeline(
    image_path: str | Path,
    question: str | None = None,
) -> dict[str, Any]:
    """
    Run PPE-Sentinel inference.

    image
      ↓
    RT-DETR
      ↓
    worker association
      ↓
    evidence sufficiency
      ↓
    deterministic reasoning
    """

    detector = PPEDetector()

    detections = detector.predict(
        image_path
    )

    workers = associate_workers(
        detections
    )

    worker_evidence = [
        build_worker_evidence(
            worker,
            detections,
        )
        for worker in workers
    ]

    output = {
        "image": str(image_path),
        "detections": detections,
        "workers": worker_evidence,
    }

    if question is not None:

        reasoning = reason(
            question=question,
            workers=worker_evidence,
            detections=detections,
        )

        output["reasoning"] = reasoning

    return output


if __name__ == "__main__":

    validation_dir = (
        ROOT
        / "data"
        / "processed"
        / "sh17"
        / "val"
        / "images"
    )

    images = sorted(
        validation_dir.glob("*")
    )

    if not images:
        raise FileNotFoundError(
            f"No validation images found in {validation_dir}"
        )

    image_path = images[0]

    output = run_pipeline(
        image_path,
        "Is the worker wearing a helmet?",
    )

    print(
        f"Image: {image_path.name}"
    )

    print(
        f"Detections: {len(output['detections'])}"
    )

    print(
        f"Workers: {len(output['workers'])}"
    )

    print()
    print(
        "Reasoning:"
    )

    print(
        output.get("reasoning")
    )
