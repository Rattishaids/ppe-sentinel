from dataclasses import dataclass
from typing import Any

from inference.association import (
    association_score,
    best_reference_for_subject,
)


ASSOCIATION_THRESHOLD = 0.60


@dataclass
class Evidence:
    state: str
    confidence: float
    source: str


@dataclass
class WorkerEvidence:
    worker_id: int
    helmet: Evidence
    safety_vest: Evidence
    gloves: Evidence


PPE_CONFIG = {
    "helmet": {
        "region": "head",
        "component": "helmet",
    },
    "safety_vest": {
        "region": "person",
        "component": "safety-vest",
    },
    "gloves": {
        "region": "hands",
        "component": "gloves",
    },
}


def confidence_from_association(
    candidate: dict,
    object_confidence: float | None = None,
) -> float:
    """
    Convert detector confidence and spatial compatibility into
    an evidence-strength score.

    This is an evidence-strength measure, NOT a calibrated
    probability of PPE compliance.
    """

    if "quality_score" in candidate:
        return float(candidate["quality_score"])

    if object_confidence is None:
        object_confidence = float(
            candidate.get(
                "subject_confidence",
                candidate.get("confidence", 0.0),
            )
        )

    reference_confidence = float(
        candidate.get(
            "reference_confidence",
            0.0,
        )
    )

    detector_confidence = (
        object_confidence + reference_confidence
    ) / 2.0

    spatial_score = (
        0.45 * float(
            candidate.get("containment", 0.0)
        )
        + 0.30 * float(
            candidate.get("iou", 0.0)
        )
        + 0.25 * float(
            candidate.get("center_inside", False)
        )
    )

    quality_score = (
        0.35 * detector_confidence
        + 0.65 * spatial_score
    )

    return min(
        1.0,
        max(0.0, quality_score),
    )


def region_visible(
    worker: dict,
    detections: list[dict],
    region_class: str,
) -> tuple[bool, float, str]:
    """
    Determine whether a PPE-relevant body region is sufficiently
    observable for this worker.

    Important:
    association_score() is evaluated as:

        region -> worker

    because we want to determine whether the region lies within
    the worker's spatial extent.

    Missing PPE is NOT treated as absence unless the relevant
    body region is sufficiently observable.
    """

    worker_person = worker["person"]

    region_detections = [
        detection
        for detection in detections
        if detection["class_name"] == region_class
    ]

    if not region_detections:
        return (
            False,
            0.0,
            "region_not_detected",
        )

    candidates = []

    for region in region_detections:

        # Correct direction:
        #
        #     region -> worker
        #
        # Example:
        #     head -> person
        #     hands -> person
        #
        geometry = association_score(
            region,
            worker_person,
        )

        if not geometry["center_inside"]:
            continue

        visibility_score = (
            0.50 * float(region["confidence"])
            + 0.30 * float(
                geometry["containment"]
            )
            + 0.20 * float(
                geometry["center_inside"]
            )
        )

        candidates.append(
            (
                min(1.0, visibility_score),
                region,
            )
        )

    if not candidates:
        return (
            False,
            0.0,
            "region_not_associated",
        )

    best_score, _ = max(
        candidates,
        key=lambda item: item[0],
    )

    if best_score >= 0.45:
        return (
            True,
            best_score,
            f"{region_class}_visible",
        )

    return (
        False,
        best_score,
        f"{region_class}_insufficient",
    )


def visibility_gate(
    worker: dict,
    detections: list[dict],
    ppe_type: str,
) -> dict:
    """
    Establish whether sufficient visual evidence exists for
    reasoning about a particular PPE item.
    """

    if ppe_type not in PPE_CONFIG:
        raise ValueError(
            f"Unsupported PPE type: {ppe_type}"
        )

    region_class = PPE_CONFIG[ppe_type]["region"]

    visible, confidence, source = region_visible(
        worker,
        detections,
        region_class,
    )

    return {
        "visible": visible,
        "confidence": float(confidence),
        "source": source,
    }


def infer_ppe_state(
    worker: dict,
    detections: list[dict],
    ppe_type: str,
) -> Evidence:
    """
    Infer PRESENT / ABSENT / UNKNOWN for one PPE type.

    Decision policy:

        body region unavailable
            -> UNKNOWN

        body region visible + strong PPE association
            -> PRESENT

        body region visible + no PPE association
            -> ABSENT

        weak PPE association
            -> UNKNOWN
    """

    if ppe_type not in PPE_CONFIG:
        raise ValueError(
            f"Unsupported PPE type: {ppe_type}"
        )

    component_class = PPE_CONFIG[ppe_type]["component"]

    visibility = visibility_gate(
        worker,
        detections,
        ppe_type,
    )

    component_detections = [
        detection
        for detection in detections
        if detection["class_name"] == component_class
    ]

    best_component = best_reference_for_subject(
        worker["person"],
        component_detections,
    )

    if best_component is not None:
        evidence_confidence = confidence_from_association(
            best_component
        )

        if evidence_confidence >= ASSOCIATION_THRESHOLD:
            return Evidence(
                state="PRESENT",
                confidence=evidence_confidence,
                source="spatial_association",
            )

        return Evidence(
            state="UNKNOWN",
            confidence=evidence_confidence,
            source="weak_association",
        )

    if not visibility["visible"]:
        return Evidence(
            state="UNKNOWN",
            confidence=visibility["confidence"],
            source=visibility["source"],
        )

    return Evidence(
        state="ABSENT",
        confidence=0.0,
        source=f"{visibility['source']}_without_{ppe_type}",
    )


def analyze_worker(
    worker: dict,
    detections: list[dict],
) -> WorkerEvidence:
    """
    Generate evidence for all supported PPE categories for
    one worker.
    """

    helmet = infer_ppe_state(
        worker,
        detections,
        "helmet",
    )

    safety_vest = infer_ppe_state(
        worker,
        detections,
        "safety_vest",
    )

    gloves = infer_ppe_state(
        worker,
        detections,
        "gloves",
    )

    return WorkerEvidence(
        worker_id=worker["worker_id"],
        helmet=helmet,
        safety_vest=safety_vest,
        gloves=gloves,
    )


def worker_evidence_to_dict(
    evidence: WorkerEvidence,
) -> dict[str, Any]:
    """
    Convert WorkerEvidence into JSON-compatible data.
    """

    return {
        "worker_id": evidence.worker_id,
        "helmet": {
            "state": evidence.helmet.state,
            "confidence": round(
                evidence.helmet.confidence,
                3,
            ),
            "source": evidence.helmet.source,
        },
        "safety_vest": {
            "state": evidence.safety_vest.state,
            "confidence": round(
                evidence.safety_vest.confidence,
                3,
            ),
            "source": evidence.safety_vest.source,
        },
        "gloves": {
            "state": evidence.gloves.state,
            "confidence": round(
                evidence.gloves.confidence,
                3,
            ),
            "source": evidence.gloves.source,
        },
    }


if __name__ == "__main__":

    person = {
        "class_name": "person",
        "confidence": 0.95,
        "bbox": {
            "x1": 0,
            "y1": 0,
            "x2": 100,
            "y2": 200,
        },
    }

    head = {
        "class_name": "head",
        "confidence": 0.90,
        "bbox": {
            "x1": 20,
            "y1": 10,
            "x2": 80,
            "y2": 60,
        },
    }

    helmet = {
        "class_name": "helmet",
        "confidence": 0.92,
        "bbox": {
            "x1": 20,
            "y1": 5,
            "x2": 80,
            "y2": 55,
        },
    }

    worker = {
        "worker_id": 1,
        "person": person,
    }

    detections = [
        person,
        head,
        helmet,
    ]

    evidence = analyze_worker(
        worker,
        detections,
    )

    print(
        worker_evidence_to_dict(evidence)
    )