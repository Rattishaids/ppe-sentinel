from inference.evidence import (
    ASSOCIATION_THRESHOLD,
    confidence_from_association,
    visibility_gate,
)


def make_detection(
    class_name,
    confidence,
    x1,
    y1,
    x2,
    y2,
):
    return {
        "class_name": class_name,
        "confidence": confidence,
        "bbox": {
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
        },
    }


def make_worker(
    confidence=0.95,
    x1=0,
    y1=0,
    x2=100,
    y2=200,
):
    person = make_detection(
        "person",
        confidence,
        x1,
        y1,
        x2,
        y2,
    )

    return {
        "worker_id": 1,
        "person": person,
    }


def test_association_threshold_is_conservative():
    assert ASSOCIATION_THRESHOLD == 0.60


def test_confidence_from_quality_score():
    candidate = {
        "quality_score": 0.87,
        "containment": 0.95,
        "iou": 0.80,
        "center_inside": True,
    }

    result = confidence_from_association(candidate)

    assert result == 0.87


def test_confidence_fallback_uses_geometry_and_detector_confidence():
    candidate = {
        "subject_confidence": 0.90,
        "reference_confidence": 0.80,
        "containment": 1.0,
        "iou": 1.0,
        "center_inside": True,
    }

    result = confidence_from_association(candidate)

    assert 0.0 <= result <= 1.0
    assert result > 0.80


def test_visibility_gate_unknown_when_head_missing():
    worker = make_worker()

    detections = [
        worker["person"],
    ]

    result = visibility_gate(
        worker,
        detections,
        "helmet",
    )

    assert result["visible"] is False
    assert result["confidence"] == 0.0
    assert result["source"] == "region_not_detected"


def test_visibility_gate_detects_visible_head():
    worker = make_worker()

    head = make_detection(
        "head",
        0.90,
        20,
        10,
        80,
        60,
    )

    detections = [
        worker["person"],
        head,
    ]

    result = visibility_gate(
        worker,
        detections,
        "helmet",
    )

    assert result["visible"] is True
    assert result["confidence"] > 0.0
    assert result["source"] == "head_visible"


def test_visibility_gate_detects_visible_hands():
    worker = make_worker()

    hands = make_detection(
        "hands",
        0.90,
        20,
        70,
        80,
        130,
    )

    detections = [
        worker["person"],
        hands,
    ]

    result = visibility_gate(
        worker,
        detections,
        "gloves",
    )

    assert result["visible"] is True
    assert result["confidence"] > 0.0
    assert result["source"] == "hands_visible"


def test_visibility_gate_ignores_region_outside_worker():
    worker = make_worker()

    head = make_detection(
        "head",
        0.95,
        200,
        200,
        250,
        250,
    )

    detections = [
        worker["person"],
        head,
    ]

    result = visibility_gate(
        worker,
        detections,
        "helmet",
    )

    assert result["visible"] is False
    assert result["source"] == "region_not_associated"


def test_visibility_gate_person_region_for_vest():
    worker = make_worker()

    detections = [
        worker["person"],
    ]

    result = visibility_gate(
        worker,
        detections,
        "safety_vest",
    )

    assert result["visible"] is True
    assert result["confidence"] > 0.0
    assert result["source"] == "person_visible"