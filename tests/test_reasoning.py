from inference.reasoning import (
    detect_intent,
    reason,
)


def make_evidence(
    state,
    confidence=0.8,
):
    return {
        "state": state,
        "confidence": confidence,
        "source": "test",
        "details": {},
    }


def test_helmet_question_routes_to_ppe():

    result = detect_intent(
        "Is the worker wearing a helmet?"
    )

    assert result["intent"] == "ppe_status"
    assert result["ppe_type"] == "helmet"
    assert result["requires_detection"] is True


def test_worker_count_routes_correctly():

    result = detect_intent(
        "How many workers are there?"
    )

    assert result["intent"] == "count_objects"
    assert result["requires_detection"] is True


def test_unsupported_question_does_not_require_detection():

    result = detect_intent(
        "What is the weather tomorrow?"
    )

    assert result["intent"] == "unsupported"
    assert result["requires_detection"] is False


def test_present_helmet():

    workers = [
        {
            "worker_id": 1,
            "helmet": make_evidence(
                "PRESENT",
                0.9,
            ),
        }
    ]

    result = reason(
        "Is the worker wearing a helmet?",
        workers,
        [],
    )

    assert result["reasoning"]["status"] == "DETERMINED"
    assert result["reasoning"]["summary"]["present"] == 1


def test_absent_helmet():

    workers = [
        {
            "worker_id": 1,
            "helmet": make_evidence(
                "ABSENT",
                0.0,
            ),
        }
    ]

    result = reason(
        "Is the worker wearing a helmet?",
        workers,
        [],
    )

    assert result["reasoning"]["status"] == "DETERMINED"
    assert result["reasoning"]["summary"]["absent"] == 1


def test_unknown_helmet_returns_insufficient_information():

    workers = [
        {
            "worker_id": 1,
            "helmet": make_evidence(
                "UNKNOWN",
                0.0,
            ),
        }
    ]

    result = reason(
        "Is the worker wearing a helmet?",
        workers,
        [],
    )

    assert (
        result["reasoning"]["status"]
        == "INSUFFICIENT_INFORMATION"
    )


def test_count_workers():

    detections = [
        {
            "detection_id": 1,
            "class_name": "person",
            "confidence": 0.9,
            "bbox": {},
        },
        {
            "detection_id": 2,
            "class_name": "person",
            "confidence": 0.8,
            "bbox": {},
        },
    ]

    result = reason(
        "How many workers are there?",
        [],
        detections,
    )

    assert result["reasoning"]["count"] == 2