from __future__ import annotations

from typing import Any


PPE_ALIASES = {
    "helmet": {
        "helmet",
        "hardhat",
        "hard hat",
        "hard-hat",
    },
    "safety_vest": {
        "safety vest",
        "safety-vest",
        "vest",
        "reflective vest",
        "high visibility vest",
        "hi vis vest",
    },
    "gloves": {
        "glove",
        "gloves",
        "hand gloves",
        "protective gloves",
    },
}


def _normalise(text: str) -> str:
    return " ".join(text.lower().strip().split())


def detect_intent(question: str) -> dict[str, Any]:
    q = _normalise(question)

    ppe_type = None

    for canonical, aliases in PPE_ALIASES.items():
        if any(alias in q for alias in aliases):
            ppe_type = canonical
            break

    if ppe_type is not None:
        return {
            "intent": "ppe_status",
            "ppe_type": ppe_type,
            "requires_detection": True,
            "confidence": 1.0,
        }

    count_terms = (
        "how many",
        "number of",
        "count",
        "how much",
    )

    if any(term in q for term in count_terms):
        return {
            "intent": "count_objects",
            "ppe_type": None,
            "requires_detection": True,
            "confidence": 1.0,
        }

    visual_terms = (
        "what is in the image",
        "describe the image",
        "what do you see",
        "what objects",
        "what can you see",
        "summarize the image",
    )

    if any(term in q for term in visual_terms):
        return {
            "intent": "visual_summary",
            "ppe_type": None,
            "requires_detection": True,
            "confidence": 1.0,
        }

    return {
        "intent": "unsupported",
        "ppe_type": None,
        "requires_detection": False,
        "confidence": 0.0,
    }


def _get_ppe(
    worker: dict[str, Any],
    ppe_type: str,
) -> dict[str, Any] | None:
    """
    Read PPE evidence from both representations used by the project.

    Authoritative representation:
        worker["ppe"]["helmet"]

    Backward-compatible representation:
        worker["helmet"]
    """

    ppe = worker.get("ppe")

    if isinstance(ppe, dict):
        value = ppe.get(ppe_type)
        if isinstance(value, dict):
            return value

    # Backward compatibility for unit tests / legacy callers.
    value = worker.get(ppe_type)

    if isinstance(value, dict):
        return value

    return None


def _worker_state(
    worker: dict[str, Any],
    ppe_type: str,
) -> dict[str, Any]:
    evidence = _get_ppe(worker, ppe_type)

    if evidence is None:
        return {
            "worker_id": worker.get("worker_id"),
            "state": "UNKNOWN",
            "confidence": 0.0,
            "source": "missing_evidence",
            "details": {},
        }

    state = str(
        evidence.get("state", "UNKNOWN")
    ).upper()

    if state not in {"PRESENT", "ABSENT", "UNKNOWN"}:
        state = "UNKNOWN"

    return {
        "worker_id": worker.get("worker_id"),
        "state": state,
        "confidence": float(
            evidence.get("confidence", 0.0) or 0.0
        ),
        "source": evidence.get(
            "source",
            "unknown",
        ),
        "details": evidence.get(
            "details",
            {},
        ),
    }


def _summary(
    states: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "total_workers": len(states),
        "present": sum(
            item["state"] == "PRESENT"
            for item in states
        ),
        "absent": sum(
            item["state"] == "ABSENT"
            for item in states
        ),
        "unknown": sum(
            item["state"] == "UNKNOWN"
            for item in states
        ),
        "states": states,
    }


def reason_ppe_status(
    question: str,
    ppe_type: str,
    workers: list[dict[str, Any]],
) -> dict[str, Any]:

    if not workers:
        return {
            "status": "INSUFFICIENT_INFORMATION",
            "answer": (
                "I cannot determine the PPE status because "
                "no workers were detected."
            ),
            "confidence": 0.0,
            "summary": {
                "total_workers": 0,
                "present": 0,
                "absent": 0,
                "unknown": 0,
                "states": [],
            },
        }

    states = [
        _worker_state(worker, ppe_type)
        for worker in workers
    ]

    present = sum(
        item["state"] == "PRESENT"
        for item in states
    )

    absent = sum(
        item["state"] == "ABSENT"
        for item in states
    )

    unknown = sum(
        item["state"] == "UNKNOWN"
        for item in states
    )

    total = len(states)

    summary = _summary(states)

    # No usable PPE evidence exists.
    if present == 0 and absent == 0:
        return {
            "status": "INSUFFICIENT_INFORMATION",
            "answer": (
                f"I cannot reliably determine "
                f"{ppe_type.replace('_', ' ')} compliance "
                "because the relevant visual evidence "
                "is insufficient."
            ),
            "confidence": 0.0,
            "summary": summary,
        }

    # Every worker has positive evidence.
    if unknown == 0 and present == total:
        confidence = min(
            item["confidence"]
            for item in states
        )

        return {
            "status": "DETERMINED",
            "answer": (
                f"All {total} detected workers have "
                f"evidence of "
                f"{ppe_type.replace('_', ' ')}."
            ),
            "confidence": confidence,
            "summary": summary,
        }

    # Every worker has negative evidence.
    if unknown == 0 and absent == total:
        confidence = min(
            item["confidence"]
            for item in states
        )

        return {
            "status": "DETERMINED",
            "answer": (
                f"None of the {total} detected workers "
                f"have sufficient evidence of "
                f"{ppe_type.replace('_', ' ')}."
            ),
            "confidence": confidence,
            "summary": summary,
        }

    # Some workers are determined while others may be
    # unknown, or there is a PRESENT/ABSENT mixture.
    determined = [
        item["confidence"]
        for item in states
        if item["state"] != "UNKNOWN"
    ]

    confidence = (
        min(determined)
        if determined
        else 0.0
    )

    present_ids = [
        item["worker_id"]
        for item in states
        if item["state"] == "PRESENT"
    ]

    absent_ids = [
        item["worker_id"]
        for item in states
        if item["state"] == "ABSENT"
    ]

    unknown_ids = [
        item["worker_id"]
        for item in states
        if item["state"] == "UNKNOWN"
    ]

    parts = []

    if present_ids:
        parts.append(
            f"workers {', '.join(map(str, present_ids))} "
            f"show evidence of "
            f"{ppe_type.replace('_', ' ')}"
        )

    if absent_ids:
        parts.append(
            f"workers {', '.join(map(str, absent_ids))} "
            f"do not show sufficient evidence of "
            f"{ppe_type.replace('_', ' ')}"
        )

    if unknown_ids:
        parts.append(
            f"workers {', '.join(map(str, unknown_ids))} "
            "cannot be determined from the available evidence"
        )

    answer = (
        "The evidence is mixed across the detected "
        "workers: "
        + "; ".join(parts)
        + "."
    )

    return {
        "status": "PARTIAL",
        "answer": answer,
        "confidence": confidence,
        "summary": summary,
    }


def reason_count(
    question: str,
    workers: list[dict[str, Any]],
    detections: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Count workers from the authoritative worker list when
    available. Fall back to person detections for compatibility
    with direct reasoning-layer tests and callers.
    """

    if workers:
        count = len(workers)
    else:
        count = sum(
            1
            for detection in detections
            if detection.get("class_name") == "person"
            or detection.get("class_id") == 0
        )

    return {
        "status": "DETERMINED",
        "answer": f"I detected {count} worker(s).",
        "count": count,
        "confidence": 1.0 if count > 0 else 0.0,
    }


def reason(
    question: str,
    workers: list[dict[str, Any]],
    detections: list[dict[str, Any]],
) -> dict[str, Any]:

    intent = detect_intent(question)

    if not intent["requires_detection"]:
        return {
            "intent": intent,
            "reasoning": {
                "status": "INSUFFICIENT_INFORMATION",
                "answer": (
                    "I cannot answer that question using "
                    "the available visual detection evidence."
                ),
                "confidence": 0.0,
            },
        }

    if intent["intent"] == "ppe_status":
        result = reason_ppe_status(
            question=question,
            ppe_type=intent["ppe_type"],
            workers=workers,
        )

        return {
            "intent": intent,
            "reasoning": result,
        }

    if intent["intent"] == "count_objects":
        return {
            "intent": intent,
            "reasoning": reason_count(
                question=question,
                workers=workers,
                detections=detections,
            ),
        }

    if intent["intent"] == "visual_summary":
        object_counts: dict[str, int] = {}

        for detection in detections:
            name = detection.get(
                "class_name",
                "unknown",
            )
            object_counts[name] = (
                object_counts.get(name, 0) + 1
            )

        return {
            "intent": intent,
            "reasoning": {
                "status": "DETERMINED",
                "answer": (
                    f"I detected {len(detections)} objects "
                    f"across {len(workers)} worker(s)."
                ),
                "confidence": (
                    1.0 if detections else 0.0
                ),
                "object_counts": object_counts,
            },
        }

    return {
        "intent": intent,
        "reasoning": {
            "status": "INSUFFICIENT_INFORMATION",
            "answer": (
                "I cannot answer that question using "
                "the available visual detection evidence."
            ),
            "confidence": 0.0,
        },
    }


if __name__ == "__main__":
    print(
        detect_intent(
            "Is the worker wearing a helmet?"
        )
    )
