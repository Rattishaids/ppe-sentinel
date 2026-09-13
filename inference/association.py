from typing import Any


# ================================================================
# BASIC BOUNDING-BOX GEOMETRY
# ================================================================

def area(box: dict[str, float]) -> float:
    """Return bounding-box area."""
    width = max(0.0, box["x2"] - box["x1"])
    height = max(0.0, box["y2"] - box["y1"])
    return width * height


def intersection_area(
    a: dict[str, float],
    b: dict[str, float],
) -> float:
    """Return intersection area of two bounding boxes."""
    x1 = max(a["x1"], b["x1"])
    y1 = max(a["y1"], b["y1"])
    x2 = min(a["x2"], b["x2"])
    y2 = min(a["y2"], b["y2"])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    return (x2 - x1) * (y2 - y1)


def iou(
    a: dict[str, float],
    b: dict[str, float],
) -> float:
    """Calculate intersection-over-union."""
    intersection = intersection_area(a, b)

    if intersection <= 0:
        return 0.0

    union = area(a) + area(b) - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def containment_ratio(
    inner: dict[str, float],
    outer: dict[str, float],
) -> float:
    """Return fraction of inner box contained by outer box."""
    inner_area = area(inner)

    if inner_area <= 0:
        return 0.0

    return intersection_area(inner, outer) / inner_area


def center(
    box: dict[str, float],
) -> tuple[float, float]:
    """Return bounding-box center."""
    return (
        (box["x1"] + box["x2"]) / 2.0,
        (box["y1"] + box["y2"]) / 2.0,
    )


def center_inside(
    inner: dict[str, float],
    outer: dict[str, float],
) -> bool:
    """Return True when inner center lies inside outer."""
    cx, cy = center(inner)

    return (
        outer["x1"] <= cx <= outer["x2"]
        and outer["y1"] <= cy <= outer["y2"]
    )


def relative_position(
    subject: dict[str, float],
    reference: dict[str, float],
) -> dict[str, float]:
    """Calculate normalized position of subject relative to reference."""
    sx, sy = center(subject)
    rx, ry = center(reference)

    rw = max(
        1.0,
        reference["x2"] - reference["x1"],
    )

    rh = max(
        1.0,
        reference["y2"] - reference["y1"],
    )

    return {
        "dx": (sx - rx) / rw,
        "dy": (sy - ry) / rh,
    }


# ================================================================
# GEOMETRIC ASSOCIATION
# ================================================================

def association_score(
    subject: dict[str, Any],
    reference: dict[str, Any],
) -> dict[str, Any]:
    """
    Calculate geometric relationship.

    Direction:

        subject -> reference
    """

    overlap = iou(
        subject["bbox"],
        reference["bbox"],
    )

    containment = containment_ratio(
        subject["bbox"],
        reference["bbox"],
    )

    inside = center_inside(
        subject["bbox"],
        reference["bbox"],
    )

    position = relative_position(
        subject["bbox"],
        reference["bbox"],
    )

    return {
        "iou": overlap,
        "containment": containment,
        "center_inside": inside,
        "relative_position": position,
    }


# ================================================================
# ASSOCIATION QUALITY
# ================================================================

def association_quality(
    subject: dict[str, Any],
    reference: dict[str, Any],
) -> dict[str, Any]:
    """
    Calculate association strength.

    Detector confidence: 35%

    Spatial evidence: 65%

    Spatial evidence:
        containment: 45%
        IoU:         30%
        center:      25%

    This is an evidence-strength score, not a calibrated
    probability.
    """

    geometry = association_score(
        subject,
        reference,
    )

    detector_confidence = (
        float(subject["confidence"])
        + float(reference["confidence"])
    ) / 2.0

    spatial_score = (
        0.45 * float(geometry["containment"])
        + 0.30 * float(geometry["iou"])
        + 0.25 * float(geometry["center_inside"])
    )

    quality_score = (
        0.35 * detector_confidence
        + 0.65 * spatial_score
    )

    return {
        "quality_score": min(
            1.0,
            max(0.0, quality_score),
        ),
        "detector_confidence": detector_confidence,
        "spatial_score": spatial_score,
        **geometry,
    }


# ================================================================
# GENERIC CANDIDATE SEARCH
# ================================================================

def find_candidates(
    detections: list[dict[str, Any]],
    subject_class: str,
    reference_class: str,
) -> list[dict[str, Any]]:
    """Return every subject-reference candidate."""

    subjects = [
        detection
        for detection in detections
        if detection["class_name"] == subject_class
    ]

    references = [
        detection
        for detection in detections
        if detection["class_name"] == reference_class
    ]

    candidates = []

    for subject in subjects:

        for reference in references:

            quality = association_quality(
                subject,
                reference,
            )

            candidates.append(
                {
                    "subject": subject,
                    "reference": reference,

                    "subject_id": subject["detection_id"],
                    "subject_class": subject["class_name"],
                    "subject_confidence": float(
                        subject["confidence"]
                    ),

                    "reference_id": reference["detection_id"],
                    "reference_class": reference["class_name"],
                    "reference_confidence": float(
                        reference["confidence"]
                    ),

                    **quality,
                }
            )

    return candidates


# ================================================================
# BEST REFERENCE FOR SUBJECT
# ================================================================

def best_reference_for_subject(
    subject: dict[str, Any],
    references: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """
    Select strongest reference for a subject.

    Example compatibility relation:

        person -> helmet

    The returned object preserves the original subject detection.
    """

    if not references:
        return None

    best = None

    for reference in references:

        quality = association_quality(
            subject,
            reference,
        )

        candidate = {
            # Preserve original subject.
            **subject,

            # Association metadata.
            "subject_id": subject["detection_id"],
            "subject_class": subject["class_name"],
            "subject_confidence": float(
                subject["confidence"]
            ),

            "reference_id": reference["detection_id"],
            "reference_class": reference["class_name"],
            "reference_confidence": float(
                reference["confidence"]
            ),

            **quality,
        }

        if (
            best is None
            or candidate["quality_score"]
            > best["quality_score"]
        ):
            best = candidate

    return best


# ================================================================
# BEST SUBJECT FOR REFERENCE
# ================================================================

def best_subject_for_reference(
    subjects: list[dict[str, Any]],
    reference: dict[str, Any],
) -> dict[str, Any] | None:
    """
    Select strongest subject for a reference.

    Used for authoritative anatomical relations:

        head   -> person
        hands  -> person
        helmet -> head
        gloves -> hands
        vest   -> person
    """

    if not subjects:
        return None

    best = None

    for subject in subjects:

        geometry = association_score(
            subject,
            reference,
        )

        if not geometry["center_inside"]:
            continue

        quality = association_quality(
            subject,
            reference,
        )

        candidate = {
            # Preserve complete original detection.
            **subject,

            "subject_id": subject["detection_id"],
            "subject_class": subject["class_name"],
            "subject_confidence": float(
                subject["confidence"]
            ),

            "reference_id": reference["detection_id"],
            "reference_class": reference["class_name"],
            "reference_confidence": float(
                reference["confidence"]
            ),

            **quality,
        }

        if (
            best is None
            or candidate["quality_score"]
            > best["quality_score"]
        ):
            best = candidate

    return best


# ================================================================
# WORKER ASSOCIATION
# ================================================================

def associate_workers(
    detections: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Build worker-level spatial associations.

    Compatibility view:
        head  -> person
        hands -> person
        person -> helmet
        person -> safety-vest
        person -> gloves

    The PPE fields are spatial candidates only.
    Authoritative evidence is produced by
    associate_worker_components().
    """

    persons = [
        d for d in detections
        if d["class_name"] == "person"
    ]

    heads = [
        d for d in detections
        if d["class_name"] == "head"
    ]

    hands = [
        d for d in detections
        if d["class_name"] == "hands"
    ]

    helmets = [
        d for d in detections
        if d["class_name"] == "helmet"
    ]

    vests = [
        d for d in detections
        if d["class_name"] == "safety-vest"
    ]

    gloves = [
        d for d in detections
        if d["class_name"] == "gloves"
    ]

    workers = []

    for worker_index, person in enumerate(persons, start=1):

        # ========================================================
        # HEAD -> PERSON
        # ========================================================

        head_candidates = [
            head
            for head in heads
            if center_inside(
                head["bbox"],
                person["bbox"],
            )
        ]

        head = best_subject_for_reference(
            head_candidates,
            person,
        )

        # ========================================================
        # HANDS -> PERSON
        # ========================================================

        hand_candidates = [
            hand
            for hand in hands
            if center_inside(
                hand["bbox"],
                person["bbox"],
            )
        ]

        hand = best_subject_for_reference(
            hand_candidates,
            person,
        )

        # ========================================================
        # PERSON -> HELMET
        #
        # Compatibility representation.
        #
        # IMPORTANT:
        # reference_id = HELMET detection ID.
        # ========================================================

        helmet_candidates = [
            helmet
            for helmet in helmets
            if center_inside(
                helmet["bbox"],
                person["bbox"],
            )
        ]

        helmet = None

        if helmet_candidates:

            best = max(
                helmet_candidates,
                key=lambda d: association_quality(
                    person,
                    d,
                )["quality_score"],
            )

            quality = association_quality(
                person,
                best,
            )

            helmet = {
                **best,

                "subject_id": person[
                    "detection_id"
                ],
                "subject_class": person[
                    "class_name"
                ],
                "subject_confidence": float(
                    person["confidence"]
                ),

                "reference_id": best[
                    "detection_id"
                ],
                "reference_class": best[
                    "class_name"
                ],
                "reference_confidence": float(
                    best["confidence"]
                ),

                **quality,
            }

        # ========================================================
        # PERSON -> SAFETY VEST
        # ========================================================

        vest_candidates = [
            vest
            for vest in vests
            if center_inside(
                vest["bbox"],
                person["bbox"],
            )
        ]

        safety_vest = None

        if vest_candidates:

            best = max(
                vest_candidates,
                key=lambda d: association_quality(
                    person,
                    d,
                )["quality_score"],
            )

            quality = association_quality(
                person,
                best,
            )

            safety_vest = {
                **best,

                "subject_id": person[
                    "detection_id"
                ],
                "subject_class": person[
                    "class_name"
                ],
                "subject_confidence": float(
                    person["confidence"]
                ),

                "reference_id": best[
                    "detection_id"
                ],
                "reference_class": best[
                    "class_name"
                ],
                "reference_confidence": float(
                    best["confidence"]
                ),

                **quality,
            }

        # ========================================================
        # PERSON -> GLOVES
        # ========================================================

        glove_candidates = [
            glove
            for glove in gloves
            if center_inside(
                glove["bbox"],
                person["bbox"],
            )
        ]

        glove = None

        if glove_candidates:

            best = max(
                glove_candidates,
                key=lambda d: association_quality(
                    person,
                    d,
                )["quality_score"],
            )

            quality = association_quality(
                person,
                best,
            )

            glove = {
                **best,

                "subject_id": person[
                    "detection_id"
                ],
                "subject_class": person[
                    "class_name"
                ],
                "subject_confidence": float(
                    person["confidence"]
                ),

                "reference_id": best[
                    "detection_id"
                ],
                "reference_class": best[
                    "class_name"
                ],
                "reference_confidence": float(
                    best["confidence"]
                ),

                **quality,
            }

        workers.append(
            {
                "worker_id": worker_index,
                "person": person,
                "head": head,
                "hands": hand,
                "helmet": helmet,
                "safety_vest": safety_vest,
                "gloves": glove,
            }
        )

    return workers


def associate_worker_components(
    worker: dict[str, Any],
    detections: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Build authoritative worker-specific PPE associations.

    Preferred evidence graph:

        helmet      -> head  -> person
        gloves      -> hands -> person
        safety-vest -> person

    If head/hands evidence is unavailable, a PPE-to-person
    spatial fallback is retained for compatibility and diagnostics,
    but evidence_valid is False.

    Therefore fallback candidates can never be interpreted as
    definitive PRESENT evidence by the downstream evidence layer.
    """

    person = worker["person"]
    head = worker.get("head")
    hands = worker.get("hands")

    helmets = [
        d for d in detections
        if d["class_name"] == "helmet"
    ]

    vests = [
        d for d in detections
        if d["class_name"] == "safety-vest"
    ]

    gloves = [
        d for d in detections
        if d["class_name"] == "gloves"
    ]

    # ============================================================
    # HELMET
    # ============================================================

    helmet = None

    if head is not None:

        helmet_candidates = [
            detection
            for detection in helmets
            if center_inside(
                detection["bbox"],
                head["bbox"],
            )
            and center_inside(
                detection["bbox"],
                person["bbox"],
            )
        ]

        helmet = best_subject_for_reference(
            helmet_candidates,
            head,
        )

        if helmet is not None:
            helmet["association_mode"] = (
                "helmet_to_head"
            )
            helmet["evidence_valid"] = True

    else:

        # --------------------------------------------------------
        # Fallback: helmet spatially associated with person.
        #
        # This is retained for compatibility and diagnostics.
        # It is NOT valid anatomical evidence.
        # --------------------------------------------------------

        fallback_candidates = [
            detection
            for detection in helmets
            if center_inside(
                detection["bbox"],
                person["bbox"],
            )
        ]

        if fallback_candidates:

            best = max(
                fallback_candidates,
                key=lambda d: association_quality(
                    person,
                    d,
                )["quality_score"],
            )

            quality = association_quality(
                person,
                best,
            )

            helmet = {
                **best,

                "subject_id": person[
                    "detection_id"
                ],
                "subject_class": person[
                    "class_name"
                ],
                "subject_confidence": float(
                    person["confidence"]
                ),

                # IMPORTANT:
                # reference_id refers to the PPE detection
                # in the compatibility representation.
                "reference_id": best[
                    "detection_id"
                ],
                "reference_class": best[
                    "class_name"
                ],
                "reference_confidence": float(
                    best["confidence"]
                ),

                **quality,

                "association_mode": (
                    "fallback_helmet_to_person"
                ),

                # Head evidence is unavailable, so this
                # candidate cannot prove PRESENT.
                "evidence_valid": False,
            }

    # ============================================================
    # GLOVES
    # ============================================================

    glove = None

    if hands is not None:

        glove_candidates = [
            detection
            for detection in gloves
            if center_inside(
                detection["bbox"],
                hands["bbox"],
            )
            and center_inside(
                detection["bbox"],
                person["bbox"],
            )
        ]

        glove = best_subject_for_reference(
            glove_candidates,
            hands,
        )

        if glove is not None:
            glove["association_mode"] = (
                "gloves_to_hands"
            )
            glove["evidence_valid"] = True

    else:

        # --------------------------------------------------------
        # Fallback: gloves spatially associated with person.
        #
        # Retained for diagnostics, but not valid evidence.
        # --------------------------------------------------------

        fallback_candidates = [
            detection
            for detection in gloves
            if center_inside(
                detection["bbox"],
                person["bbox"],
            )
        ]

        if fallback_candidates:

            best = max(
                fallback_candidates,
                key=lambda d: association_quality(
                    person,
                    d,
                )["quality_score"],
            )

            quality = association_quality(
                person,
                best,
            )

            glove = {
                **best,

                "subject_id": person[
                    "detection_id"
                ],
                "subject_class": person[
                    "class_name"
                ],
                "subject_confidence": float(
                    person["confidence"]
                ),

                # Compatibility representation:
                # reference_id = glove detection ID.
                "reference_id": best[
                    "detection_id"
                ],
                "reference_class": best[
                    "class_name"
                ],
                "reference_confidence": float(
                    best["confidence"]
                ),

                **quality,

                "association_mode": (
                    "fallback_gloves_to_person"
                ),

                "evidence_valid": False,
            }

    # ============================================================
    # SAFETY VEST
    # ============================================================

    vest_candidates = [
        detection
        for detection in vests
        if center_inside(
            detection["bbox"],
            person["bbox"],
        )
    ]

    safety_vest = None

    if vest_candidates:

        best = max(
            vest_candidates,
            key=lambda d: association_quality(
                person,
                d,
            )["quality_score"],
        )

        quality = association_quality(
            person,
            best,
        )

        safety_vest = {
            **best,

            "subject_id": person[
                "detection_id"
            ],
            "subject_class": person[
                "class_name"
            ],
            "subject_confidence": float(
                person["confidence"]
            ),

            # Compatibility representation:
            # reference_id = safety-vest detection ID.
            "reference_id": best[
                "detection_id"
            ],
            "reference_class": best[
                "class_name"
            ],
            "reference_confidence": float(
                best["confidence"]
            ),

            **quality,

            "association_mode": (
                "vest_to_person"
            ),

            "evidence_valid": True,
        }

    # ============================================================
    # RETURN
    # ============================================================

    return {
        "worker_id": worker["worker_id"],
        "person": person,

        "components": {
            "head": head,
            "hands": hands,
            "helmet": helmet,
            "safety_vest": safety_vest,
            "gloves": glove,
        },
    }


