from inference.association import (
    associate_workers,
    associate_worker_components,
)


def make_detection(
    detection_id,
    class_name,
    confidence,
    x1,
    y1,
    x2,
    y2,
):
    return {
        "detection_id": detection_id,
        "class_name": class_name,
        "confidence": confidence,
        "bbox": {
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
        },
    }


def test_single_worker_association():

    detections = [
        make_detection(
            0,
            "person",
            0.95,
            100,
            100,
            400,
            700,
        ),
        make_detection(
            1,
            "head",
            0.94,
            150,
            110,
            350,
            300,
        ),
        make_detection(
            2,
            "hands",
            0.90,
            120,
            400,
            220,
            520,
        ),
        make_detection(
            3,
            "helmet",
            0.91,
            170,
            120,
            330,
            270,
        ),
        make_detection(
            4,
            "safety-vest",
            0.88,
            130,
            280,
            370,
            500,
        ),
        make_detection(
            5,
            "gloves",
            0.87,
            130,
            410,
            210,
            500,
        ),
    ]

    workers = associate_workers(detections)

    assert len(workers) == 1

    worker = workers[0]

    assert worker["worker_id"] == 1
    assert worker["person"]["class_name"] == "person"

    assert worker["head"] is not None
    assert worker["hands"] is not None
    assert worker["helmet"] is not None
    assert worker["safety_vest"] is not None
    assert worker["gloves"] is not None


def test_worker_without_ppe_has_no_false_association():

    detections = [
        make_detection(
            0,
            "person",
            0.95,
            100,
            100,
            400,
            700,
        ),
    ]

    workers = associate_workers(detections)

    assert len(workers) == 1

    worker = workers[0]

    assert worker["helmet"] is None
    assert worker["safety_vest"] is None
    assert worker["gloves"] is None


def test_multiple_workers_are_kept_separate():

    detections = [
        make_detection(
            0,
            "person",
            0.95,
            50,
            100,
            300,
            700,
        ),
        make_detection(
            1,
            "person",
            0.94,
            500,
            100,
            750,
            700,
        ),
        make_detection(
            2,
            "helmet",
            0.92,
            100,
            120,
            250,
            260,
        ),
        make_detection(
            3,
            "helmet",
            0.91,
            550,
            120,
            700,
            260,
        ),
    ]

    workers = associate_workers(detections)

    assert len(workers) == 2

    assert workers[0]["helmet"] is not None
    assert workers[1]["helmet"] is not None

    assert workers[0]["helmet"]["reference_id"] == 2
    assert workers[1]["helmet"]["reference_id"] == 3


def test_worker_component_association_is_spatially_restricted():

    detections = [
        # Worker 1
        make_detection(
            0,
            "person",
            0.95,
            50,
            100,
            300,
            700,
        ),

        # Worker 2
        make_detection(
            1,
            "person",
            0.94,
            500,
            100,
            750,
            700,
        ),

        # Helmet belonging to worker 1
        make_detection(
            2,
            "helmet",
            0.92,
            100,
            120,
            250,
            260,
        ),

        # Helmet belonging to worker 2
        make_detection(
            3,
            "helmet",
            0.91,
            550,
            120,
            700,
            260,
        ),

        # Safety vest belonging to worker 1
        make_detection(
            4,
            "safety-vest",
            0.90,
            80,
            280,
            270,
            500,
        ),

        # Gloves belonging to worker 2
        make_detection(
            5,
            "gloves",
            0.89,
            550,
            400,
            650,
            520,
        ),
    ]

    workers = associate_workers(detections)

    assert len(workers) == 2

    worker_1 = associate_worker_components(
        workers[0],
        detections,
    )

    worker_2 = associate_worker_components(
        workers[1],
        detections,
    )

    # Worker 1 must receive only components
    # whose centers are inside worker 1's box.
    assert worker_1["components"]["helmet"] is not None
    assert (
        worker_1["components"]["helmet"]["reference_id"]
        == 2
    )

    assert worker_1["components"]["safety_vest"] is not None
    assert (
        worker_1["components"]["safety_vest"]["reference_id"]
        == 4
    )

    # Worker 1 must NOT receive worker 2's helmet.
    assert (
        worker_1["components"]["helmet"]["reference_id"]
        != 3
    )

    # Worker 2 must receive its own helmet.
    assert worker_2["components"]["helmet"] is not None
    assert (
        worker_2["components"]["helmet"]["reference_id"]
        == 3
    )

    # Worker 2 must receive its own gloves.
    assert worker_2["components"]["gloves"] is not None
    assert (
        worker_2["components"]["gloves"]["reference_id"]
        == 5
    )

    # Worker 2 must NOT receive worker 1's helmet.
    assert (
        worker_2["components"]["helmet"]["reference_id"]
        != 2
    )


def test_worker_component_association_returns_none_when_no_component_exists():

    detections = [
        make_detection(
            0,
            "person",
            0.95,
            100,
            100,
            400,
            700,
        ),
    ]

    workers = associate_workers(detections)

    assert len(workers) == 1

    result = associate_worker_components(
        workers[0],
        detections,
    )

    assert result["worker_id"] == 1
    assert result["person"]["detection_id"] == 0

    assert result["components"]["head"] is None
    assert result["components"]["hands"] is None
    assert result["components"]["helmet"] is None
    assert result["components"]["safety_vest"] is None
    assert result["components"]["gloves"] is None