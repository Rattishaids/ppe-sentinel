from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import cv2

from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
)

from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from inference.detector import PPEDetector
from inference.pipeline import run_pipeline
from inference.reasoning import detect_intent


ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    ROOT
    / "models"
    / "ppe_sentinel_rtdetr_l_best.pt"
)

FRONTEND_PATH = ROOT / "frontend"


app = FastAPI(
    title="PPE-Sentinel API",
    description=(
        "Evidence-grounded industrial PPE detection "
        "and deterministic safety reasoning."
    ),
    version="1.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def validate_image_bytes(
    data: bytes,
) -> None:

    if not data:
        raise HTTPException(
            status_code=400,
            detail="Uploaded image is empty.",
        )

    image = cv2.imdecode(
        __import__("numpy").frombuffer(
            data,
            dtype=__import__("numpy").uint8,
        ),
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is not a valid image.",
        )


def save_upload(
    data: bytes,
    suffix: str,
) -> Path:

    temp = NamedTemporaryFile(
        delete=False,
        suffix=suffix,
    )

    temp.write(data)
    temp.flush()
    temp.close()

    return Path(temp.name)


def cleanup(
    path: Path,
) -> None:

    try:
        path.unlink(missing_ok=True)
    except Exception:
        pass


@app.get("/health")
def health() -> dict[str, Any]:

    return {
        "status": "ok",
        "service": "ppe-sentinel",
        "model": str(MODEL_PATH),
        "model_exists": MODEL_PATH.exists(),
    }


@app.post("/detect")
async def detect(
    image: UploadFile = File(...),
) -> dict[str, Any]:

    if not image.content_type:
        raise HTTPException(
            status_code=400,
            detail="Missing content type.",
        )

    if not image.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Only image uploads are supported.",
        )

    data = await image.read()

    validate_image_bytes(data)

    suffix = Path(
        image.filename or "image.jpg"
    ).suffix

    if not suffix:
        suffix = ".jpg"

    temp_path = save_upload(data, suffix)

    try:

        detector = PPEDetector()

        detections = detector.predict(temp_path)

        return {
            "success": True,
            "filename": image.filename,
            "model": "RT-DETR-L",
            "image_size": 640,
            "detection_count": len(detections),
            "detections": detections,
        }

    except FileNotFoundError as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=f"Detection failed: {exc}",
        )

    finally:

        cleanup(temp_path)


@app.post("/reason")
async def reason_image(
    image: UploadFile = File(...),
    question: str = Form(...),
) -> dict[str, Any]:

    question = question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    if not image.content_type:
        raise HTTPException(
            status_code=400,
            detail="Missing content type.",
        )

    if not image.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Only image uploads are supported.",
        )

    intent = detect_intent(question)

    if not intent["requires_detection"]:

        return {
            "success": True,
            "question": question,
            "intent": intent,
            "detection_used": False,
            "reasoning": {
                "status": "INSUFFICIENT_INFORMATION",
                "answer": (
                    "I cannot answer that question "
                    "using the available visual detection evidence."
                ),
                "confidence": 0.0,
            },
        }

    data = await image.read()

    validate_image_bytes(data)

    suffix = Path(
        image.filename or "image.jpg"
    ).suffix

    if not suffix:
        suffix = ".jpg"

    temp_path = save_upload(data, suffix)

    try:

        result = run_pipeline(
            temp_path,
            question=question,
        )

        return {
            "success": True,
            "question": question,
            "intent": result["reasoning"].get(
                "intent",
                intent,
            ),
            "detection_used": True,
            "detection_count": len(
                result["detections"]
            ),
            "worker_count": len(
                result["workers"]
            ),
            "detections": result["detections"],
            "workers": result["workers"],
            "reasoning": result["reasoning"],
        }

    except FileNotFoundError as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=f"Reasoning failed: {exc}",
        )

    finally:

        cleanup(temp_path)


@app.get("/")
def root():

    index_path = FRONTEND_PATH / "index.html"

    if index_path.exists():
        return FileResponse(index_path)

    return {
        "service": "PPE-Sentinel",
        "description": (
            "Evidence-Grounded Industrial Safety Intelligence"
        ),
        "docs": "/docs",
        "endpoints": [
            "/health",
            "/detect",
            "/reason",
        ],
    }


app.mount(
    "/",
    StaticFiles(
        directory=FRONTEND_PATH,
    ),
    name="frontend",
)
