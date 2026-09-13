from pathlib import Path

from ultralytics import RTDETR


ROOT = Path(__file__).resolve().parents[1]

DATASET = ROOT / "configs" / "dataset.yaml"
OUTPUT_DIR = ROOT / "outputs" / "training"


def main():
    print("=" * 70)
    print("PPE-SENTINEL RT-DETR TRAINING SMOKE TEST")
    print("=" * 70)

    print(f"Dataset config: {DATASET}")
    print(f"Output directory: {OUTPUT_DIR}")

    model = RTDETR("rtdetr-l.pt")

    print()
    print("Starting controlled training test...")
    print("Model: RT-DETR-L")
    print("Image size: 640")
    print("Batch size: 1")
    print("Epochs: 1")
    print("Dataset fraction: 2%")
    print("Device: CUDA GPU 0")
    print()

    results = model.train(
        data=str(DATASET),
        epochs=1,
        imgsz=640,
        batch=1,
        device=0,
        workers=2,
        amp=True,
        cache=False,
        fraction=0.02,
        deterministic=False,
        project=str(OUTPUT_DIR),
        name="smoke_test",
        exist_ok=True,
        pretrained=True,
        verbose=True,
        plots=True,
    )

    print()
    print("=" * 70)
    print("SMOKE TEST COMPLETED")
    print("=" * 70)
    print(f"Results: {results}")


if __name__ == "__main__":
    main()