from pathlib import Path
import torch
from ultralytics import RTDETR


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "configs" / "dataset.yaml"
OUTPUT_DIR = ROOT / "outputs" / "training"


def test_batch(batch_size):
    print()
    print("=" * 70)
    print(f"TESTING BATCH SIZE: {batch_size}")
    print("=" * 70)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model = RTDETR("rtdetr-l.pt")

    try:
        model.train(
            data=str(DATASET),
            epochs=1,
            imgsz=640,
            batch=batch_size,
            device=0,
            workers=2,
            amp=True,
            cache=False,
            fraction=0.01,
            deterministic=False,
            project=str(OUTPUT_DIR),
            name=f"batch_test_{batch_size}",
            exist_ok=True,
            pretrained=True,
            plots=False,
            verbose=False,
        )

        peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)

        print(f"Batch {batch_size}: SUCCESS")
        print(f"Peak allocated GPU memory: {peak_memory:.2f} GB")

        return True

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"Batch {batch_size}: CUDA OUT OF MEMORY")
            return False

        raise


def main():
    print("=" * 70)
    print("PPE-SENTINEL GPU BATCH-SIZE TEST")
    print("=" * 70)

    for batch_size in [2]:
        success = test_batch(batch_size)

        if not success:
            break

    print()
    print("=" * 70)
    print("BATCH TEST FINISHED")
    print("=" * 70)


if __name__ == "__main__":
    main()