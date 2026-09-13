from pathlib import Path
import torch

from ultralytics import RTDETR


ROOT = Path(__file__).resolve().parents[1]

DATASET = ROOT / "configs" / "dataset.yaml"
OUTPUT_DIR = ROOT / "outputs" / "training"


def main():
    print("=" * 70)
    print("PPE-SENTINEL HIGH-RESOLUTION VRAM TEST")
    print("=" * 70)

    imgsz = 896
    batch = 2

    print(f"Image size : {imgsz}")
    print(f"Batch      : {batch}")
    print("Epochs     : 1")
    print("Fraction   : 0.01")
    print("AMP        : True")
    print("Device     : RTX 4050")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model = RTDETR("rtdetr-l.pt")

    try:
        model.train(
            data=str(DATASET),
            epochs=1,
            imgsz=imgsz,
            batch=batch,
            device=0,
            workers=2,
            amp=True,
            cache=False,
            fraction=0.01,
            deterministic=False,
            project=str(OUTPUT_DIR),
            name="resolution_test_896",
            exist_ok=True,
            pretrained=True,
            plots=False,
            verbose=False,
        )

        torch.cuda.synchronize()

        peak_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)
        peak_reserved = torch.cuda.max_memory_reserved() / (1024 ** 3)

        print()
        print("=" * 70)
        print("HIGH-RESOLUTION TEST SUCCESS")
        print("=" * 70)
        print(f"Peak allocated VRAM : {peak_allocated:.2f} GB")
        print(f"Peak reserved VRAM  : {peak_reserved:.2f} GB")

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print()
            print("=" * 70)
            print("CUDA OUT OF MEMORY")
            print("=" * 70)
            print(str(e))
        else:
            raise


if __name__ == "__main__":
    main()