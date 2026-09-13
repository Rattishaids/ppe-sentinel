from pathlib import Path

from ultralytics import RTDETR


ROOT = Path(__file__).resolve().parents[1]

DATASET = ROOT / "configs" / "dataset.yaml"
OUTPUT_DIR = ROOT / "outputs" / "training"


def main():

    print("=" * 70)
    print("PPE-SENTINEL — 896px HIGH-RESOLUTION PROBE")
    print("=" * 70)

    print(f"Dataset : {DATASET}")
    print(f"Output  : {OUTPUT_DIR}")

    model = RTDETR("rtdetr-l.pt")

    model.train(
        data=str(DATASET),

        # ----------------------------------------------------------
        # Diagnostic experiment
        # ----------------------------------------------------------
        epochs=10,
        imgsz=896,
        batch=2,
        device=0,

        # ----------------------------------------------------------
        # Reproducibility
        # ----------------------------------------------------------
        seed=0,
        workers=4,
        cache="disk",
        deterministic=False,

        # ----------------------------------------------------------
        # Mixed precision
        # ----------------------------------------------------------
        amp=True,

        # ----------------------------------------------------------
        # Optimization
        # ----------------------------------------------------------
        optimizer="auto",

        # ----------------------------------------------------------
        # Augmentation
        # ----------------------------------------------------------
        augment=True,

        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,

        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,

        fliplr=0.5,
        flipud=0.0,

        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.0,

        # ----------------------------------------------------------
        # Validation
        # ----------------------------------------------------------
        val=True,
        save=True,
        save_period=5,
        plots=True,

        # ----------------------------------------------------------
        # Separate experiment output
        # ----------------------------------------------------------
        project=str(OUTPUT_DIR),
        name="rtdetr_l_896_probe_10e",
        exist_ok=True,

        pretrained=True,
        verbose=True,
    )


if __name__ == "__main__":
    main()
