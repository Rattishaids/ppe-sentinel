from pathlib import Path

from ultralytics import RTDETR


ROOT = Path(__file__).resolve().parents[1]

DATASET = ROOT / "configs" / "dataset.yaml"
OUTPUT_DIR = ROOT / "outputs" / "training"


def main():
    print("=" * 70)
    print("PPE-SENTINEL RT-DETR-L TRAINING")
    print("=" * 70)

    print(f"Dataset : {DATASET}")
    print(f"Output  : {OUTPUT_DIR}")

    # Load pretrained RT-DETR-L
    model = RTDETR("rtdetr-l.pt")

    model.train(
        data=str(DATASET),

        # ==============================================================
        # CORE TRAINING
        # ==============================================================
        epochs=50,
        imgsz=640,
        batch=4,
        device=0,

        # ==============================================================
        # REPRODUCIBILITY / DATA LOADING
        # ==============================================================
        seed=0,
        workers=4,
        cache="disk",
        deterministic=False,

        # ==============================================================
        # MIXED PRECISION
        # ==============================================================
        amp=True,

        # ==============================================================
        # OPTIMIZATION
        # ==============================================================
        optimizer="auto",

        # ==============================================================
        # AUGMENTATION
        # ==============================================================
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

        # ==============================================================
        # VALIDATION / CHECKPOINTING
        # ==============================================================
        val=True,
        save=True,

        # Save a checkpoint every 5 epochs
        save_period=5,

        # Generate training/validation plots
        plots=True,

        # ==============================================================
        # OUTPUT
        # ==============================================================
        project=str(OUTPUT_DIR),
        name="rtdetr_l_batch4_50e",
        exist_ok=True,

        # ==============================================================
        # PRETRAINED INITIALIZATION
        # ==============================================================
        pretrained=True,

        verbose=True,
    )


if __name__ == "__main__":
    main()