"""Fine-tune a pretrained YOLO11 detector on the 4-class PPE dataset.

    python train.py --data data/sh17_ppe/sh17_ppe.yaml --epochs 50
"""

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="yolo11n.pt", help="pretrained COCO checkpoint to start from")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None, help="e.g. 0, cpu, mps (default: auto)")
    ap.add_argument("--name", default="ppe_yolo11n")
    ap.add_argument("--out", type=Path, default=Path("weights/ppe_yolo11n.pt"))
    args = ap.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(Path("runs").resolve()),
        name=args.name,
        exist_ok=True,
        patience=15,
        seed=0,
        plots=True,
    )

    best = Path(model.trainer.best)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(best, args.out)
    print(f"Best weights: {best} -> copied to {args.out}")


if __name__ == "__main__":
    main()
