"""Evaluate trained weights on the held-out test split and write a per-class table.

    python evaluate.py --weights weights/ppe_yolo11n.pt --data data/sh17_ppe/sh17_ppe.yaml
"""

import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def summarize(metrics, names: dict[int, str]) -> dict:
    box = metrics.box
    per_class = {}
    for i, cls in enumerate(box.ap_class_index):
        p, r, ap50, ap = box.class_result(i)
        per_class[names[int(cls)]] = {"precision": p, "recall": r, "mAP50": ap50, "mAP50-95": ap}
    return {
        "all": {"precision": box.mp, "recall": box.mr, "mAP50": box.map50, "mAP50-95": box.map},
        "per_class": per_class,
        "speed_ms_per_image": metrics.speed,
    }


def to_markdown(summary: dict) -> str:
    rows = [("all", summary["all"]), *summary["per_class"].items()]
    lines = ["| class | precision | recall | mAP50 | mAP50-95 |", "|---|---|---|---|---|"]
    for name, m in rows:
        lines.append(f"| {name} | {m['precision']:.3f} | {m['recall']:.3f} | {m['mAP50']:.3f} | {m['mAP50-95']:.3f} |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args()

    model = YOLO(args.weights)
    metrics = model.val(data=args.data, split="test", imgsz=args.imgsz, batch=args.batch,
                        device=args.device, project=str(Path("runs").resolve()), name="test_eval", exist_ok=True, plots=True)
    summary = summarize(metrics, model.names)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "test_metrics.json").write_text(json.dumps(summary, indent=2))
    table = to_markdown(summary)
    (args.out / "test_metrics.md").write_text(table)
    print(table)


if __name__ == "__main__":
    main()
