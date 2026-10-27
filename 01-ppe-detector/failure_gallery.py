"""Find the test images the model gets most wrong (and most right) and draw them.

Each prediction is matched to a ground-truth box of the same class (IoU >= 0.5).
Drawn colours: green = correct, red = false positive, orange = missed object.
Also reports how misses split by object size, since small PPE (far-away helmets)
is the usual weak spot.

    python failure_gallery.py --weights weights/ppe_yolo11n.pt --data data/sh17_ppe/sh17_ppe.yaml
"""

import argparse
import csv
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

GREEN, RED, ORANGE = (60, 180, 75), (40, 40, 230), (0, 150, 255)
SIZE_BINS = (("small", 0.0, 0.01), ("medium", 0.01, 0.05), ("large", 0.05, 1.01))  # fraction of image area


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between every box in a (N,4) and b (M,4), xyxy format."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(br - tl, 0, None), axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def match(gt_boxes, gt_cls, pred_boxes, pred_cls, pred_conf, iou_thr=0.5):
    """Greedy same-class matching, highest confidence first.

    Returns (tp_pred_mask, fp_pred_mask, missed_gt_mask).
    """
    gt_boxes, pred_boxes = np.asarray(gt_boxes, float).reshape(-1, 4), np.asarray(pred_boxes, float).reshape(-1, 4)
    gt_cls, pred_cls, pred_conf = np.asarray(gt_cls), np.asarray(pred_cls), np.asarray(pred_conf)
    ious = iou_matrix(pred_boxes, gt_boxes)
    gt_used = np.zeros(len(gt_boxes), bool)
    tp = np.zeros(len(pred_boxes), bool)
    for p in np.argsort(-pred_conf):
        cand = np.where((gt_cls == pred_cls[p]) & ~gt_used & (ious[p] >= iou_thr))[0]
        if len(cand):
            g = cand[np.argmax(ious[p, cand])]
            gt_used[g] = tp[p] = True
    return tp, ~tp, ~gt_used


def load_gt(label_path: Path, w: int, h: int) -> tuple[np.ndarray, np.ndarray]:
    rows = [line.split() for line in label_path.read_text().splitlines() if line.strip()] if label_path.exists() else []
    if not rows:
        return np.zeros((0, 4)), np.zeros(0, int)
    arr = np.array(rows, float)
    cx, cy, bw, bh = arr[:, 1] * w, arr[:, 2] * h, arr[:, 3] * w, arr[:, 4] * h
    return np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1), arr[:, 0].astype(int)


def size_bin(box: np.ndarray, w: int, h: int) -> str:
    frac = (box[2] - box[0]) * (box[3] - box[1]) / (w * h)
    return next(name for name, lo, hi in SIZE_BINS if lo <= frac < hi)


def draw(img, boxes, labels, color):
    thick = max(2, round(max(img.shape[:2]) / 400))
    for (x1, y1, x2, y2), text in zip(boxes.astype(int), labels):
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)
        cv2.putText(img, text, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.22 * thick, color, max(1, thick - 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=None)
    ap.add_argument("--worst", type=int, default=12)
    ap.add_argument("--best", type=int, default=6)
    ap.add_argument("--out", type=Path, default=Path("results/failures"))
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.data).read_text())
    root, names = Path(cfg["path"]), cfg["names"]
    images = sorted((root / "images" / "test").glob("*.jpg"))
    model = YOLO(args.weights)

    records, missed_by_size, gt_by_size = [], Counter(), Counter()
    fp_by_class, fn_by_class = Counter(), Counter()
    for res in model.predict([str(p) for p in images], conf=args.conf, imgsz=args.imgsz,
                             device=args.device, stream=True, verbose=False):
        path = Path(res.path)
        h, w = res.orig_shape
        gt_boxes, gt_cls = load_gt(root / "labels" / "test" / f"{path.stem}.txt", w, h)
        pb = res.boxes
        pred_boxes, pred_cls, pred_conf = pb.xyxy.cpu().numpy(), pb.cls.cpu().numpy().astype(int), pb.conf.cpu().numpy()
        tp, fp, missed = match(gt_boxes, gt_cls, pred_boxes, pred_cls, pred_conf)

        for box, c, m in zip(gt_boxes, gt_cls, missed):
            key = (names[c], size_bin(box, w, h))
            gt_by_size[key] += 1
            missed_by_size[key] += int(m)
        fp_by_class.update(names[c] for c in pred_cls[fp])
        fn_by_class.update(names[c] for c in gt_cls[missed])
        records.append({"image": path.name, "gt": len(gt_boxes), "tp": int(tp.sum()), "fp": int(fp.sum()),
                        "fn": int(missed.sum()), "errors": int(fp.sum() + missed.sum()),
                        "_draw": (pred_boxes, pred_cls, pred_conf, tp, fp, gt_boxes, gt_cls, missed)})

    args.out.mkdir(parents=True, exist_ok=True)
    ranked = sorted(records, key=lambda r: (-r["errors"], -r["gt"]))
    best = sorted([r for r in records if r["gt"] >= 3], key=lambda r: (r["errors"], -r["gt"]))
    for tag, subset in (("worst", ranked[: args.worst]), ("best", best[: args.best])):
        for k, r in enumerate(subset, 1):
            pred_boxes, pred_cls, pred_conf, tp, fp, gt_boxes, gt_cls, missed = r["_draw"]
            img = cv2.imread(str(root / "images" / "test" / r["image"]))
            draw(img, pred_boxes[tp], [f"{names[c]} {s:.2f}" for c, s in zip(pred_cls[tp], pred_conf[tp])], GREEN)
            draw(img, pred_boxes[fp], [f"FP {names[c]} {s:.2f}" for c, s in zip(pred_cls[fp], pred_conf[fp])], RED)
            draw(img, gt_boxes[missed], [f"MISS {names[c]}" for c in gt_cls[missed]], ORANGE)
            cv2.imwrite(str(args.out / f"{tag}_{k:02d}_{Path(r['image']).stem}.jpg"), img)

    with open(args.out / "per_image_errors.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "gt", "tp", "fp", "fn", "errors"])
        writer.writeheader()
        writer.writerows({k: v for k, v in r.items() if k != "_draw"} for r in ranked)

    lines = ["| class | size | ground truth | missed | miss rate |", "|---|---|---|---|---|"]
    for cname in names.values():
        for sname, _, _ in SIZE_BINS:
            n = gt_by_size[(cname, sname)]
            if n:
                k = missed_by_size[(cname, sname)]
                lines.append(f"| {cname} | {sname} | {n} | {k} | {100 * k / n:.1f}% |")
    lines += ["", "| class | false positives | missed |", "|---|---|---|"]
    lines += [f"| {c} | {fp_by_class[c]} | {fn_by_class[c]} |" for c in names.values()]
    summary = "\n".join(lines) + "\n"
    (args.out / "error_summary.md").write_text(summary)
    print(summary)
    print(f"Gallery written to {args.out}")


if __name__ == "__main__":
    main()
