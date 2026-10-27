"""Measure how much accuracy the detector loses under plant-like camera conditions.

For every condition x severity it writes a degraded copy of the test split
(resized to the model input size first), evaluates the model on it, and reports
the drop versus the clean baseline.

    python robustness.py --weights weights/ppe_yolo11n.pt --data data/sh17_ppe/sh17_ppe.yaml
"""

import argparse
import csv
import shutil
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from ultralytics import YOLO

from degrade import CONDITIONS, SEVERITIES


def resize_max_side(img: np.ndarray, max_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    return img if s >= 1 else cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)


def build_variant(src_root: Path, dst_root: Path, fn, severity: int, imgsz: int, seed: int) -> None:
    img_dir, lbl_dir = dst_root / "images" / "test", dst_root / "labels" / "test"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    for i, src in enumerate(sorted((src_root / "images" / "test").glob("*.jpg"))):
        img = resize_max_side(cv2.imread(str(src)), imgsz)
        if fn is not None:
            img = fn(img, severity, np.random.default_rng(seed + i))
        cv2.imwrite(str(img_dir / src.name), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        shutil.copy(src_root / "labels" / "test" / f"{src.stem}.txt", lbl_dir)


def save_examples(variant_roots: dict[str, Path], out_path: Path) -> None:
    """One test image shown under every condition, for the README."""
    first = sorted((variant_roots["clean"] / "images" / "test").glob("*.jpg"))
    if not first:
        return
    name = first[min(3, len(first) - 1)].name
    keys = list(variant_roots)
    fig, axes = plt.subplots(1, len(keys), figsize=(3.2 * len(keys), 3))
    for ax, key in zip(np.atleast_1d(axes), keys):
        img = cv2.imread(str(variant_roots[key] / "images" / "test" / name))
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(key, fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_results(rows: list[dict], class_names: list[str], out_path: Path) -> None:
    labels = [r["condition"] for r in rows]
    x = np.arange(len(labels))
    series = ["all", *class_names]
    width = 0.8 / len(series)
    fig, ax = plt.subplots(figsize=(max(8, 1.3 * len(labels)), 4))
    for k, s in enumerate(series):
        vals = [r[f"mAP50_{s}"] for r in rows]
        if all(np.isnan(vals)):
            continue
        ax.bar(x + (k - (len(series) - 1) / 2) * width, vals, width, label=s)
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set_ylabel("mAP@0.5")
    ax.set_ylim(0, 1)
    ax.set_title("Detector accuracy under degraded camera conditions (test split)")
    ax.legend(ncol=len(series), fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None)
    ap.add_argument("--work", type=Path, default=Path("data/robustness"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.data).read_text())
    src_root = Path(cfg["path"])
    class_names = [cfg["names"][i] for i in sorted(cfg["names"])]
    model = YOLO(args.weights)

    variants = [("clean", None, 0)] + [(f"{c}_s{s}", fn, s) for c, fn in CONDITIONS.items() for s in SEVERITIES]
    variant_roots, rows = {}, []
    for key, fn, severity in variants:
        root = (args.work / key).resolve()
        if not (root / "images" / "test").exists():
            print(f"Building {key} ...")
            build_variant(src_root, root, fn, severity, args.imgsz, args.seed)
        variant_roots[key] = root
        data_yaml = root / "data.yaml"
        data_yaml.write_text(yaml.safe_dump({"path": str(root), "train": "images/test", "val": "images/test",
                                             "test": "images/test", "names": cfg["names"]}))
        m = model.val(data=str(data_yaml), split="test", imgsz=args.imgsz, batch=args.batch, device=args.device,
                      plots=False, verbose=False, project=str(Path("runs").resolve()), name=f"robust_{key}", exist_ok=True)
        row = {"condition": key, "mAP50_all": m.box.map50, "mAP50-95_all": m.box.map, "recall_all": m.box.mr}
        per_class = {int(c): m.box.class_result(i) for i, c in enumerate(m.box.ap_class_index)}
        for ci, name in enumerate(class_names):
            row[f"mAP50_{name}"] = per_class[ci][2] if ci in per_class else float("nan")  # class absent
        rows.append(row)
        print(f"{key:<16} mAP50={row['mAP50_all']:.3f}")

    clean = rows[0]["mAP50_all"]
    for r in rows:
        r["drop_vs_clean_%"] = 100 * (clean - r["mAP50_all"]) / clean if clean else 0.0

    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "robustness.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    header = "| condition | mAP50 (all) | drop vs clean | " + " | ".join(f"mAP50 {n}" for n in class_names) + " |"
    lines = [header, "|" + "---|" * (3 + len(class_names))]
    for r in rows:
        cells = [r["condition"], f"{r['mAP50_all']:.3f}", f"{r['drop_vs_clean_%']:.1f}%"]
        cells += ["n/a" if np.isnan(r[f"mAP50_{n}"]) else f"{r[f'mAP50_{n}']:.3f}" for n in class_names]
        lines.append("| " + " | ".join(cells) + " |")
    table = "\n".join(lines) + "\n"
    (args.out / "robustness.md").write_text(table)
    plot_results(rows, class_names, args.out / "robustness.png")
    save_examples(variant_roots, args.out / "robustness_examples.png")
    print(table)


if __name__ == "__main__":
    main()
