"""Turn raw SH17 into a 4-class YOLO dataset: person, head, helmet, safety-vest.

- Keeps only the 4 classes ("head" covers every head, helmeted or not).
- Splits the official train list 90/10 into train/val; the official val list
  becomes a held-out TEST set that is never used for training or model selection.
- Downscales images (SH17 originals are up to 8192 px) to save disk and I/O.

    python prepare_data.py --raw data/sh17_raw --out data/sh17_ppe
"""

import argparse
import os
import random
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image, ImageOps

# SH17 class id -> our class id (see sh17.yaml in the SH17 repo)
SH17_TO_PPE = {0: 0, 12: 1, 10: 2, 16: 3}
CLASS_NAMES = ["person", "head", "helmet", "safety-vest"]


def remap_label_lines(lines: list[str]) -> list[str]:
    """Keep boxes of the 4 target classes and renumber them. Drops malformed lines."""
    out = []
    for line in lines:
        parts = line.split()
        if len(parts) != 5:
            continue
        cls = int(float(parts[0]))
        if cls in SH17_TO_PPE:
            out.append(" ".join([str(SH17_TO_PPE[cls]), *parts[1:]]))
    return out


def split_names(train_names: list[str], val_frac: float, seed: int) -> tuple[list[str], list[str]]:
    names = sorted(train_names)
    random.Random(seed).shuffle(names)
    n_val = max(1, round(len(names) * val_frac))
    return names[n_val:], names[:n_val]


def resize_image(src: Path, dst: Path, max_side: int) -> None:
    with Image.open(src) as im:
        im.draft("RGB", (max_side, max_side))  # fast reduced-size JPEG decode
        im = ImageOps.exif_transpose(im).convert("RGB")
        im.thumbnail((max_side, max_side))
        im.save(dst, quality=92)


def _process(job: tuple[Path, Path, Path, Path, int]) -> list[int]:
    img_src, lbl_src, img_dst, lbl_dst, max_side = job
    resize_image(img_src, img_dst, max_side)
    lines = lbl_src.read_text().splitlines() if lbl_src.exists() else []
    kept = remap_label_lines(lines)
    lbl_dst.write_text("\n".join(kept) + ("\n" if kept else ""))
    return [int(line.split()[0]) for line in kept]


def find_root(raw: Path) -> Path:
    """kagglehub may nest the dataset a level or two down; locate it by its split file."""
    hits = sorted(raw.rglob("train_files.txt"))
    if not hits:
        raise FileNotFoundError(f"train_files.txt not found under {raw}")
    return hits[0].parent


def read_names(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, required=True, help="SH17 root (contains images/, labels/, *_files.txt)")
    ap.add_argument("--out", type=Path, default=Path("data/sh17_ppe"))
    ap.add_argument("--max-side", type=int, default=1280)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()

    root = find_root(args.raw)
    available = {p.name for p in (root / "images").iterdir()}
    train_all = [n for n in read_names(root / "train_files.txt") if n in available]
    test = [n for n in read_names(root / "val_files.txt") if n in available]
    train, val = split_names(train_all, args.val_frac, args.seed)
    splits = {"train": train, "val": val, "test": test}

    jobs = []
    for split, names in splits.items():
        (args.out / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.out / "labels" / split).mkdir(parents=True, exist_ok=True)
        for name in names:
            stem = Path(name).stem
            jobs.append((
                root / "images" / name,
                root / "labels" / f"{stem}.txt",
                args.out / "images" / split / f"{stem}.jpg",
                args.out / "labels" / split / f"{stem}.txt",
                args.max_side,
            ))

    counts = {s: Counter() for s in splits}
    split_of = [s for s, names in splits.items() for _ in names]
    with ProcessPoolExecutor(args.workers) as pool:
        for i, (split, classes) in enumerate(zip(split_of, pool.map(_process, jobs, chunksize=16)), 1):
            counts[split].update(classes)
            if i % 200 == 0 or i == len(jobs):
                print(f"\r{i}/{len(jobs)} images", end="", flush=True)
    print()

    out = args.out.resolve()
    yaml_text = f"path: {out}\ntrain: images/train\nval: images/val\ntest: images/test\n\nnames:\n"
    yaml_text += "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASS_NAMES))
    (args.out / "sh17_ppe.yaml").write_text(yaml_text)

    print(f"{'split':<6} {'images':>7} " + " ".join(f"{n:>12}" for n in CLASS_NAMES))
    for split, names in splits.items():
        print(f"{split:<6} {len(names):>7} " + " ".join(f"{counts[split][i]:>12}" for i in range(len(CLASS_NAMES))))
    print(f"Dataset config: {args.out / 'sh17_ppe.yaml'}")


if __name__ == "__main__":
    main()
