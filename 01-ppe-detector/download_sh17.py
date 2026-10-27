"""Download the SH17 PPE dataset (images, YOLO labels, split files) from Kaggle.

For the full dataset (~14 GB) use kagglehub on Colab/Kaggle (see colab_train.ipynb).
This script fetches files one by one without a Kaggle account, which is handy for
a small local sample to smoke-test the pipeline:

    python download_sh17.py --out data/sh17_raw --limit 40
"""

import argparse
import io
import urllib.parse
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DATASET = "mugheesahmad/sh17-dataset-for-ppe-detection"
BASE_URL = f"https://www.kaggle.com/api/v1/datasets/download/{DATASET}/"
SPLIT_FILES = ["train_files.txt", "val_files.txt"]


def fetch(remote_path: str, out_dir: Path) -> Path:
    dst = out_dir / remote_path
    if dst.exists() and dst.stat().st_size > 0 and not zipfile.is_zipfile(dst):
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    url = BASE_URL + urllib.parse.quote(remote_path, safe="")
    with urllib.request.urlopen(url, timeout=120) as r:
        data = r.read()
    if data[:4] == b"PK\x03\x04":  # Kaggle zips larger single-file downloads
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            data = zf.read(zf.namelist()[0])
    dst.write_bytes(data)
    return dst


def read_names(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("data/sh17_raw"))
    ap.add_argument("--limit", type=int, default=40, help="total images to fetch (split ~80/20 train/val)")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    for name in SPLIT_FILES:
        fetch(name, args.out)

    train = read_names(args.out / "train_files.txt")
    val = read_names(args.out / "val_files.txt")
    n_val = max(1, args.limit // 5)
    images = train[: args.limit - n_val] + val[:n_val]

    remote = []
    for img in images:
        remote += [f"images/{img}", f"labels/{Path(img).stem}.txt"]

    with ThreadPoolExecutor(args.workers) as pool:
        for i, _ in enumerate(pool.map(lambda p: fetch(p, args.out), remote), 1):
            print(f"\r{i}/{len(remote)} files", end="", flush=True)
    print(f"\nSaved {len(images)} images to {args.out}")


if __name__ == "__main__":
    main()
