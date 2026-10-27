"""Turn a slice of the annotated video into a small GIF for the README.

    python make_gif.py --video output/demo/annotated.mp4 --start 5 --seconds 6 --out docs/demo.gif
"""

import argparse
from pathlib import Path

import cv2
from PIL import Image


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--fps", type=float, default=8.0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--out", type=Path, default=Path("docs/demo.gif"))
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, round(src_fps / args.fps))
    cap.set(cv2.CAP_PROP_POS_MSEC, args.start * 1000)
    frames = []
    for i in range(int(args.seconds * src_fps)):
        ok, f = cap.read()
        if not ok:
            break
        if i % step == 0:
            h, w = f.shape[:2]
            f = cv2.resize(f, (args.width, round(h * args.width / w)), interpolation=cv2.INTER_AREA)
            frames.append(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
    if not frames:
        raise SystemExit("no frames read")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(args.out, save_all=True, append_images=frames[1:], duration=round(1000 * step / src_fps), loop=0)
    print(f"{len(frames)} frames -> {args.out}")


if __name__ == "__main__":
    main()
