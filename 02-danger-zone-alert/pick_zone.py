"""Click on a video frame to draw a danger-zone polygon; prints it for config.yaml.

Left click: add point. Right click / Backspace: undo. Enter or Space: finish. Q / Esc: quit.

    python pick_zone.py --source videos/demo.mp4 --frame 30
"""

import argparse

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True)
    ap.add_argument("--frame", type=int, default=0, help="frame index to draw on")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.source)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"cannot read frame {args.frame} of {args.source}")

    h, w = frame.shape[:2]
    scale = min(1.0, 1280 / w)
    base = cv2.resize(frame, (round(w * scale), round(h * scale)))
    points: list[tuple[int, int]] = []

    def on_mouse(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            points.pop()

    cv2.namedWindow("pick zone")
    cv2.setMouseCallback("pick zone", on_mouse)
    while True:
        img = base.copy()
        if points:
            pts = np.array(points, np.int32)
            cv2.polylines(img, [pts], len(points) > 2, (40, 40, 230), 2)
            for p in points:
                cv2.circle(img, p, 4, (0, 165, 255), -1)
        cv2.putText(img, "click points | right-click undo | Enter done | q quit", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow("pick zone", img)
        key = cv2.waitKey(20) & 0xFF
        if key in (8, 127) and points:
            points.pop()
        elif key in (13, 10, 32) and len(points) >= 3:
            break
        elif key in (ord("q"), 27):
            cv2.destroyAllWindows()
            return
    cv2.destroyAllWindows()

    sw, sh = base.shape[1], base.shape[0]
    norm = [[round(x / sw, 3), round(y / sh, 3)] for x, y in points]
    print("\nPaste into config.yaml under the zone:\n")
    print(f"    polygon: {norm}")


if __name__ == "__main__":
    main()
