"""Watch a fixed-camera video and alert on SOP violations.

Rules:
  - zone_dwell: a worker stays inside a danger zone longer than the allowed time
  - no_helmet:  a worker's head is seen with no helmet on it in k of the last n frames

Outputs (in --out): annotated video, events.csv, one snapshot per alert, and tracks.csv
(every tracked person in every frame, for offline evaluation).

    python zone_alert.py --source videos/demo.mp4
    python zone_alert.py --source videos/demo.mp4 --no-filter   # compare: alert on every raw frame
"""

import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

from rules import SafetyRules, Zone

GREEN, ORANGE, RED, WHITE = (60, 180, 75), (0, 165, 255), (40, 40, 230), (255, 255, 255)


def load_rules(cfg: dict, w: int, h: int, have_helmet_classes: bool, no_filter: bool,
               exit_grace_s: float | None = None) -> SafetyRules:
    zones = [Zone(z["name"], [(x * w, y * h) for x, y in z["polygon"]], z["sop"], float(z["max_dwell_s"]))
             for z in cfg.get("zones", [])]
    hr = cfg.get("helmet_rule", {})
    helmet_sop = hr.get("sop") if hr.get("enabled") and have_helmet_classes else None
    k, n = (1, 1) if no_filter else (hr.get("k", 8), hr.get("n", 12))
    a = cfg.get("alerts", {})
    grace = exit_grace_s if exit_grace_s is not None else a.get("exit_grace_s", 0.5)
    return SafetyRules(zones, helmet_sop, k, n, a.get("cooldown_s", 15), grace)


def role_class_ids(model_names: dict, mapping: dict) -> dict[str, set[int]]:
    by_name = {v: k for k, v in model_names.items()}
    roles = {"person": set(), "helmet": set(), "head": set()}
    for name, role in mapping.items():
        if name in by_name:
            roles[role].add(by_name[name])
        else:
            print(f"[warn] model has no class '{name}' (role {role}) - skipped")
    return roles


def draw_zones(img, zones: list[Zone]):
    overlay = img.copy()
    for z in zones:
        pts = np.array(z.polygon, np.int32)
        cv2.fillPoly(overlay, [pts], (0, 0, 200))
        cv2.polylines(img, [pts], True, RED, 2)
        x, y = pts.min(axis=0)
        cv2.putText(img, z.name, (int(x) + 4, int(y) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, RED, 2)
    cv2.addWeighted(overlay, 0.18, img, 0.82, 0, dst=img)


def draw_person(img, tid: int, box, st):
    color = RED if st.violations else ORANGE if st.zone else GREEN
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    helmet = {True: "helmet", False: "NO HELMET", None: ""}[st.helmet]
    parts = [f"#{tid}", f"{st.dwell_s:.1f}s" if st.zone else "", helmet]
    label = " ".join(p for p in parts if p)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
    cv2.putText(img, label, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)
    cv2.circle(img, (int((x1 + x2) / 2), y2), 4, color, -1)  # foot point used for the zone test


def draw_hud(img, t: float, n_people: int, n_violating: int, n_events: int, fps: float):
    lines = [f"t={t:6.1f}s  people={n_people}  violating={n_violating}", f"alerts={n_events}  {fps:4.1f} FPS"]
    cv2.rectangle(img, (0, 0), (360, 48), (0, 0, 0), -1)
    for i, s in enumerate(lines):
        cv2.putText(img, s, (8, 20 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="video file, RTSP URL, or webcam index")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--weights", help="override model.weights from the config")
    ap.add_argument("--out", type=Path, help="output dir (default: output/<video name>)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--no-filter", action="store_true", help="disable k-of-n smoothing (for comparison)")
    ap.add_argument("--helmet-conf", type=float, help="override model.helmet_conf (e.g. 0.35 to compare)")
    ap.add_argument("--exit-grace", type=float, help="override alerts.exit_grace_s (e.g. 0 to compare)")
    ap.add_argument("--show", action="store_true", help="display a live window")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    mcfg = cfg["model"]
    model = YOLO(args.weights or mcfg["weights"])
    roles = role_class_ids(model.names, mcfg["classes"])
    if not roles["person"]:
        raise SystemExit("model has no class mapped to role 'person'")

    live = args.source.isdigit() or "://" in args.source
    cap = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    rules = load_rules(cfg, w, h, bool(roles["helmet"] and roles["head"]), args.no_filter, args.exit_grace)
    if rules.helmet_sop is None:
        print("[info] helmet rule off (disabled in config or model lacks helmet/head classes)")

    name = "live" if live else Path(args.source).stem
    out = args.out or Path("output") / (name + ("_nofilter" if args.no_filter else ""))
    (out / "snapshots").mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    csv_file = open(out / "events.csv", "w", newline="")
    log = csv.writer(csv_file)
    log.writerow(["time_s", "frame", "track_id", "rule", "zone", "sop", "detail", "snapshot", "x1", "y1", "x2", "y2"])
    tracks_file = open(out / "tracks.csv", "w", newline="")
    tracks_log = csv.writer(tracks_file)
    tracks_log.writerow(["frame", "time_s", "track_id", "x1", "y1", "x2", "y2", "zone", "dwell_s", "helmet"])

    keep = sorted(roles["person"] | roles["helmet"] | roles["head"])
    # Asymmetric thresholds: weak helmet evidence is enough to suppress a no-helmet alarm,
    # because a missed helmet turns straight into a false alert.
    conf = mcfg.get("conf", 0.35)
    helmet_conf = args.helmet_conf if args.helmet_conf is not None else mcfg.get("helmet_conf", conf)
    n_events, frame_idx, t_model, t_rest = 0, 0, 0.0, 0.0
    start = time.monotonic()
    while True:
        ok, frame = cap.read()
        if not ok or (args.max_frames and frame_idx >= args.max_frames):
            break
        t = time.monotonic() - start if live else frame_idx / fps

        t0 = time.perf_counter()
        res = model.track(frame, persist=True, tracker=cfg.get("tracker", "bytetrack.yaml"), classes=keep,
                          conf=min(conf, helmet_conf), imgsz=mcfg.get("imgsz", 640), device=args.device,
                          verbose=False)[0]
        t1 = time.perf_counter()

        b = res.boxes
        xyxy, cls, scores = b.xyxy.cpu().numpy(), b.cls.cpu().numpy().astype(int), b.conf.cpu().numpy()
        ids = b.id.cpu().numpy().astype(int) if b.id is not None else np.full(len(cls), -1)
        people = {int(i): tuple(bx) for bx, c, s, i in zip(xyxy, cls, scores, ids)
                  if c in roles["person"] and s >= conf and i >= 0}
        helmets = [tuple(bx) for bx, c, s in zip(xyxy, cls, scores) if c in roles["helmet"] and s >= helmet_conf]
        heads = [tuple(bx) for bx, c, s in zip(xyxy, cls, scores) if c in roles["head"] and s >= conf]

        events, states = rules.update(frame_idx, t, people, helmets, heads)

        draw_zones(frame, rules.zones)
        for tid, box in people.items():
            draw_person(frame, tid, box, states[tid])
            st = states[tid]
            tracks_log.writerow([frame_idx, f"{t:.3f}", tid, *(f"{v:.1f}" for v in box), st.zone or "",
                                 f"{st.dwell_s:.2f}", {True: "helmet", False: "no_helmet", None: "unknown"}[st.helmet]])
        n_events += len(events)
        done = frame_idx + 1
        draw_hud(frame, t, len(people), sum(bool(s.violations) for s in states.values()), n_events,
                 done / (t_model + t_rest + 1e-9))

        for e in events:
            snap = out / "snapshots" / f"{e.time_s:07.2f}s_{e.rule}_id{e.track_id}.jpg"
            cv2.imwrite(str(snap), frame)
            log.writerow([f"{e.time_s:.2f}", e.frame, e.track_id, e.rule, e.zone, e.sop, e.detail, snap.name,
                          *(f"{v:.1f}" for v in people[e.track_id])])
            print(f"[ALERT] t={e.time_s:6.1f}s worker #{e.track_id} {e.rule:<10} {e.zone or '-'} | {e.sop} | {e.detail}")

        writer.write(frame)
        if args.show:
            cv2.imshow("zone alert", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        t_rest += time.perf_counter() - t1
        t_model += t1 - t0
        frame_idx += 1

    cap.release()
    writer.release()
    csv_file.close()
    tracks_file.close()
    cv2.destroyAllWindows()
    if frame_idx:
        print(f"\n{frame_idx} frames, {n_events} alerts -> {out}")
        print(f"avg per frame: detect+track {1000 * t_model / frame_idx:.1f} ms, "
              f"rules+drawing+I/O {1000 * t_rest / frame_idx:.1f} ms "
              f"({frame_idx / (t_model + t_rest):.1f} FPS overall)")


if __name__ == "__main__":
    main()
