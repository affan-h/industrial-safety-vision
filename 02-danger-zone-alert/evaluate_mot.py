"""Score zone-dwell alerts against a benchmark video with human-annotated person tracks.

Unlike evaluate_events.py (hand-written incident times), the ground truth here is computed:
every annotated person's foot point goes through the same zone polygon, dwell limit and exit
grace as the alert system (read from the same config file). Each stay longer than the limit
is one real incident, lasting from the moment the limit is crossed until the person leaves.

An alert is matched to a person by box overlap (IoU >= --iou with that person's annotated box
in the alert frame), so an alert only counts if it is on the right person at the right time:
  correct    - first alert on a real incident of that person (within --tol seconds)
  repeat     - another alert on an incident already caught (cooldown reminder or ID switch)
  false alarm- the person was not over the limit, or no annotated person is there
  missed     - a real incident that got no alert

Annotations use the MOTChallenge gt.txt format: frame, id, x, y, w, h, ... (frames from 1).

    python evaluate_mot.py --gt videos/pets09/gt.txt --video videos/pets09/PETS09-S2L1-raw.mp4 \\
        --config config_pets09.yaml --pred output/pets09_ppe
"""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

from rules import DwellTimer, foot_point, point_in_polygon


def load_mot_gt(path) -> dict[int, list[tuple[int, tuple]]]:
    """frame index (from 0, like the video reader) -> [(person id, (x1, y1, x2, y2)), ...]"""
    by_frame = defaultdict(list)
    for row in np.loadtxt(path, delimiter=",", ndmin=2):
        f, pid, x, y, w, h = row[:6]
        by_frame[int(f) - 1].append((int(pid), (x, y, x + w, y + h)))
    return by_frame


def gt_incidents(by_frame: dict, polygon, max_dwell_s: float, exit_grace_s: float, fps: float) -> list[dict]:
    """Every stay in the zone that exceeds the limit, from limit crossed to leaving the zone."""
    timer, open_inc, incidents = DwellTimer(exit_grace_s), {}, []
    for f in sorted(by_frame):
        t = f / fps
        for pid, box in by_frame[f]:
            inside = point_in_polygon(foot_point(box), polygon)
            dwell = timer.update(pid, inside, t)
            if dwell >= max_dwell_s and pid not in open_inc:
                open_inc[pid] = {"person": pid, "start_s": t, "end_s": t}
            if pid in open_inc:
                if inside:
                    open_inc[pid]["end_s"] = t
                elif pid not in timer.entered:  # seen outside for longer than the grace period
                    incidents.append(open_inc.pop(pid))
    incidents += open_inc.values()  # still inside when the video (or the track) ends
    return sorted(incidents, key=lambda i: i["start_s"])


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def who(box, frame: int, by_frame: dict, min_iou: float):
    """Annotated person under this box in this frame, or None."""
    best = max(((iou(box, b), pid) for pid, b in by_frame.get(frame, [])), default=(0.0, None))
    return best[1] if best[0] >= min_iou else None


def score(alerts: list[dict], incidents: list[dict], by_frame: dict, tol: float, min_iou: float) -> dict:
    caught = [False] * len(incidents)
    out = {"correct": 0, "repeat": 0, "false_alarm": 0, "missed": 0, "latency_s": [], "false_alarms": []}
    for a in sorted(alerts, key=lambda r: float(r["time_s"])):
        t, box = float(a["time_s"]), tuple(float(a[k]) for k in ("x1", "y1", "x2", "y2"))
        pid = who(box, int(a["frame"]), by_frame, min_iou)
        hits = [i for i, inc in enumerate(incidents)
                if inc["person"] == pid and inc["start_s"] - tol <= t <= inc["end_s"] + tol]
        new = [i for i in hits if not caught[i]]
        if new:
            caught[new[0]] = True
            out["correct"] += 1
            out["latency_s"].append(t - incidents[new[0]]["start_s"])
        elif hits:
            out["repeat"] += 1
        else:
            out["false_alarm"] += 1
            out["false_alarms"].append({"time_s": t, "track_id": a["track_id"], "person": pid})
    out["missed"] = caught.count(False)
    out["missed_incidents"] = [inc for inc, c in zip(incidents, caught) if not c]
    return out


def id_fragmentation(tracks: list[dict], by_frame: dict, min_iou: float = 0.5) -> dict[int, int]:
    """How many different tracker IDs each annotated person was given (1 = no ID switches)."""
    ids = defaultdict(set)
    for r in tracks:
        box = tuple(float(r[k]) for k in ("x1", "y1", "x2", "y2"))
        pid = who(box, int(r["frame"]), by_frame, min_iou)
        if pid is not None:
            ids[pid].add(r["track_id"])
    return {pid: len(s) for pid, s in sorted(ids.items())}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", required=True, help="MOTChallenge gt.txt")
    ap.add_argument("--video", required=True, help="the video the annotations belong to (for size and fps)")
    ap.add_argument("--config", required=True, help="the config zone_alert.py was run with")
    ap.add_argument("--pred", type=Path, required=True, help="zone_alert.py output dir (events.csv, tracks.csv)")
    ap.add_argument("--exit-grace", type=float, help="must match the run, if it was overridden there")
    ap.add_argument("--tol", type=float, default=1.0, help="seconds of slack around each incident")
    ap.add_argument("--iou", type=float, default=0.3, help="min box overlap to say an alert is on a person")
    ap.add_argument("--nobody-wears-helmets", action="store_true",
                    help="every annotated person is bareheaded (e.g. PETS09): also report helmet-rule coverage")
    args = ap.parse_args()

    import cv2
    cap = cv2.VideoCapture(args.video)
    w, h, fps = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT), cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    cfg = yaml.safe_load(Path(args.config).read_text())
    (zone,) = cfg["zones"]
    polygon = [(x * w, y * h) for x, y in zone["polygon"]]
    grace = args.exit_grace if args.exit_grace is not None else cfg["alerts"]["exit_grace_s"]

    by_frame = load_mot_gt(args.gt)
    incidents = gt_incidents(by_frame, polygon, zone["max_dwell_s"], grace, fps)
    with open(args.pred / "events.csv", newline="") as f:
        events = list(csv.DictReader(f))
    with open(args.pred / "tracks.csv", newline="") as f:
        tracks = list(csv.DictReader(f))

    s = score([e for e in events if e["rule"] == "zone_dwell"], incidents, by_frame, args.tol, args.iou)
    n_alerts = s["correct"] + s["repeat"] + s["false_alarm"]
    precision = (s["correct"] + s["repeat"]) / n_alerts if n_alerts else float("nan")
    recall = s["correct"] / len(incidents) if incidents else float("nan")
    frag = id_fragmentation(tracks, by_frame)
    helmet = Counter(r["helmet"] for r in tracks)

    report = {
        "incidents": len(incidents), "alerts": n_alerts, **{k: s[k] for k in ("correct", "repeat", "false_alarm", "missed")},
        "precision": round(precision, 3), "recall": round(recall, 3),
        "median_latency_s": round(float(np.median(s["latency_s"])), 2) if s["latency_s"] else None,
        "ids_per_person": round(float(np.mean(list(frag.values()))), 2) if frag else None,
        "helmet_status_frames": dict(helmet),
        "no_helmet_alerts": sum(e["rule"] == "no_helmet" for e in events),
        "annotated_people": len({pid for boxes in by_frame.values() for pid, _ in boxes}),
        "false_alarms_detail": s["false_alarms"], "missed_detail": s["missed_incidents"],
    }
    if args.nobody_wears_helmets:
        alerted = [who(tuple(float(e[k]) for k in ("x1", "y1", "x2", "y2")), int(e["frame"]), by_frame, args.iou)
                   for e in events if e["rule"] == "no_helmet"]
        report["no_helmet_people_alerted"] = len({p for p in alerted if p is not None})
        report["no_helmet_alerts_on_nobody"] = alerted.count(None)
    (args.pred / "eval.json").write_text(json.dumps(report, indent=2))

    print(f"zone '{zone['name']}', limit {zone['max_dwell_s']}s, exit grace {grace}s: "
          f"{len(incidents)} real incidents in the annotations")
    print("| alerts | correct | repeat | false alarms | missed | precision | recall | median delay | IDs per person |")
    print("|---|---|---|---|---|---|---|---|---|")
    print(f"| {n_alerts} | {s['correct']} | {s['repeat']} | {s['false_alarm']} | {s['missed']} | {precision:.2f} | "
          f"{recall:.2f} | {report['median_latency_s']}s | {report['ids_per_person']} |")
    print(f"helmet status over all person-frames: {dict(helmet)}; no_helmet alerts: {report['no_helmet_alerts']}")
    if args.nobody_wears_helmets:
        print(f"helmet rule: {report['no_helmet_people_alerted']} of {report['annotated_people']} bareheaded people "
              f"alerted; {report['no_helmet_alerts_on_nobody']} alerts not on any annotated person")
    for fa in s["false_alarms"]:
        print(f"  false alarm  t={fa['time_s']:6.1f}s track #{fa['track_id']} (annotated person: {fa['person']})")
    for m in s["missed_incidents"]:
        print(f"  missed       person {m['person']} over the limit {m['start_s']:.1f}-{m['end_s']:.1f}s")


if __name__ == "__main__":
    main()
