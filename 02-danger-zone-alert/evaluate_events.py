"""Score alerts against hand-labelled ground truth: precision, recall, false alarms.

Ground-truth CSV (one row per real incident, times in seconds of video):

    rule,start_s,end_s,note
    zone_dwell,12.0,19.5,worker walks into crane bay and waits
    no_helmet,31.0,40.0,worker removes helmet

An alert is a true positive if it has the same rule and falls inside
[start_s - tol, end_s + tol] of an incident not already matched. Extra alerts for the same
incident (e.g. after a tracker ID switch) count as false alarms.

    python evaluate_events.py --pred output/demo/events.csv --gt videos/demo_gt.csv
"""

import argparse
import csv
from collections import defaultdict


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def score(preds: list[dict], gts: list[dict], tol: float) -> dict[str, dict]:
    used = [False] * len(gts)
    stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    for p in sorted(preds, key=lambda r: float(r["time_s"])):
        t = float(p["time_s"])
        hit = next((i for i, g in enumerate(gts) if not used[i] and g["rule"] == p["rule"]
                    and float(g["start_s"]) - tol <= t <= float(g["end_s"]) + tol), None)
        if hit is None:
            stats[p["rule"]]["fp"] += 1
        else:
            used[hit] = True
            stats[p["rule"]]["tp"] += 1
    for g, u in zip(gts, used):
        if not u:
            stats[g["rule"]]["fn"] += 1
    total = {k: sum(s[k] for s in stats.values()) for k in ("tp", "fp", "fn")}
    return {**stats, "all": total}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", required=True, help="events.csv written by zone_alert.py")
    ap.add_argument("--gt", required=True)
    ap.add_argument("--tol", type=float, default=1.0, help="seconds of slack around each incident")
    args = ap.parse_args()

    results = score(read_csv(args.pred), read_csv(args.gt), args.tol)
    print("| rule | TP | false alarms | missed | precision | recall |\n|---|---|---|---|---|---|")
    for rule, s in results.items():
        p = s["tp"] / (s["tp"] + s["fp"]) if s["tp"] + s["fp"] else float("nan")
        r = s["tp"] / (s["tp"] + s["fn"]) if s["tp"] + s["fn"] else float("nan")
        print(f"| {rule} | {s['tp']} | {s['fp']} | {s['fn']} | {p:.2f} | {r:.2f} |")


if __name__ == "__main__":
    main()
