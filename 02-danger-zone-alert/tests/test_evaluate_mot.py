from evaluate_mot import gt_incidents, id_fragmentation, score

ZONE = [(100, 100), (200, 100), (200, 200), (100, 200)]
INSIDE = (140, 120, 160, 150)    # feet at (150, 150)
OUTSIDE = (340, 120, 360, 150)
FPS = 10


def frames(*segments):
    """segments of (person id, box, number of frames) laid end to end, starting at frame 0."""
    by_frame, f = {}, 0
    for pid, box, n in segments:
        for _ in range(n):
            by_frame.setdefault(f, []).append((pid, box))
            f += 1
    return by_frame


def test_incident_only_for_stays_over_the_limit():
    gt = frames((1, INSIDE, 20), (1, OUTSIDE, 10), (1, INSIDE, 60), (1, OUTSIDE, 10))
    (inc,) = gt_incidents(gt, ZONE, max_dwell_s=3, exit_grace_s=0.5, fps=FPS)
    # second stay: enters at frame 30 (3.0 s), crosses the 3 s limit at 6.0 s, last inside at 8.9 s
    assert inc["person"] == 1
    assert abs(inc["start_s"] - 6.0) < 1e-9 and abs(inc["end_s"] - 8.9) < 1e-9


def test_short_exit_within_grace_does_not_split_the_stay():
    gt = frames((1, INSIDE, 25), (1, OUTSIDE, 3), (1, INSIDE, 25))
    (inc,) = gt_incidents(gt, ZONE, max_dwell_s=3, exit_grace_s=0.5, fps=FPS)
    assert abs(inc["start_s"] - 3.0) < 1e-9


def alert(t, box, track_id=7):
    x1, y1, x2, y2 = box
    return {"time_s": str(t), "frame": str(round(t * FPS)), "track_id": str(track_id),
            "x1": x1, "y1": y1, "x2": x2, "y2": y2}


def test_score_needs_the_right_person_not_just_the_right_time():
    gt = frames((1, INSIDE, 60))
    for f in range(60):
        gt[f].append((2, OUTSIDE))
    incidents = gt_incidents(gt, ZONE, max_dwell_s=3, exit_grace_s=0.5, fps=FPS)
    alerts = [alert(3.2, INSIDE), alert(4.0, INSIDE), alert(4.5, OUTSIDE, track_id=9)]
    s = score(alerts, incidents, gt, tol=1.0, min_iou=0.3)
    assert (s["correct"], s["repeat"], s["false_alarm"], s["missed"]) == (1, 1, 1, 0)
    assert s["false_alarms"][0]["person"] == 2


def test_missed_incident():
    gt = frames((1, INSIDE, 60))
    incidents = gt_incidents(gt, ZONE, max_dwell_s=3, exit_grace_s=0.5, fps=FPS)
    assert score([], incidents, gt, tol=1.0, min_iou=0.3)["missed"] == 1


def test_id_fragmentation_counts_tracker_ids_per_person():
    gt = frames((1, INSIDE, 4))
    tracks = [{"frame": str(f), "track_id": tid, "x1": 140, "y1": 120, "x2": 160, "y2": 150}
              for f, tid in enumerate(["3", "3", "8", "8"])]
    assert id_fragmentation(tracks, gt) == {1: 2}
