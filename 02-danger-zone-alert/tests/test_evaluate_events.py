from evaluate_events import score


def test_score_counts_duplicates_as_false_alarms():
    gts = [{"rule": "zone_dwell", "start_s": "10", "end_s": "20"},
           {"rule": "no_helmet", "start_s": "30", "end_s": "40"}]
    preds = [{"rule": "zone_dwell", "time_s": "13.0"},
             {"rule": "zone_dwell", "time_s": "15.0"},   # same incident again (ID switch)
             {"rule": "no_helmet", "time_s": "5.0"}]     # nothing happening then
    r = score(preds, gts, tol=1.0)
    assert r["zone_dwell"] == {"tp": 1, "fp": 1, "fn": 0}
    assert r["no_helmet"] == {"tp": 0, "fp": 1, "fn": 1}
    assert r["all"] == {"tp": 1, "fp": 2, "fn": 1}


def test_tolerance_window():
    gts = [{"rule": "zone_dwell", "start_s": "10", "end_s": "20"}]
    assert score([{"rule": "zone_dwell", "time_s": "20.8"}], gts, tol=1.0)["all"]["tp"] == 1
    assert score([{"rule": "zone_dwell", "time_s": "21.5"}], gts, tol=1.0)["all"]["tp"] == 0
