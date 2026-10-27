# Danger-Zone & Helmet Alerts from a Fixed Camera

Watches a camera feed and raises an alert, tied to a specific SOP clause, when a worker:
- **stays inside a danger zone** (e.g. a crane bay) longer than allowed, or
- is **not wearing a helmet**.

Each alert produces a snapshot and a CSV log line (`time, worker ID, rule, zone, SOP clause, reason`), plus an annotated video. It reuses the PPE detector trained in [`../01-ppe-detector`](../01-ppe-detector).

<!-- Replace with your own clip: python make_gif.py --video output/<clip>/annotated.mp4 --start 2 --seconds 6 -->
![demo](docs/demo.gif)

## How it works

```
video / RTSP ─► YOLO detector (person, helmet, bare head) ─► ByteTrack (stable worker IDs)
                                                                   │
                       ┌───────────────────────────────────────────┤
                       ▼                                           ▼
     zone rule: foot point inside polygon?            helmet rule: helmet or bare head
     dwell timer per worker > limit?                  inside the person's head region?
                       │                              confirmed in k of last n frames?
                       └──────────────┬────────────────────────────┘
                                      ▼
                  cooldown per worker+rule ─► alert: snapshot + events.csv + console
```

## Design decisions (and why)

| Decision | Why |
|---|---|
| **Zone test uses the person's feet** (bottom-centre of the box), not the box centre | From an elevated camera, a person's box centre can lie over a floor zone while they're standing outside it |
| **Dwell time limit**, not instant alerts | Walking past a zone edge isn't a violation. Staying in a crane bay is |
| **Exit grace period** (0.5 s) | A box flickering across the zone edge, or briefly lost, shouldn't reset the timer. Only being *seen outside* for longer does |
| **Helmet status has three states**: helmet / bare head / *unknown* | Back turned, head occluded or too far away means "can't tell", which must not become an alert, or supervisors drown in false alarms |
| **k-of-n confirmation** (default 8 of 12 frames) | A single-frame "bare head" misdetection doesn't trigger anything. `--no-filter` turns it off so the effect can be measured |
| **Cooldown** per worker per rule | One incident = one alert, not 30 per second |
| **Rules in `config.yaml`, each citing an SOP clause** | Safety teams can change zones and limits without touching code, and every alert is traceable to a procedure |
| **Zone polygons in 0–1 coordinates** | The same config works if the stream resolution changes |
| **Rule logic in `rules.py` with no CV dependencies** | It's fully unit-tested (16 tests) without a GPU or video |

## Results

> ⚠️ Fill in from your own clip after recording it (see "Measure it" below). Don't invent numbers.

| run | alerts | correct | false alarms | missed | precision | recall |
|---|---|---|---|---|---|---|
| with k-of-n filter | – | – | – | – | – | – |
| `--no-filter` | – | – | – | – | – | – |

**Speed** (printed at the end of every run): a smoke test on a 2015 dual-core laptop CPU, 854×480 video, COCO `yolo11n`, ran at **≈4 FPS** (≈235 ms detect+track, ≈11 ms everything else per frame). Detection dominates, so a GPU or an exported ONNX/OpenVINO model is where speed-ups come from, not the rule logic.

**Failure modes seen in testing** (construction-site clip, CC BY-SA 4.0, Wikimedia Commons):
- Workers **behind mesh fencing or crouching were missed** by the generic COCO model at confidence 0.35.
- The visible worker's **track ID switched** mid-clip (#3 → #10), and the dwell timer restarts with a new ID. Handling this properly needs re-identification, or zone-level occupancy timing instead of per-ID timing.

## Run it

```bash
pip install -r requirements.txt
python -m pytest -q tests

# 1. draw the danger zone on your video and paste the polygon into config.yaml
python pick_zone.py --source videos/my_clip.mp4

# 2. run (uses ../01-ppe-detector/weights/ppe_yolo11n.pt by default)
python zone_alert.py --source videos/my_clip.mp4            # add --show for a live window
python zone_alert.py --source videos/my_clip.mp4 --no-filter  # same, without k-of-n smoothing

# person-only test with the stock COCO model (helmet rule switches itself off)
python zone_alert.py --source videos/my_clip.mp4 --weights yolo11n.pt

# 3. GIF for this README
python make_gif.py --video output/my_clip/annotated.mp4 --start 2 --seconds 6
```

Outputs go to `output/<clip name>/`: `annotated.mp4`, `events.csv`, `snapshots/`.

## Measure it

1. **Record a 1–2 minute clip** with a phone on a tripod (it must not move, like a CCTV camera). Act out several incidents: walk through the zone quickly (should *not* alert), stand in it for 5+ s (should alert), take a helmet off for 10 s (should alert), turn your back with no helmet (ideally *no* alert).
2. **Write the ground truth** in `videos/my_clip_gt.csv`:
   ```
   rule,start_s,end_s,note
   zone_dwell,12.0,19.5,stands in crane bay
   no_helmet,31.0,40.0,helmet removed
   ```
3. **Score both runs:**
   ```bash
   python evaluate_events.py --pred output/my_clip/events.csv          --gt videos/my_clip_gt.csv
   python evaluate_events.py --pred output/my_clip_nofilter/events.csv --gt videos/my_clip_gt.csv
   ```
   The difference in false alarms between the two runs is the measured value of the k-of-n filter.

## Files

| File | Purpose |
|---|---|
| `zone_alert.py` | Main loop: read video → detect+track → rules → draw → log |
| `rules.py` | Point-in-polygon, foot point, helmet association, dwell timer, k-of-n, cooldown |
| `config.yaml` | Model, class→role mapping, zones, SOP text, thresholds |
| `pick_zone.py` | Click to draw a zone polygon on a frame |
| `evaluate_events.py` | Precision/recall of alerts vs. hand-labelled incidents |
| `make_gif.py` | GIF from the annotated video for the README |
| `tests/` | Unit tests for rules and scoring |

## Limitations and next steps

- Per-ID dwell timing breaks on tracker ID switches. Next step: re-identification, or "zone occupied for N s" timing.
- A single camera and a 2D zone can't judge height: a worker on a ladder above the zone counts as inside it.
- Helmet association is geometric (a helmet centre in the top 30% of the person box). Crowded scenes with overlapping people can attach a helmet to the wrong worker.
- Not tested on real CCTV, IR or night footage. See Project 1's robustness study for how detection degrades.
