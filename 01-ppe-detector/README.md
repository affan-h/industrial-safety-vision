# PPE Detector for Industrial Workers, and How It Fails on Plant Cameras

Fine-tunes a pretrained **YOLO11n** detector to find **people, bare heads, helmets and safety vests** in industrial scenes. Then it measures how much accuracy is lost under conditions real plant cameras face: **low light, smoke/haze and motion blur**.

> **Question:** a PPE detector that scores well on clean photos can still fail on dark, dusty factory feeds. How big is that drop, and which classes suffer most?

<!-- After the Colab run, replace these with real images from results/ -->
![Robustness results](results/robustness.png)
![Same image under each condition](results/robustness_examples.png)

## Results

> ⚠️ Fill these in from `results/` after running `colab_train.ipynb`. Don't edit the numbers by hand.

**Held-out test set** (1,620 images, never used for training or model selection): copy from `results/test_metrics.md`

| class | precision | recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| all | – | – | – | – |

**Robustness:** copy from `results/robustness.md`

**Where it fails:** copy from `results/failures/error_summary.md`, and write one line per failure pattern you see in `results/failures/worst_*.jpg`, e.g.:
- *Small helmets on distant workers are missed most (x% miss rate for small vs y% for large).*
- *…*

**Takeaways (write 3 bullets from your numbers):**
- Which condition hurts most, and which class is most fragile?
- What would you do about it: low-light augmentation, a higher input resolution for small PPE, or camera placement/IR?

## Method

| Step | Details |
|---|---|
| Data | [SH17](https://github.com/ahmadmughees/SH17dataset): 8,099 manufacturing images (CC BY-NC-SA 4.0). Kept 4 of 17 classes: `person`, `head`, `helmet`, `safety-vest`. **`head` = a bare head**, which the model must learn to tell apart from a helmet |
| Splits | Official train list → 90% train / 10% val (seeded). Official val list → **held-out test** |
| Preprocessing | Images downscaled to a 1280 px long side (originals reach 8192 px). This cuts disk use and load time without affecting 640 px training |
| Model | YOLO11n (2.6 M params), COCO-pretrained, fine-tuned 50 epochs at 640 px, early stopping (patience 15), seed 0 |
| Robustness | The test set is resized to 640 px, then degraded at two severities: **low light** (brightness ×0.35 / ×0.15 + sensor noise), **haze** (scattering model `I = J·t + A·(1−t)`, t = 0.55 / 0.3), **motion blur** (9 / 21 px kernel at a random angle). Fixed seeds, so it's reproducible |
| Error analysis | Predictions matched to ground truth per class at IoU ≥ 0.5. Misses broken down by object size (small < 1% of image area, medium < 5%, large) |

## Run it

**Train on a free GPU (recommended):** open `colab_train.ipynb` in Google Colab, select a T4 GPU and run all cells (about 1.5–2 h). It downloads SH17, trains, evaluates, runs the robustness test and the failure gallery, then zips `results/` and `weights/` for download.

**Locally** (a CPU is enough for a smoke test, not for real training):
```bash
pip install -r requirements.txt
python -m pytest -q tests                               # unit tests
python download_sh17.py --out data/sh17_raw --limit 40  # small sample, no Kaggle account needed
python prepare_data.py --raw data/sh17_raw --out data/sh17_ppe
python train.py --data data/sh17_ppe/sh17_ppe.yaml --epochs 1 --imgsz 320 --device cpu --out weights/smoke.pt
python evaluate.py      --weights weights/smoke.pt --data data/sh17_ppe/sh17_ppe.yaml --imgsz 320
python robustness.py    --weights weights/smoke.pt --data data/sh17_ppe/sh17_ppe.yaml --imgsz 320
python failure_gallery.py --weights weights/smoke.pt --data data/sh17_ppe/sh17_ppe.yaml --imgsz 320
```

## Files

| File | Purpose |
|---|---|
| `download_sh17.py` | Fetch SH17 files from Kaggle (a sample, or everything) |
| `prepare_data.py` | Filter to 4 classes, remap IDs, split, resize, write `sh17_ppe.yaml` |
| `train.py` | Fine-tune YOLO11n and copy the best weights to `weights/` |
| `evaluate.py` | Per-class precision/recall/mAP on the test split → `results/test_metrics.*` |
| `degrade.py` | Low-light / haze / motion-blur functions |
| `robustness.py` | Evaluate under each degradation → table, chart, example strip |
| `failure_gallery.py` | Worst/best predictions drawn, plus misses by object size |
| `colab_train.ipynb` | The whole pipeline on a Colab GPU |
| `tests/` | Unit tests for label remapping, splitting, degradations, box matching |

## Limitations

- SH17 is stock photography of workers, not CCTV. It's clearer and closer-up than real plant footage, so these numbers are optimistic for a real deployment.
- The degradations are synthetic approximations, not real smoke or IR footage.
- `head` vs `helmet` only checks whether something covers the head. It can't tell whether a hard hat is the correct rated type, or whether the chin strap is fastened.
- SH17 is licensed CC BY-NC-SA 4.0 (non-commercial) and Ultralytics is AGPL-3.0. Both are fine for a portfolio, but not for a commercial product as-is.

## Used by

`../02-danger-zone-alert` uses `weights/ppe_yolo11n.pt` to flag workers without helmets in a video.
