# Industrial Safety Vision: Portfolio Projects

Two connected computer-vision projects for worker safety in heavy industry.

| Project | What it shows |
|---|---|
| [`01-ppe-detector`](01-ppe-detector) | Fine-tuning YOLO11n (PyTorch) to detect people, heads, helmets and safety vests. Held-out test mAP50 0.715, then **stress-tested under low light, haze and motion blur** (−87% in severe low light) and broken down by object size (small people missed 62% of the time vs 5% for large) |
| [`02-danger-zone-alert`](02-danger-zone-alert) | Turning that detector into a working system: **tracking (ByteTrack) + danger-zone dwell rule + helmet rule**, with false-alarm control, SOP-linked alerts, snapshots and CSV logs. **Measured on the PETS09 CCTV benchmark: precision 0.87, recall 0.91**, with each error traced to its cause |

Setup (Python 3.10–3.12):
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r 01-ppe-detector/requirements.txt -r 02-danger-zone-alert/requirements.txt
```
Training needs a GPU: run `01-ppe-detector/colab_train.ipynb` on free Google Colab or Kaggle.
