# Industrial Safety Vision: Portfolio Projects

Two connected computer-vision projects for worker safety in heavy industry.

| Project | What it shows |
|---|---|
| [`01-ppe-detector`](01-ppe-detector) | Fine-tuning YOLO11n (PyTorch) to detect people, bare heads, helmets and safety vests. Evaluated on a held-out test set, then **stress-tested under low light, haze and motion blur**, with a failure gallery |
| [`02-danger-zone-alert`](02-danger-zone-alert) | Turning that detector into a working system: **tracking (ByteTrack) + danger-zone dwell rule + helmet rule**, with false-alarm control, SOP-linked alerts, snapshots, CSV logs and event-level precision/recall |

Setup (Python 3.10–3.12):
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r 01-ppe-detector/requirements.txt -r 02-danger-zone-alert/requirements.txt
```
Training needs a GPU: use `01-ppe-detector/colab_train.ipynb` on free Google Colab.
