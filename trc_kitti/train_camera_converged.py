import os
from pathlib import Path

from ultralytics import YOLO


HERE = Path(os.environ.get("TRC_RUN_ROOT", Path(__file__).resolve().parent))
ROOT = Path(os.environ.get("TRC_EXPERIMENT_ROOT", HERE.parents[0]))
MODEL = Path(os.environ.get("TRC_YOLO_BASE_WEIGHTS", ROOT / "yolo11n.pt"))

model = YOLO(str(MODEL))
model.train(
    data=str(ROOT / "kitti_yolo" / "kitti_locked.yaml"),
    epochs=30,
    patience=7,
    batch=16,
    imgsz=640,
    device="cpu",
    workers=4,
    project=str(HERE / "training"),
    name="camera_locked_converged",
    exist_ok=False,
    pretrained=True,
    optimizer="auto",
    seed=2026,
    deterministic=True,
    cache=False,
    plots=True,
    save=True,
)
