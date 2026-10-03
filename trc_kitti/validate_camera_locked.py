import os
from pathlib import Path

from ultralytics import YOLO


HERE = Path(os.environ.get("TRC_RUN_ROOT", Path(__file__).resolve().parent))
ROOT = Path(os.environ.get("TRC_EXPERIMENT_ROOT", HERE.parents[0]))
RUN = HERE / "training" / "camera_locked_converged"
CHECKPOINT = Path(os.environ.get("TRC_CAMERA_CHECKPOINT", RUN / "weights" / "best.pt"))
RUN_NAME = os.environ.get("TRC_VALIDATION_NAME", "camera_locked_test")

model = YOLO(str(CHECKPOINT))
model.val(
    data=str(ROOT / "kitti_yolo" / "kitti_locked.yaml"),
    split="test",
    imgsz=640,
    batch=16,
    device="cpu",
    workers=4,
    project=str(HERE / "training"),
    name=RUN_NAME,
    exist_ok=True,
    plots=True,
    save_json=False,
)
