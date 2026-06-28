# Camera-first range-gated perception: KITTI reproducibility package

This repository contains the code and non-restricted artifacts used for the
locked KITTI benchmark reported in *Testing Camera-First Range-Gated
Perception: A Locked KITTI Study of Recall, Geometry, and Response Time*.

## Included

- Reproduction and evaluation scripts in `trc_kitti/`.
- Frozen split identifiers: 1,886 calibration frames and 1,883 locked-test
  frames.
- Primary and sensitivity-analysis aggregate results in CSV and JSON format.
- Camera training arguments and epoch-level CSV logs.
- A deterministic CPU-compatible PointPillars inference implementation.
- Machine-readable protocol settings in `configs/benchmark_locked.yaml`.

## Not included

KITTI images, labels, calibration files, point clouds, frame-level predictions,
YOLO/PointPillars weights, and other large binaries are intentionally excluded.
The root `.gitignore` prevents these files from being committed accidentally.
Download KITTI from its official distribution and obtain model weights from
their original providers under the applicable terms.

## Expected data layout

Place the standard KITTI object-detection training data into the two split
directories used by the experiment:

```text
kitti_train/{image_2,label_2,calib,velodyne}/
kitti_val/{image_2,label_2,calib,velodyne}/
```

The `PointPillars/pointpillars/dataset/ImageSets/` files define the standard
3,712/3,769 train/validation partition. The validation IDs are then frozen by
SHA-256 into `trc_kitti/calibration_ids.txt` and
`trc_kitti/locked_test_ids.txt`.

## Environment

The reported experiment used the versions recorded in `environment.txt`.
Create an isolated environment and install the CPU dependencies:

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-cpu.txt
python -m pip install -e PointPillars
```

On Linux or macOS, use the platform-specific activation command. Install the
CPU PyTorch wheel appropriate for the platform if the default index selects a
different accelerator build.

## Reproduction sequence

Set optional environment variables when the data or weight locations differ
from the defaults:

```powershell
$env:TRC_EXPERIMENT_ROOT = (Get-Location).Path
$env:TRC_YOLO_BASE_WEIGHTS = "external_weights/yolo11n.pt"
$env:TRC_CAMERA_CHECKPOINT = "external_weights/camera_locked_converged_best.pt"
```

Prepare the camera dataset and freeze the protocol:

```powershell
python trc_kitti/prepare_yolo.py
python trc_kitti/prepare_locked_protocol.py
```

Train and validate the converged camera detector:

```powershell
python trc_kitti/train_camera_converged.py
python trc_kitti/validate_camera_locked.py
```

Generate resumable frame-level predictions and evaluate only the locked IDs:

```powershell
python trc_kitti/run_benchmark.py infer-yolo --data kitti_val `
  --output trc_kitti/output_locked `
  --yolo-checkpoint external_weights/camera_locked_converged_best.pt

python trc_kitti/run_benchmark.py infer --data kitti_val `
  --output trc_kitti/output_locked `
  --checkpoint external_weights/pointpillars_epoch_160.pth `
  --threads 14 --skip-camera

python trc_kitti/run_benchmark.py evaluate --data kitti_val `
  --output trc_kitti/output_locked --camera-score 0.25 `
  --lidar-score 0.50 --metrics-tag locked `
  --id-list trc_kitti/locked_test_ids.txt
```

Generated `.npz` predictions are intentionally ignored. The committed files
`trc_kitti/output_locked/metrics_locked.{csv,json}` and
`class_metrics_locked.csv` are the aggregate primary results. Files with
`gate30`, `gate50`, `no_roi`, and `no_fallback` suffixes contain sensitivity
analyses. `trc_kitti/output/` preserves the earlier convergence and component
checks referenced during manuscript development.

## Important interpretation limits

The ROI range measurement is a camera-directed subset of synchronized KITTI
LiDAR returns, not an independent narrow-beam laser. Latency was measured on a
CPU-only sequential research implementation. KITTI object data are evaluated
as single annotated frames; this package does not validate temporal tracking,
vehicle control, or a deployment safety case.

## Third-party code

`PointPillars/` is derived from the MIT-licensed
[`zhulf0804/PointPillars`](https://github.com/zhulf0804/PointPillars). Its
original license is retained in `PointPillars/LICENSE`. See
`PointPillars/CPU_NOTES.md` for the CPU changes. No third-party model weights
are redistributed.
