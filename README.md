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

## Revision analyses (October 2026)

These analyses were added in response to peer review. They use the saved per-frame detections of the locked test
(`trc_kitti/output_locked/predictions` and `camera_yolo`, which are not committed) and are pure NumPy/SciPy, so no GPU
or PyTorch is needed for them. `revision/analysis_r1.py` re-implements `run_benchmark.py evaluate` with per-object
logging and reproduces `metrics_locked.json` exactly; `revision/stats_r1.py` computes paired cluster-bootstrap
differences (2,000 resamples over frames), exact McNemar tests with Holm correction, near-field localization on the
common matched subset, the region-of-interest proxy accuracy, gate variants (30–70 m; 70 m verifies every camera
candidate), matched-false-positive operating points, thresholds selected on the calibration subset, far-field
pedestrian/cyclist failure analysis and latency dispersion.

```powershell
$env:TRC_EXPERIMENT_ROOT = (Get-Location).Path
python trc_kitti/revision/analysis_r1.py locked      # evaluation of the locked-test frames
python trc_kitti/revision/analysis_r1.py calib       # calibration-subset frames (gate/threshold selection)
python trc_kitti/revision/stats_r1.py                # writes trc_kitti/revision/results/r1_results.json
```

Committed results: `trc_kitti/revision/results/r1_results.json` (primary LiDAR input) and
`r1_results_redlidar.json` (field-of-view-reduced LiDAR input), with the printed tables in `stats_out*.txt`.

### KITTI 2D/BEV/3D average precision and the LiDAR input

`run_pointpillars_3d.py` saves the full 3D boxes of the CPU PointPillars path, `eval_3d_ap.py` scores them with the
vendored 40-point KITTI evaluation (`PointPillars/evaluate.py:do_eval`; only `Tensor.cuda()` is made a no-op), and
`make_reduced_lidar_predictions.py` converts the 3D outputs into the per-frame prediction files used by the benchmark.

```powershell
python trc_kitti/run_pointpillars_3d.py --output trc_kitti/output_3d_unreduced               # input of the primary experiment
python trc_kitti/run_pointpillars_3d.py --reduced --output trc_kitti/output_3d_reduced        # clouds reduced to the camera FOV
python trc_kitti/eval_3d_ap.py --pred trc_kitti/output_3d_reduced --out trc_kitti/output_3d_eval_reduced --subset all
python trc_kitti/make_reduced_lidar_predictions.py --pred3d trc_kitti/output_3d_reduced --output trc_kitti/output_locked_reducedlidar
$env:TRC_PRED_DIR = "trc_kitti/output_locked_reducedlidar/predictions"; $env:TRC_TAG = "_redlidar"
python trc_kitti/revision/analysis_r1.py locked; python trc_kitti/revision/stats_r1.py
```

Findings (see `trc_kitti/output_3d_eval/*/eval_results.txt` and `ap_summary.json`):

- With the reference input (point clouds reduced to the camera field of view, as in the upstream evaluation) the CPU path
  reproduces the published 2D AP of the checkpoint within 0.15 points overall (Easy/Moderate/Hard 80.43/74.48/71.35 against
  80.51/74.61/71.48); BEV and 3D AP are 0.9–2.8 points lower overall (3D 71.82/61.54/56.92 against 73.33/62.78/59.63).
- The primary experiment fed **unreduced** point clouds (`run_benchmark.py infer`). This lowers Moderate 2D AP by 4–6 points
  (Car 89.33→85.27, Pedestrian 61.35→54.98, Cyclist 72.75→67.13) and increases LiDAR false positives
  (1,855 against 1,272 on the locked test). The manuscript reports both configurations.
- Latencies were measured once, on the original workstation, for the primary experiment. The LiDAR latency of the re-runs
  (about 0.2 s per frame) is not comparable and is not used.

Environment of the revision runs: Python 3.12, NumPy 2.5.3, SciPy, Pillow, PyTorch 2.14.1+cpu, numba 0.68.0
(`PointPillars/pointpillars/utils/process.py` imports numba). A 12-frame check confirmed that this setup reproduces all
109 saved detections of the original run (identical labels, scores within 1e-5).

### File integrity

`MANIFEST.sha256` lists SHA-256 hashes of all committed files except itself. It was computed on a Windows working tree;
text files have CRLF line endings there, so on other systems compare after normalising line endings.
