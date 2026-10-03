# CPU compatibility notes

This directory vendors the MIT-licensed PointPillars implementation by
`zhulf0804/PointPillars`, with the inference operations used by this study
adapted for CPU execution.

- `pointpillars/ops/cpu_ops.py` implements deterministic hard voxelization and
  rotated non-maximum suppression in NumPy/PyTorch.
- `pointpillars/ops/__init__.py` exposes the CPU implementations instead of the
  custom CUDA extensions.
- `pointpillars/ops/iou3d_module.py` computes BEV overlap on CPU.
- `setup.py` installs a pure Python package and does not compile CUDA code.

The original CUDA/C++ source is retained only for upstream provenance. It is
not imported or compiled by the CPU package. The pretrained checkpoint is not
redistributed.
