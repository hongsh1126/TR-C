import math

import numpy as np
import torch
import torch.nn as nn


class Voxelization(nn.Module):
    """Deterministic CPU equivalent of the project's hard voxelizer."""

    def __init__(self, voxel_size, point_cloud_range, max_num_points,
                 max_voxels, deterministic=True):
        super().__init__()
        self.voxel_size = np.asarray(voxel_size, dtype=np.float32)
        self.point_cloud_range = np.asarray(point_cloud_range, dtype=np.float32)
        self.max_num_points = max_num_points
        self.max_voxels = max_voxels

    def forward(self, points):
        pts = points.detach().cpu().numpy()
        xyz = pts[:, :3]
        keep = np.all(xyz >= self.point_cloud_range[:3], axis=1)
        keep &= np.all(xyz < self.point_cloud_range[3:], axis=1)
        pts = pts[keep]
        coords = np.floor(
            (pts[:, :3] - self.point_cloud_range[:3]) / self.voxel_size
        ).astype(np.int32)

        max_voxels = self.max_voxels[0] if self.training else self.max_voxels[1]
        voxel_map = {}
        voxel_points = []
        voxel_coords = []
        counts = []
        for point, coord in zip(pts, coords):
            key = tuple(coord.tolist())
            idx = voxel_map.get(key)
            if idx is None:
                if len(voxel_points) >= max_voxels:
                    continue
                idx = len(voxel_points)
                voxel_map[key] = idx
                voxel_points.append(np.zeros((self.max_num_points, pts.shape[1]), dtype=np.float32))
                voxel_coords.append(coord)
                counts.append(0)
            if counts[idx] < self.max_num_points:
                voxel_points[idx][counts[idx]] = point
                counts[idx] += 1

        if not voxel_points:
            return (torch.empty((0, self.max_num_points, points.shape[1])),
                    torch.empty((0, 3), dtype=torch.int32),
                    torch.empty((0,), dtype=torch.int32))
        return (torch.from_numpy(np.stack(voxel_points)),
                torch.from_numpy(np.stack(voxel_coords)),
                torch.tensor(counts, dtype=torch.int32))


def _corners(box):
    x1, y1, x2, y2, angle = [float(v) for v in box]
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    hx, hy = (x2 - x1) / 2.0, (y2 - y1) / 2.0
    c, s = math.cos(angle), math.sin(angle)
    local = np.array([[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]])
    rot = np.array([[c, -s], [s, c]])
    return local @ rot.T + np.array([cx, cy])


def _cross(a, b, p):
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def _intersection(p1, p2, q1, q2):
    d1, d2 = p2 - p1, q2 - q1
    den = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(den) < 1e-9:
        return p2
    t = ((q1[0] - p1[0]) * d2[1] - (q1[1] - p1[1]) * d2[0]) / den
    return p1 + t * d1


def _clip(subject, clipper):
    output = [p for p in subject]
    for a, b in zip(clipper, np.roll(clipper, -1, axis=0)):
        input_poly, output = output, []
        if not input_poly:
            break
        prev = input_poly[-1]
        for cur in input_poly:
            cur_in, prev_in = _cross(a, b, cur) >= 0, _cross(a, b, prev) >= 0
            if cur_in:
                if not prev_in:
                    output.append(_intersection(prev, cur, a, b))
                output.append(cur)
            elif prev_in:
                output.append(_intersection(prev, cur, a, b))
            prev = cur
    return np.asarray(output)


def _area(poly):
    if len(poly) < 3:
        return 0.0
    return abs(np.dot(poly[:, 0], np.roll(poly[:, 1], -1)) -
               np.dot(poly[:, 1], np.roll(poly[:, 0], -1))) * 0.5


def _rotated_iou(a, b):
    pa, pb = _corners(a), _corners(b)
    inter = _area(_clip(pa, pb))
    union = _area(pa) + _area(pb) - inter
    return inter / union if union > 0 else 0.0


def nms_cuda(boxes, scores, thresh, pre_maxsize=None, post_max_size=None):
    """CPU rotated NMS with the same call signature as the CUDA operation."""
    order = scores.argsort(descending=True)
    if pre_maxsize is not None:
        order = order[:pre_maxsize]
    boxes_np = boxes.detach().cpu().numpy()
    keep = []
    for idx in order.detach().cpu().tolist():
        if all(_rotated_iou(boxes_np[idx], boxes_np[j]) <= thresh for j in keep):
            keep.append(idx)
    if post_max_size is not None:
        keep = keep[:post_max_size]
    return torch.tensor(keep, dtype=torch.long, device=boxes.device)
