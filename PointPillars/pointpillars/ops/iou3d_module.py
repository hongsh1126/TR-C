import numpy as np
import torch

from .cpu_ops import _area, _clip, _corners, nms_cuda


def boxes_overlap_bev(boxes_a, boxes_b):
    a = boxes_a.detach().cpu().numpy()
    b = boxes_b.detach().cpu().numpy()
    out = np.zeros((len(a), len(b)), dtype=np.float32)
    for i, box_a in enumerate(a):
        pa = _corners(box_a)
        for j, box_b in enumerate(b):
            out[i, j] = _area(_clip(pa, _corners(box_b)))
    return torch.as_tensor(out, device=boxes_a.device)


def boxes_iou_bev(boxes_a, boxes_b):
    overlap = boxes_overlap_bev(boxes_a, boxes_b)
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    return overlap / torch.clamp(area_a[:, None] + area_b[None, :] - overlap, min=1e-7)


def nms_normal_gpu(boxes, scores, thresh):
    return nms_cuda(boxes, scores, thresh)
