import torch
import torch.nn.functional as F

from .reconstruction_loss import _coverage, _point_indices

def points_inside_volume(points, volume):
    batch_indices, grid_indices = _point_indices(points, volume)
    i = batch_indices.reshape(-1, 1) + grid_indices
    return volume.reshape(-1)[i] > 0

def iou(volume, P, params):
    points = torch.rand(P.dims.size(0), params.iou_n_points, 3, device = volume.device) - .5

    inside_primitive = _coverage(P, points) <= 0

    inside_volume = points_inside_volume(points, volume)

    intersection = (inside_primitive & inside_volume).sum(-1)
    union = (inside_primitive | inside_volume).sum(-1)

    return intersection / union
