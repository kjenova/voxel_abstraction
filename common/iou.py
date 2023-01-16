import torch
import torch.nn.functional as F

from .reconstruction_loss import _coverage, point_indices

def iou(volume, P, params):
    points = torch.rand(P.dims.size(0), params.iou_n_points, 3) - .5

    primitive_dist = points_to_primitives_distance_squared(P, points)
    inside_primitive = _coverage(P, points) <= 0

    i = point_indices(points, volume)
    inside_volume = volume.reshape(-1)[i] > 0

    intersection = (inside_primitive & inside_volume).sum(-1)
    union = (inside_primitive | inside_volume).sum(-1)

    return intersection / union
