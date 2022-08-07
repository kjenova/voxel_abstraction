import torch
from quat_utils import quat_rotate, quat_inverse_rotate

def prepare_trans(trans, points):
    [b, p] = trans.size()[:2]
    n = points.size(2)
    return trans.unsqueeze(2).repeat(1, 1, n, 1)

# points: B x P x N x 3, trans: B x P x 3, quat: B x P x 4
def primitive_to_world_space(points, quat, trans):
    points = quat_rotate(points, quat)
    return points + prepare_trans(trans, points)

def world_to_primitive_space(points, quat, trans):
    points = points - prepare_trans(trans, points)
    return quat_inverse_rotate(points, quat)
