import torch
import numpy as np
from transform import primitive_to_world_space

def predictions_to_mesh(P):
    [b, p] = P.dims.size()[:2]
    vertices = torch.zeros(b, p, 8, 3, device = P.dims.device)

    dims = P.dims.view(b, p, 1, 3).repeat(1, 1, 4, 1)

    vertices[..., [0, 1, 4, 5], 0] = dims[..., 0]
    vertices[..., [2, 3, 6, 7], 0] = - dims[..., 0]

    vertices[..., ::2, 1] = dims[..., 1]
    vertices[..., 1::2, 1] = - dims[..., 1]

    vertices[..., :4, 2] = dims[..., 2]
    vertices[..., 4:, 2] = - dims[..., 2]

    return primitive_to_world_space(vertices, P.quat, P.trans)
