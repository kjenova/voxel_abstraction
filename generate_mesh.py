import torch
import numpy as np
from transform import primitive_to_world_space

def predictions_to_mesh(shape, quat, trans):
    [b, p] = trans.size()[:2]
    vertices = torch.zeros(b, p, 8, 3, device = shape.device)

    shape = shape.view(b, p, 1, 3).repeat(1, 1, 4, 1)

    vertices[..., [0, 1, 4, 5], 0] = shape[..., 0]
    vertices[..., [2, 3, 6, 7], 0] = - shape[..., 0]

    vertices[..., ::2, 1] = shape[..., 1]
    vertices[..., 1::2, 1] = - shape[..., 1]

    vertices[..., :4, 2] = shape[..., 2]
    vertices[..., 4:, 2] = - shape[..., 2]

    return primitive_to_world_space(vertices, quat, trans)
