import torch
from .quat_utils import quat_rotate, quat_inverse_rotate

def prepare_trans(trans, points):
    return trans.unsqueeze(2).repeat(1, 1, points.size(2), 1)

# points: B x P x N x 3, trans: B x P x 3, quat: B x P x 4
def primitive_to_world_space(points, quat, trans):
    points = quat_rotate(points, quat)
    return points + prepare_trans(trans, points)

def world_to_primitive_space(points, quat, trans):
    points = points - prepare_trans(trans, points)
    return quat_inverse_rotate(points, quat)

def predictions_to_mesh_vertices(P):
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

if __name__ == "__main__":
    import numpy as np
    from quat_utils import get_random_quat

    def test_transform():
        points = torch.empty(10, 3, 4, 3).uniform_(-1, 1)
        quat, _ = get_random_quat(10, 3)
        trans = torch.empty(10, 3, 3).uniform_(-1, 1)
        points_2 = primitive_to_world_space(points, quat, trans)
        points_2 = world_to_primitive_space(points_2, quat, trans)
        assert (points_2 - points).abs().mean() < 1e-4, 'Transform/Inverse transform is incorrect'

        point = torch.empty(1, 1, 1, 3).uniform_(-1, 1)
        quat, q = get_random_quat()
        trans = torch.empty(1, 1, 3).uniform_(-1, 1)

        t1 = primitive_to_world_space(point, quat, trans).squeeze()
        t2 = torch.Tensor(q.rotate(point.squeeze())).float() + trans.squeeze()
        assert (t2 - t1).abs().mean() < 1e-4, 'Transform is incorrect'

        t1 = world_to_primitive_space(point, quat, trans).squeeze()
        t2 = torch.Tensor(q.inverse.rotate((point - trans).squeeze())).float()
        assert (t2 - t1).abs().mean() < 1e-4, 'Inverse transform is incorrect'

    test_transform()
