import torch
import torch.nn.functional as F

from .transform import world_to_primitive_space, primitive_to_world_space
from .cuboid import CuboidSurface

def points_to_primitives_distance_squared(P, shape_points):
    [b, p] = P.dims.size()[:2]
    n = shape_points.size(1)
    points = shape_points.unsqueeze(1).repeat(1, p, 1, 1)
    point = world_to_primitive_space(points, P.quat, P.trans)

    dims = P.dims.unsqueeze(2).repeat(1, 1, n, 1)
    return F.relu(points.abs() - dims).pow(2).sum(-1)

def _coverage(P, shape_points):
    distance = points_to_primitives_distance_squared(P, shape_points)
    distance += 10 * (1 - P.exist.unsqueeze(-1))
    distance, _ = distance.min(1)
    return distance

def coverage(P, shape_points):
    return _coverage(P, shape_points).mean(1)

def point_indices(points, volume):
    [b, grid_size] = volume.size()[:2]
    min_center = -.5 + .5 / grid_size
    i = (points - min_center) * grid_size
    i = i.clamp(0, grid_size - 1).round().long()
    u = grid_size ** 3 * torch.arange(0, b, device = volume.device)
    v = grid_size ** 2 * i[..., 0]
    w = grid_size * i[..., 1]
    return u.reshape(-1, 1, 1) + w + v + i[..., 2]

def _consistency(volume, P, closest_points_grid, n_samples_per_primitive, expected_value = False):
    sampler = CuboidSurface(n_samples_per_primitive)

    primitive_points = sampler.sample_points(P.dims)
    primitive_points = primitive_to_world_space(primitive_points, P.quat, P.trans)

    weights = sampler.get_importance_weights(P.dims)

    if not expected_value:
        weights *= P.exist.unsqueeze(-1)

    weights /= weights.sum((1, 2), keepdim = True) + 1e-7

    if expected_value:
        weight *= P.prob.unsqueeze(-1)

    i = point_indices(primitive_points, volume)
    closest_points = closest_points_grid.reshape(-1, 3)[i]
    distance = (closest_points - primitive_points).pow(2).sum(-1)

    # Ko je točka znotraj polnega voksla, naj bo razdalja nič:
    distance *= 1 - volume.take(i)

    return distance, weights

def consistency(volume, P, closest_points_grid, n_samples_per_primitive, expected_value = False):
    distance, weights = _consistency(volume, P, closest_points_grid, n_samples_per_primitive, expected_value)
    return (distance * weights).sum((1, 2))

def reconstruction_loss(volume, primitives, shape_points, closest_points_grid, n_samples_per_primitive):
    cov = coverage(primitives, shape_points)
    cons = consistency(volume, primitives, closest_points_grid, n_samples_per_primitive)
    return cov, cons

def paschalidou_reconstruction_loss(volume, P, shape_points, closest_points_grid, params):
    distance = points_to_primitives_distance_squared(P, shape_points)
    distance = distance.transpose(1, 2)

    sorted_distance, indices = distance.sort()
    [b, n, p] = indices.size()
    indices = indices + p * torch.arange(0, b, device = indices.device).reshape(b, 1, 1)
    sorted_prob = P.prob.take(indices)

    # Verjetnost, da bližji primitivi niso prisotni.
    neg_cumprod = (1 - sorted_prob).cumprod(-1)
    neg_cumprod = torch.cat(
        (neg_cumprod.new_ones(b, n, 1), neg_cumprod[..., :-1]),
        dim = -1
    )

    # Verjetnost, da je k-ti primitiv najbližji.
    minprob = sorted_prob * neg_cumprod

    cov = (sorted_distance * minprob).mean((1, 2))

    cons = consistency(volume, P, closest_points_grid, params.n_samples_per_primitive, True)

    return cov, cons

def paschalidou_parsimony_loss(P, params):
    prob_sum = P.prob.sum(-1)
    return F.relu(params.paschalidou_alpha * (1 - prob_sum)) + params.paschalidou_beta * prob_sum.sqrt()

if __name__ == "__main__":
    from load_shapes import voxel_center_points

    def test_point_indices():
        p = 5
        volume = torch.zeros(2, 3, 3, 3)
        points = voxel_center_points([3, 3, 3]).reshape(1, 1, 27, 3).repeat(2, p, 1, 1)
        indices = point_indices(points, volume)

        target_indices = torch.arange(0, 2 * 3 * 3 * 3).reshape(2, -1, 1).repeat(1, p, 1)
        assert torch.allclose(indices.reshape(-1), target_indices.reshape(-1)), 'point_indices is incorrect'

    from cuboid import CuboidSurface

    def test_coverage():
        p = 5
        dims = torch.rand(1, p, 3) * 0.5
        quat = torch.empty(1, p, 4).uniform_(-1, 1)
        quat = F.normalize(quat, dim = -1)
        trans = torch.empty(1, p, 3).uniform_(-.5, .5)
        exist = torch.ones(1, p)

        n = 150
        sampler = CuboidSurface(n)
        points = sampler.sample_points(dims)
        points = primitive_to_world_space(points, quat, trans)

        points = world_to_primitive_space(points, quat, trans)
        dims = dims.unsqueeze(2).repeat(1, 1, n, 1)
        distance = F.relu(points.abs() - dims).pow(2).sum(-1)
        distance += 10 * (1 - exist.unsqueeze(-1))
        assert distance.mean() < 1e-4, 'coverage is incorrect'

    from primitives import Primitives

    def test_consistency():
        grid_size = 5
        volume = torch.zeros([1] + [grid_size] * 3)
        closest_points_grid = torch.rand([1] + [grid_size] * 3 + [3])
        dims = torch.zeros(1, 1, 3)
        quat = torch.Tensor([1, 0, 0, 0]).reshape(1, 1, 4)
        exist = torch.ones(1, 1)

        for i in range(grid_size):
            for j in range(grid_size):
                for k in range(grid_size):
                    trans = torch.Tensor([i, j, k]).reshape(1, 1, 3)
                    trans = (trans + .5) / grid_size - .5
                    P = Primitives(dims, quat, trans, exist)

                    cons, _ = _consistency(volume, P, CuboidSurface(150), closest_points_grid)
                    cons = cons.mean()
                    distance = (trans.reshape(3) - closest_points_grid[0, i, j, k]).pow(2).sum()
                    assert torch.allclose(cons, distance), 'consistency is incorrect'

    test_point_indices()
    test_coverage()
    test_consistency()
