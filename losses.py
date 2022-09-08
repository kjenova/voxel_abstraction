import torch
import torch.nn.functional as F
from transform import world_to_primitive_space, primitive_to_world_space
from cuboid import CuboidSurface

def coverage(P, sampled_points):
    [b, p] = P.dims.size()[:2]
    n = sampled_points.size(1)
    points = sampled_points.unsqueeze(1).repeat(1, p, 1, 1)
    points = world_to_primitive_space(points, P.quat, P.trans)

    dims = P.dims.unsqueeze(2).repeat(1, 1, n, 1)
    distance = F.relu(points.abs() - dims).pow(2).sum(-1)
    distance += 10 * (1 - P.exist.unsqueeze(-1))
    distance, _ = distance.min(1)
    return distance.mean(1)

def point_indices(points, volume):
    [b, grid_size] = volume.size()[:2]
    min_center = -.5 + .5 / grid_size
    i = (points - min_center) * grid_size
    i = i.clamp(0, grid_size - 1).round().long()
    u = grid_size ** 3 * torch.arange(0, b, device = volume.device)
    v = grid_size ** 2 * i[..., 0]
    w = grid_size * i[..., 1]
    return u.reshape(-1, 1, 1) + w + v + i[..., 2]

def consistency(volume, P, sampler, closest_points_grid):
    primitive_points = sampler.sample_points(P.dims)
    primitive_points = primitive_to_world_space(primitive_points, P.quat, P.trans)

    weights = sampler.get_importance_weights(P.dims)
    weights *= P.exist.unsqueeze(-1)
    weights /= weights.sum(1, keepdim = True) + 1e-7

    i = point_indices(points, volume)
    x, y, z = closest_points_grid.chunk(3, dim = -1)
    closest_points = torch.stack([x.take(i), y.take(i), z.take(i)], dim = -1)
    diff = (closest_points - primitive_points).pow(2).sum(-1)

    # Ko je točka znotraj polnega voksla, naj bo razdalja nič:
    diff = (1 - volume.take(i)) * diff
    return (diff * weights).sum((1, 2))

def loss(volume, primitives, sampled_points, closest_points_grid, sampler):
    cov = coverage(primitives, sampled_points)
    cons = consistency(volume, primitives, sampler, closest_points_grid)
    return cov + cons

if __name__ == "__main__":
    from load_shapes import voxel_center_points

    def test_point_indices():
        p = 5
        volume = torch.zeros(2, 3, 3, 3)
        points = voxel_center_points([3, 3, 3]).reshape(1, 1, 27, 3).repeat(2, p, 1, 1)
        indices = point_indices(points, volume)

        target_indices = torch.arange(0, 2 * 3 * 3 * 3).reshape(2, -1, 1).repeat(1, p, 1)
        assert torch.allclose(indices.reshape(-1), target_indices.reshape(-1)), 'point_indices is incorrect'
        
    test_point_indices()
