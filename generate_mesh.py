import torch
import numpy as np
from transform import primitive_to_world_space

def voxels_to_mesh(voxels):
    mx = voxels.shape[0]
    my = voxels.shape[1]
    mz = voxels.shape[2]

    max_n_quads = 6 * mx * my * mz
    voxel_centers = np.zeros((max_n_quads, 3))
    normals = np.zeros((max_n_quads, 3))
    tangents = np.zeros((max_n_quads, 2, 3))
    n_quads = 0
    for x in range(mx):
        for y in range(my):
            for z in range(mz):
                if not voxels[x, y, z]:
                    continue

                for side in range(6):
                    orientation = 1 - 2 * (side % 2)
                    direction = side // 2

                    u = [x, y, z]
                    u[direction] += orientation

                    if u[0] < 0 or u[0] >= mx or \
                       u[1] < 0 or u[1] >= my or \
                       u[2] < 0 or u[2] >= mz or \
                       not voxels[u[0], u[1], u[2]]:
                        voxel_centers[n_quads] = [x, y, z]
                        normals[n_quads, direction] = orientation
                        tangents[n_quads, 0, (direction + 1) % 3] = 0.5 * orientation
                        tangents[n_quads, 1, (direction + 2) % 3] = 0.5 * orientation
                        n_quads += 1

    voxel_centers = voxel_centers[:n_quads]
    normals = normals[:n_quads]
    tangents = tangents[:n_quads]

    voxel_centers += 0.5
    quads_centers = voxel_centers + 0.5 * normals

    quads_points = np.zeros((n_quads, 4, 3))
    quads_points[:, 0] = quads_centers - tangents[:, 0] - tangents[:, 1]
    quads_points[:, 1] = quads_centers + tangents[:, 0] - tangents[:, 1]
    quads_points[:, 2] = quads_centers + tangents[:, 0] + tangents[:, 1]
    quads_points[:, 3] = quads_centers - tangents[:, 0] + tangents[:, 1]

    m = max(voxels.shape)
    quads_points *= 2 / max(voxels.shape)
    quads_points[..., 0] -= voxels.shape[0] / m
    quads_points[..., 1] -= voxels.shape[1] / m
    quads_points[..., 2] -= voxels.shape[2] / m

    vertices = quads_points.reshape(-1, 3)
    faces = np.repeat([[[0, 1, 2], [2, 3, 0]]], n_quads, axis = 0)
    faces += np.arange(0, n_quads * 4, 4).reshape(-1, 1, 1)
    faces = faces.reshape(-1, 3)

    return vertices, faces

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

