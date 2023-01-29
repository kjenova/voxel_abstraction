import torch
import open3d as o3d
from tqdm import tqdm
from trimesh import Trimesh
from scipy.io import savemat

from .load_shapes import resize_volume, VolumeFaces, closest_points_grid, centers_linspace

def preprocess(shapes, grid_size = 64, n_points_per_shape = 10000):
    with torch.no_grad():
        c = centers_linspace(grid_size)
        voxel_centers = torch.cartesian_prod(c, c, c).reshape(-1, 3)
        voxel_centers = o3d.core.Tensor(voxel_centers.numpy())

    for volume, file in tqdm(shapes):
        resized_volume = resize_volume(volume, grid_size)
        volume_faces = VolumeFaces(volume)
        shape_points = volume_faces.sample(n_points_per_shape)
        normals = volume_faces.get_interpolated_normals(shape_points)

        mesh = Trimesh(*volume_faces.get_mesh())

        closest_points = closest_points_grid(resized_volume, mesh.as_open3d, voxel_centers)

        savemat(file, {
            'Volume': resized_volume,
            'surfaceSamples': shape_points,
            'normals': normals,
            'closestPoints': closest_points,
            'vertices': mesh.vertices,
            'faces': mesh.faces + 1
        })
