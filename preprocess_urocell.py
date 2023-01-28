import nibabel
import torch
import open3d as o3d
import os
from tqdm import tqdm
from skimage import measure
from trimesh import Trimesh
from scipy.io import savemat

from loader.load_shapes import resize_volume, VolumeFaces, closest_points_grid, centers_linspace

class UroCellShape:
    def __init__(self, volume, grid_size, n_points_per_shape):
        self.volume = volume
        self.resized_volume = resize_volume(volume, grid_size)
        self.volume_faces = VolumeFaces(volume)
        # self.shape_points = self.volume_faces.sample(n_points_per_shape)
        # self.closest_points_grid = closest_points_grid(self.resized_volume, self.volume_faces)

test_data = [
    ('fib1-1-0-3.nii.gz', [[0], [1, 2]]),
    ('fib1-3-2-1.nii.gz', [[0, 1], [18]]),
    ('fib1-3-3-0.nii.gz', [[0, 3], [0]]),
    ('fib1-4-3-0.nii.gz', [[0], []])
]

def preprocess(basedir, grid_size = 64, n_points_per_shape = 10000):
    shapes = []

    for file, indices_by_label in test_data:
        volume = nibabel.load(f'{basedir}/{file}')
        volume = volume.get_fdata()

        volume_name = file.replace('.nii.gz', '')

        for label, indices in enumerate(indices_by_label):
            v = volume == (label + 1)
            labelled = measure.label(volume)
            props = measure.regionprops(labelled)
            props.sort(key = lambda x: x.area, reverse = True)

            dir = f'data/urocell/{volume_name}/{label + 1}'
            os.makedirs(dir, exist_ok = True)

            for i, p in enumerate(props):
                shapes.append((p.filled_image, f'{dir}/{i + 1}.mat'))

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

preprocess('matlab/branched')
