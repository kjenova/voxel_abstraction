import numpy as np
import open3d as o3d
from load_shapenet import load_shapenet, ShapeNetShape
from write_mesh import write_volume_mesh
from load_shapes import closest_points_grid
from train import BatchProvider
from load_shapes import VolumeFaces

shapenet_dir = 'shapenet/chamferData/03001627'
n_examples = 40
dataset = load_shapenet(shapenet_dir)[:n_examples]

def write_shape(shape, name):
    write_volume_mesh(shape, name)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(shape.shape_points)
    o3d.io.write_point_cloud(f"results/{name}_shape_points.ply", pcd)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(shape.closest_points.reshape(-1, 3))
    o3d.io.write_point_cloud(f"results/{name}_closest_points.ply", pcd)

def plot():
    for i, shape in enumerate(dataset):
        write_shape(shape, i + 1)


def index_to_coordinate(i, n):
    return -.5 + (i + .5) / n

def test_closest_points():
    n = 32
    for shape in dataset:
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    x = index_to_coordinate(i, n)
                    y = index_to_coordinate(j, n)
                    z = index_to_coordinate(k, n)
                    u = np.asarray([x, y, z])

                    d1 = (shape.closest_points - u).pow(2).sum(-1).min()
                    d2 = (shape.closest_points[i, j, k] - u).pow(2).sum(-1)
                    assert np.allclose(d1, d2), 'closest_points incorrect'

class Shape:
    def __init__(self, volume, sampled_points, closest_points):
        self.volume = volume
        self.shape_points = sampled_points
        self.closest_points = closest_points
        self.volume_faces = None
        self.resized_volume_faces = VolumeFaces(volume)

def test_batches():

    batch_provider = BatchProvider(dataset)
    (volume, sampled_points, closest_points) = batch_provider.get()
    volume = volume.cpu().numpy()
    sampled_points = sampled_points.cpu().numpy()
    closest_points = closest_points.cpu().numpy()

    for i in range(volume.shape[0]):
        shape = Shape(volume[i], sampled_points[i], closest_points[i])
        write_shape(shape, f'b{i + 1}')

test_batches()
