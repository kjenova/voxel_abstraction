import numpy as np
import open3d as o3d
from load_shapenet import load_shapenet, ShapeNetShape
from write_mesh import write_volume_mesh, write_predictions_mesh
from load_shapes import closest_points_grid

shapenet_dir = 'shapenet/chamferData/03001627'
n_examples = 5
dataset = load_shapenet(shapenet_dir)[:n_examples]

def plot():
    for i, shape in enumerate(dataset):
        write_volume_mesh(shape, i + 1)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(shape.shape_points)
        o3d.io.write_point_cloud(f"results/{i + 1}_shape_points.ply", pcd)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(shape.closest_points.reshape(-1, 3))
        o3d.io.write_point_cloud(f"results/{i + 1}_closest_points.ply", pcd)

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

                    d1 = (shape.closest_points - u).pow(2).sum(-1)
                    d2 = (shape.closest_points[i, j, k] - u).pow(2).sum(-1)
                    assert np.allclose(d1, d2), 'closest_points incorrect'

test_closest_points()
