import numpy as np
import open3d as o3d

from loader.load_preprocessed import load_preprocessed
from graphics.write_mesh import write_volume_mesh

shapenet_dir = 'data/chamferData/01'
dataset = load_preprocessed(shapenet_dir, 5)

def to_xzy(m):
    t = str(type(m))
    m_xzy = m.copy() if 'numpy.ndarray' in t else m.clone()
    m_xzy[..., 1] = m[..., 2]
    m_xzy[..., 2] = m[..., 1]
    return m_xzy

def write_shape(shape, name):
    write_volume_mesh(shape, name)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(to_xzy(shape.shape_points))
    o3d.io.write_point_cloud(f"results/{name}_shape_points.ply", pcd)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(to_xzy(shape.closest_points.reshape(-1, 3)))
    o3d.io.write_point_cloud(f"results/{name}_closest_points.ply", pcd)

def plot():
    for i, shape in enumerate(dataset):
        write_shape(shape, i + 1)

def index_to_coordinate(i, n):
    return -.5 + (i + .5) / n

def test_closest_points():
    n = 64
    for shape in dataset:
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    x = index_to_coordinate(i, n)
                    y = index_to_coordinate(j, n)
                    z = index_to_coordinate(k, n)
                    u = np.asarray([x, y, z])

                    d1 = ((shape.closest_points - u) ** 2).sum(-1).min() # (shape.closest_points - u).pow(2).sum(-1).min()
                    d2 = ((shape.closest_points[i, j, k] - u) ** 2).sum(-1) # (shape.closest_points[i, j, k] - u).pow(2).sum(-1)
                    assert np.allclose(d1, d2), 'closest_points incorrect'

plot()
