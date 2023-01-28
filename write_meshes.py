from trimesh import Trimesh
import open3d as o3d

from loader.load_urocell import load_urocell_preprocessed
from loader.load_shapes import VolumeFaces

validation, test = load_urocell_preprocessed('data/urocell')
dataset = validation + test

def write_helper(vertices, faces, filename):
    # xyz => xzy
    vertices_xzy = vertices.copy()
    vertices_xzy[..., 1] = vertices[..., 2]
    vertices_xzy[..., 2] = vertices[..., 1]
    triangles = Trimesh(vertices_xzy, faces)
    triangles.export(filename)

def to_xzy(m):
    t = str(type(m))
    m_xzy = m.copy() if 'numpy.ndarray' in t else m.clone()
    m_xzy[..., 1] = m[..., 2]
    m_xzy[..., 2] = m[..., 1]
    return m_xzy

for i, x in enumerate(dataset[:5]):
    write_helper(x.vertices, x.faces - 1, f'models/{i + 1}_full.stl')
    v = VolumeFaces(x.resized_volume)
    write_helper(*v.get_mesh(), f'models/{i + 1}_resized.stl')

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(to_xzy(x.shape_points))
    o3d.io.write_point_cloud(f"models/{i + 1}_shape_points.ply", pcd)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(to_xzy(x.closest_points.reshape(-1, 3)))
    o3d.io.write_point_cloud(f"models/{i + 1}_closest_points.ply", pcd)
