import numpy as np
import open3d as o3d
from load_shapenet import load_shapenet, ShapeNetShape
from write_mesh import write_volume_mesh, write_predictions_mesh

shapenet_dir = 'shapenet/chamferData/03001627'
n_examples = 5
dataset = load_shapenet(shapenet_dir)[:n_examples]

for i, shape in enumerate(dataset):
    write_volume_mesh(shape, i + 1)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(shape.shape_points)
    o3d.io.write_point_cloud(f"results/{i + 1}.ply", pcd)
