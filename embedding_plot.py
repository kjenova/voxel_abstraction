import torch
import numpy as np
from scipy.spatial.transform import Rotation
from trimesh import Trimesh
import pyvista as pv
from load_shapenet import load_shapenet, ShapeNetShape

shapenet_dir = 'shapenet/chamferData/00'
max_n_examples = 1
dataset = load_shapenet(shapenet_dir, max_n_examples)

# Število vrednosti kota okoli vsake osi
n_angles = 4
angles = torch.arange(0, n_angles) * (2 * np.pi / n_angles)
euler_angles = torch.cartesian_prod(angles, angles, angles)
quaternions = Rotation.from_euler('xyz', euler_angles).as_quat()
quaternions = torch.Tensor(quaternions).unsqueeze(0)

#for i, shape in enumerate(dataset):
vertices, faces = dataset[0].resized_volume_faces.get_mesh()
mesh = pv.wrap(Trimesh(vertices, faces))
p = pv.Plotter()
p.add_mesh(mesh)
p.show()

if False:
    # Za zdaj random
    embedding = np.random.rand(n, 2)
    embedding = embedding - embedding.min(0)
    embedding = embedding / embedding.max(0)
