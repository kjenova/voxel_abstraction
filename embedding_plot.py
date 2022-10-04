import torch
import numpy as np
from scipy.spatial.transform import Rotation
from trimesh import Trimesh
from PIL import Image
import pyvista as pv
from load_shapenet import load_shapenet, ShapeNetShape
from quat_utils import quat_rotate

shapenet_dir = 'shapenet/chamferData/00'
max_n_examples = 1
dataset = load_shapenet(shapenet_dir, max_n_examples)

# Število vrednosti kota okoli vsake osi
n_angles = 4
n_rotations = n_angles ** 3
angles = torch.arange(0, n_angles) * (2 * np.pi / n_angles)
euler_angles = torch.cartesian_prod(angles, angles, angles)
quaternions = Rotation.from_euler('xyz', euler_angles).as_quat()
quaternions = torch.Tensor(quaternions).unsqueeze(0)

for i, shape in enumerate(dataset):
    v, faces = shape.resized_volume_faces.get_mesh()

    with torch.no_grad():
        vertices = torch.Tensor(v).reshape(1, 1, -1, 3)
        vertices = vertices.repeat(1, n_rotations, 1, 1)
        vertices = quat_rotate(vertices, quaternions)
        vertices = vertices.squeeze(0).numpy()

    # To je slika i-te oblike, ki je tako zarotirana,
    # da je slika čimbolj zapolnjena.
    best_image = None
    min_n_empty_pixels = float('inf')

    for r in range(n_rotations):
        mesh = pv.wrap(Trimesh(vertices[r], faces))
        p = pv.Plotter(off_screen = True, window_size = [1024, 1024])
        p.add_mesh(mesh, color = True)
        p.store_image = True

        image = p.screenshot()
        depth = p.get_image_depth()
        n_empty_pixels = np.count_nonzero(~np.isnan(depth))

        if n_empty_pixels < min_n_empty_pixels:
            min_n_empty_pixels = n_empty_pixels
            best_image = image

    img = Image.fromarray(image, 'RGB')
    img.save(f'render/{i + 1}.png')

if False:
    # Za zdaj random
    embedding = np.random.rand(n, 2)
    embedding = embedding - embedding.min(0)
    embedding = embedding / embedding.max(0)
