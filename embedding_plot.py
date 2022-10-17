import numpy as np
from trimesh import Trimesh
from PIL import Image
import pyvista as pv
from sklearn.manifold import TSNE
from load_shapenet import load_shapenet, ShapeNetShape
from bruteforce_view import bruteforce_view

max_n_examples = 500
shape_parameters = np.load('shape_parameters.npy')[:max_n_examples]
n = shape_parameters.shape[0]

shapenet_dir = 'shapenet/chamferData/00'
dataset = load_shapenet(shapenet_dir, n)

n_angles = 8 # Število vrednosti elevation in azimuth kota kamere
shape_image_size = 256
plot_image_size = 16384

images = []
p = pv.Plotter(off_screen = True, window_size = [shape_image_size] * 2)

for i, shape in enumerate(dataset):
    vertices, faces = shape.resized_volume_faces.get_mesh()
    mesh = pv.wrap(Trimesh(vertices, faces))

    mesh_actor = p.add_mesh(mesh)
    best_image = bruteforce_view(p, n_angles, transparent = True)
    p.remove_actor(mesh_actor)

    images.append(best_image)

embedding = TSNE(n_components = 2).fit_transform(shape_parameters)
embedding = embedding - embedding.min(0)
embedding = embedding / embedding.max(0)
s = plot_image_size - shape_image_size
embedding *= s
embedding = np.floor(embedding).clip(0, s - 1).astype(int)

plot = Image.new('RGBA', (plot_image_size,) * 2, (0, 0, 0, 0))

for i in range(n):
    plot.paste(images[i], box = (embedding[i, 0], embedding[i, 1]), mask = images[i])

plot.save('plot.png')
