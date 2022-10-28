import numpy as np
from trimesh import Trimesh
from PIL import Image
import pyvista as pv
from sklearn.manifold import TSNE

from loader.load_urocell import load_urocell_preprocessed
from render_utils.bruteforce_view import bruteforce_view

n_angles = 8 # Število vrednosti elevation in azimuth kota kamere
shape_image_size = 256
plot_image_size = 16384

validation, test = load_urocell_preprocessed(params.urocell_dir)
dataset = validation + test
n = len(dataset)

shape_parameters = np.zeros((n, params.n_primitives, 10))
result_batches = inference(dataset)

k = 0
for batch in result_batches:
    m = batch.dims.size(0)

    shape_parameters[k : k + m, :, :3] = batch.dims.cpu().numpy()
    shape_parameters[k : k + m, :, 3:7] = batch.quat.cpu().numpy()
    shape_parameters[k : k + m, :, 7:] = batch.trans.cpu().numpy()

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
