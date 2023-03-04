import numpy as np
from trimesh import Trimesh
from PIL import Image
import pyvista as pv
from sklearn.manifold import TSNE
from tqdm import tqdm
import matplotlib.pyplot as plt

from loader.load_urocell import load_urocell_preprocessed
from graphics.bruteforce_view import rotate_scene

# Koliko kotov med tremi naključnimi točkami na povšrju oblike vzorčimo:
n_angles = 10000

shape_image_size = 256
plot_image_size = 16384

validation, test = load_urocell_preprocessed('data/urocell')
# validation, test = load_urocell_preprocessed('data/urocell_contacting', contacting = True)
dataset = validation + test

p = pv.Plotter(off_screen = True, window_size = [shape_image_size] * 2)
rng = np.random.default_rng(seed = 0x5EED)

angles = np.zeros((len(dataset), n_components * (n_components - 1)))

for i, shape in enumerate(tqdm(dataset)):
    """
    m = Trimesh(shape.vertices, shape.faces - 1)
    mesh = pv.wrap(m)
    mesh_actor = p.add_mesh(mesh, color = 'red' if shape.branched else None)
    shape.image = rotate_scene(p, 0, 0, transparent = True)
    p.remove_actor(mesh_actor)
    """

    j = np.choice(shape.shape_points.shape[0], (n_angles, 3))
    j = j[(j[:, 0] != j[:, 1]) & (j[:, 1] != j[:, 2]) & (j[:, 0] != j[:, 2])]
    x = shape.shape_points[j]

    a = x[:, 0] - x[:, 1]
    b = x[:, 2] - x[:, 1]
    cos = (a * b).sum(-1) / np.sqrt((a * a).sum(-1) * (b * b).sum(-1))

    plt.hist(cos, bins = 'auto')
    plt.savefig(f'angles/hist_{i}.png')
    plt.clf()

"""
embedding = TSNE(n_components = 2).fit_transform(features)
embedding = embedding - embedding.min(0)
embedding = embedding / embedding.max(0)
s = plot_image_size - shape_image_size
embedding *= s
embedding = np.floor(embedding).clip(0, s - 1).astype(int)

plot = Image.new('RGBA', (plot_image_size,) * 2, (0, 0, 0, 0))

for i, shape in enumerate(dataset):
    plot.paste(shape.image, box = (embedding[i, 0], embedding[i, 1]), mask = shape.image)

image = Image.new('RGBA', (plot_image_size,) * 2, (76, 76, 76, 255))
image.paste(plot, mask = plot)
image = image.convert('RGB')

image.save('angles_embedding.png')
image.resize((2048, 2048)).save('angles_embedding_resized.png')
"""
