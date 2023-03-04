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

for i, shape in enumerate(tqdm(dataset)):
    j = np.random.choice(shape.shape_points.shape[0], (n_angles, 3))
    j = j[(j[:, 0] != j[:, 1]) & (j[:, 1] != j[:, 2]) & (j[:, 0] != j[:, 2])]
    x = shape.shape_points[j]

    a = x[:, 0] - x[:, 1]
    b = x[:, 2] - x[:, 1]
    cos = (a * b).sum(-1) / np.sqrt((a * a).sum(-1) * (b * b).sum(-1))

    plt.hist(cos, bins = 'auto')
    plt.savefig(f'angles/{"" if shape.branched else "un"}branched/hist_{i}.png')
    plt.clf()
