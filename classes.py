import torch
import numpy as np
from trimesh import Trimesh
from PIL import Image
import pyvista as pv

from loader.load_urocell import load_urocell_preprocessed

from graphics.bruteforce_view import bruteforce_view

from tulsiani.parameters import params

h = 2
w = 3
n = h * w

n_angles = 8 # Število vrednosti elevation in azimuth kota kamere
shape_image_size = 512
plot_image_height = h * shape_image_size
plot_image_width = w * shape_image_size
plot_image_size = (plot_image_width, plot_image_height)

p = pv.Plotter(off_screen = True, window_size = [shape_image_size] * 2)

def plot(dataset, name):
    dataset_plot = Image.new('RGB', plot_image_size, (0, 0, 0))
    for i in range(n):
        volume_mesh = Trimesh(datatest[i].vertices, datatest[i].faces - 1)
        volume_actor = p.add_mesh(volume_mesh)
        image, _ = bruteforce_view(p, n_angles)
        p.remove_actor(volume_actor)

        k = i % w
        j = i // w
        dataset_plot.paste(image, box = (k * shape_image_size, j * shape_image_size))

    dataset_plot.save(f'{name}.png')

def load(dir)
    validation, test = load_urocell_preprocessed(dir)
    return validation + test

combined = load('data/urocell')
branched = [x for x in combined if x.branched]
unbranched = [x for x in combined if not x.branched]

plot(unbranched, 'normal')
plot(branched, 'branched')

combined = load('data/urocell_contacting')
plot([x for x in combined if x.branched], 'contacting')
