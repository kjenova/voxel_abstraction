import torch
import numpy as np
from trimesh import Trimesh
from PIL import Image
import pyvista as pv
from train import load_evaluation_model, ShapeParams
from load_urocell import load_validation_and_test
from generate_mesh import predictions_to_mesh
from write_mesh import prediction_vertices_to_trimesh
from bruteforce_view import bruteforce_view

model = load_evaluation_model()

grid_size = 32
prob_threshold = .5
remove_redundant = False # TODO

basedir = '/home/klemenjan/UroCell/mito/branched'
_, test = load_validation_and_test(basedir, grid_size, discard_validation = True)

volume_batch = torch.stack([shape.volume for shape in test])
P = network(volume_batch, ShapeParams(1))
predictions_vertices = predictions_to_mesh(P).cpu()

n_angles = 8 # Število vrednosti elevation in azimuth kota kamere
shape_image_size = 512
plot_image_height = 2 * shape_image_size
plot_image_width = 5 * shape_image_size
plot_image_size = [plot_image_width, plot_image_height]

images = []
p = pv.Plotter(off_screen = True, window_size = [shape_image_size] * 2)

for i in range(10):
    vertices, faces = test[i].volume_faces.get_mesh()
    volume_mesh = pv.wrap(Trimesh(vertices, faces))

    v = predictions_vertices[i, P.prob[i].cpu() > prob_threshold].numpy()
    predictions_mesh = pv.wrap(prediction_vertices_to_trimesh(v))

    volume_actor = p.add_mesh(volume_mesh)
    predictions_actor = p.add_mesh(predictions_mesh)
    best_image = bruteforce_view(p, n_angles)
    p.remove_actor(volume_actor)
    p.remove_actor(predictions_actor)

    images.append(best_image)

plot = Image.new('RGBA', (plot_image_width, plot_image_height), (0, 0, 0, 0))

for i in range(2):
    for j in range(5):
        image = images[i * 5 + j]
        plot.paste(image, box = (j * shape_image_size, i * shape_image_size))

plot.save('abstraction_tulsiani.png')
