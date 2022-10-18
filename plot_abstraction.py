import torch
import numpy as np
from trimesh import Trimesh
from PIL import Image
import pyvista as pv
from train import Network, PhaseParams
from load_urocell import load_urocell_preprocessed
from generate_mesh import predictions_to_mesh
from bruteforce_view import bruteforce_view
from write_mesh import cuboid_faces
from colors import colors

model = Network(PhaseParams(1))
model.load_state_dict(torch.load('save.torch'))
model.eval()

grid_size = 32
prob_threshold = .5
remove_redundant = False # TODO

basedir = 'shapenet/chamferData/urocell'
_, test = load_urocell_preprocessed(basedir)

volume_batch = torch.stack([torch.Tensor(shape.resized_volume) for shape in test])
with torch.no_grad():
    P = model(volume_batch, PhaseParams(1))
    predictions_vertices = predictions_to_mesh(P).cpu()

n_angles = 8 # Število vrednosti elevation in azimuth kota kamere
shape_image_size = 512
plot_image_height = 2 * shape_image_size
plot_image_width = 5 * shape_image_size
plot_image_size = [plot_image_width, plot_image_height]

images = []
p = pv.Plotter(off_screen = True, window_size = [shape_image_size] * 2)

def prediction_vertices_to_mesh(vertices):
    p = vertices.shape[0]
    v = vertices.reshape(-1, 3)
    f = cuboid_faces.reshape(1, 6, 4).repeat(p, axis = 0)
    f += 8 * np.arange(p).reshape(p, 1, 1)
    f = f.reshape(-1, 4)
    f = np.concatenate((4 * np.ones((f.shape[0], 1), dtype = int), f), axis = 1)
    c = colors[:p].reshape(p, 1, 3).repeat(6, axis = 1)
    c = c.reshape(-1, 3)

    mesh = pv.PolyData(v, faces = f.reshape(-1), n_faces = 6 * p)
    mesh["colors"] = c
    return mesh

for i in range(10):
    vertices, faces = test[i].volume_faces.get_mesh()
    volume_mesh = pv.wrap(Trimesh(vertices, faces))

    v = predictions_vertices[i, P.prob[i].cpu() > prob_threshold].numpy()
    predictions_mesh = prediction_vertices_to_mesh(v)

    volume_actor = p.add_mesh(volume_mesh, opacity = .5)
    predictions_actor = p.add_mesh(predictions_mesh, scalars = "colors", rgb = True)
    best_image = bruteforce_view(p, n_angles)
    p.remove_actor(volume_actor)
    p.remove_actor(predictions_actor)

    images.append(best_image)

plot = Image.new('RGB', (plot_image_width, plot_image_height), (0, 0, 0))

for i in range(2):
    for j in range(5):
        image = images[i * 5 + j]
        plot.paste(image, box = (j * shape_image_size, i * shape_image_size))

plot.save('abstraction_tulsiani.png')
