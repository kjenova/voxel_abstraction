import torch
import numpy as np
from trimesh import Trimesh

from yang.inference import inference

from loader.load_urocell import load_urocell_preprocessed

from common.transform import predictions_to_mesh_vertices
from graphics.write_mesh import cuboid_faces

import pyvista as pv
from graphics.bruteforce_view import rotate_scene, bruteforce_view

prob_threshold = .5
# Kvadre podaljšamo vzdolž največje dimenzije, da dodamo "manevrski prostor"
# za kolizije med bližnjimi predmeti.
elongation_factor = 1.2

def prediction_vertices_to_mesh(vertices):
    p = vertices.shape[0]
    v = vertices.reshape(-1, 3)
    f = cuboid_faces.reshape(1, 6, 4).repeat(p, axis = 0)
    f += 8 * np.arange(p).reshape(p, 1, 1)
    f = f.reshape(-1, 4)

    return Trimesh(vertices, f)

_, test = load_urocell_preprocessed(params.urocell_dir)

p = pv.Plotter(off_screen = True, window_size = [512] * 2)

result_batches = inference(test)
j = 0
for batch in result_batches:
    m = batch.dims.argmax(-1)
    batch.dims[m] *= elongation_factor

    vertices = predictions_to_mesh_vertices(batch).cpu()

    for i in range(vertices.shape[0]):
        v = predictions_vertices[i, batch.prob[i].cpu() > prob_threshold].numpy()
        mesh = prediction_vertices_to_mesh(v)

        actor = p.add_mesh(pv.wrap(mesh))
        image, _ = bruteforce_view(p, 8)
        p.remove_actor(actor)

        image.save(f'{j}.png')
        j += 1
