import torch
import numpy as np
import random
from trimesh import Trimesh
from trimesh.collision import CollisionManager

from yang.inference import inference

from loader.load_urocell import load_urocell_preprocessed

from common.transform import predictions_to_mesh_vertices
from graphics.write_mesh import cuboid_faces

import pyvista as pv
from graphics.bruteforce_view import rotate_scene, bruteforce_view

import networkx as nx
import matplotlib.pyplot as plt

prob_threshold = .5
# Kvadre podaljšamo vzdolž največje dimenzije, da dodamo "manevrski prostor"
# za kolizije med bližnjimi predmeti.
elongation_factor = 1.

def prediction_vertices_to_mesh(vertices):
    p = vertices.shape[0]
    v = vertices.reshape(-1, 3)
    f = cuboid_faces.reshape(1, 6, 4).repeat(p, axis = 0)
    f += 8 * np.arange(p).reshape(p, 1, 1)

    return Trimesh(vertices.reshape(-1, 3), f.reshape(-1, 4))

# _, test = load_urocell_preprocessed('data/urocell')
validation, test = load_urocell_preprocessed('data/urocell')
dataset = validation + test

branched = [x for x in dataset if x.branched]
unbranched = [x for x in dataset if not x.branched]

p = pv.Plotter(off_screen = True, window_size = [512] * 2)

def process(x, dir):
    result_batches = inference(x)
    j = 0
    for batch in result_batches:
        m = batch.dims.argmax(-1)
        batch.dims[m] *= elongation_factor

        vertices = predictions_to_mesh_vertices(batch).cpu()

        for i in range(vertices.shape[0]):
            v = vertices[i, batch.exist[i] == 1.].numpy()
            mesh = prediction_vertices_to_mesh(v)

            actor = p.add_mesh(pv.wrap(mesh))
            image, _ = bruteforce_view(p, 8)
            p.remove_actor(actor)

            image.save(f'{dir}/mesh_{j}.png')

            manager = CollisionManager()
            for k in range(v.shape[0]):
                primitive_mesh = prediction_vertices_to_mesh(v[k : (k + 1)])
                manager.add_object(str(k), primitive_mesh)

            _, collisions = manager.in_collision_internal(return_names = True)

            graph = nx.Graph(collisions)
            nx.draw(graph)
            plt.savefig(f'{dir}/graph_{j}.png')
            plt.clf()

            j += 1

process(branched, 'collisions/branched')
random.shuffle(unbranched)
process(unbranched[:20], 'collisions/unbranched')
