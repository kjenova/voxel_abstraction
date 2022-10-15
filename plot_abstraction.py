import numpy as np
from trimesh import Trimesh
from PIL import Image
import pyvista as pv
from load_shapes import VolumeFaces
from load_urocell import load_validation_and_test
from bruteforce_view import bruteforce_view

basedir = '/home/klemenjan/UroCell/mito/branched'
_, test = load_validation_and_test(basedir)

n_angles = 8 # Število vrednosti elevation in azimuth kota kamere
shape_image_size = 512
plot_image_height = 2 * shape_image_size
plot_image_width = 5 * shape_image_size
plot_image_size = [plot_image_width, plot_image_height]

images = []
p = pv.Plotter(off_screen = True, window_size = [shape_image_size] * 2)

for component in test:
    volume_faces = VolumeFaces(component)
    vertices, faces = volume_faces.get_mesh()
    mesh = pv.wrap(Trimesh(vertices, faces))

    mesh_actor = p.add_mesh(mesh)
    best_image = bruteforce_view(p, n_angles)
    p.remove_actor(mesh_actor)

    images.append(best_image)

plot = Image.new('RGBA', (plot_image_width, plot_image_height), (0, 0, 0, 0))

for i in range(2):
    for j in range(5):
        image = images[i * 5 + j]
        plot.paste(image, box = (j * shape_image_size, i * shape_image_size))

plot.save('abstraction_tulsiani.png')
