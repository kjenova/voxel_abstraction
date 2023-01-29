import nibabel
import numpy as np
from scipy.ndimage import binary_erosion
from skimage import measure
from trimesh import Trimesh
from PIL import Image
import pyvista as pv

from graphics.write_mesh import cuboid_faces
from graphics.bruteforce_view import rotate_scene, bruteforce_view

from loader.load_urocell import load_urocell_preprocessed

shape_image_size = 512
n_angles = 8 # Število vrednosti elevation in azimuth kota kamere

_, test = load_urocell_preprocessed(params.urocell_dir)

p = pv.Plotter(off_screen = True, window_size = [shape_image_size] * 2)

def test_set_plot():
    volume_meshes = [pv.wrap(Trimesh(test[i].vertices, test[i].faces - 1)) for i in range(10)]
    test_set_plot = Image.new('RGB', plot_image_size, (0, 0, 0))
    best_angles = []
    for i, volume_mesh in enumerate(volume_meshes):
        volume_actor = p.add_mesh(volume_mesh)
        image, best_angle = bruteforce_view(p, n_angles)
        p.remove_actor(volume_actor)

        best_angles.append(best_angle)

        k = i % 5
        j = i // 5
        test_set_plot.paste(image, box = (k * shape_image_size, j * shape_image_size))

    test_set_plot.save('analysis/test_set_plot.png')

def normals_plot():
    x = test[5]
    mesh = pv.wrap(Trimesh(x.vertices, x.faces - 1))

    volume_actor = p.add_mesh(mesh)
    _, best_angles = bruteforce_view(p, n_angles)
    arrows_actor = p.add_arrows(x.shape_points, x.normals, mag = 0.2, show_scalar_bar = False)

    image = rotate_scene(p, *best_angles)
    image.save('analysis/normals.png')

    p.remove_actor(volume_actor)
    p.remove_actor(arrows_actor)

test_set_plot()
normals_plot()
