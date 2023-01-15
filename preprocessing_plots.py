import nibabel
import numpy as np
from scipy.io import loadmat
from scipy.ndimage import binary_erosion
from skimage import measure
from trimesh import Trimesh
from PIL import Image
import pyvista as pv

from graphics.write_mesh import cuboid_faces
from graphics.bruteforce_view import rotate_scene, bruteforce_view

shape_image_size = 512
n_angles = 8 # Število vrednosti elevation in azimuth kota kamere

p = pv.Plotter(off_screen = True, window_size = [shape_image_size] * 2)

def erosion_plot():
    for u in range(1, 11):
        plot = Image.new('RGB', (3 * shape_image_size, 2 * shape_image_size), (0, 0, 0))
        best_angles = None

        for i in range(1, 7):
            mat = loadmat(f'analysis/erosion/{u}_{i}.mat')
            mesh = pv.wrap(Trimesh(mat['vertices'], mat['faces'] - 1))
            volume_actor = p.add_mesh(mesh)

            if i == 1:
                image, best_angles = bruteforce_view(p, n_angles)
            else:
                image = rotate_scene(p, *best_angles)

            p.remove_actor(volume_actor)

            k = (i - 1) % 3
            j = (i - 1) // 3

            plot.paste(image, box = (k * shape_image_size, j * shape_image_size))

        plot.save(f'analysis/erosion_{u}.png')

def resizing_plot():
    plot = Image.new('RGB', (2 * shape_image_size, shape_image_size), (0, 0, 0))

    best_angles = None

    for i, grid_size in enumerate([32, 64]):
        mat = loadmat(f'analysis/resizing/{grid_size}.mat')
        mesh = pv.wrap(Trimesh(mat['vertices'], mat['faces'] - 1))
        volume_actor = p.add_mesh(mesh)

        if i == 0:
            image, best_angles = bruteforce_view(p, n_angles)
        else:
            image = rotate_scene(p, *best_angles)

        plot.paste(image, box = (i * shape_image_size, 0))

        p.remove_actor(volume_actor)

    plot.save('analysis/resizing.png')

def normals_plot():
    mat = loadmat('analysis/normals/1.mat')
    mesh = pv.wrap(Trimesh(mat['vertices'], mat['faces'] - 1))

    volume_actor = p.add_mesh(mesh)
    _, best_angles = bruteforce_view(p, n_angles)
    arrows_actor = p.add_arrows(mat['points'], mat['normals'], mag = 0.2, show_scalar_bar = False)

    image = rotate_scene(p, *best_angles)
    image.save('analysis/normals.png')

    p.remove_actor(volume_actor)
    p.remove_actor(arrows_actor)

erosion_plot()
resizing_plot()
normals_plot()

