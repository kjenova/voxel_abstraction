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
from loader.load_shapes import resize_volume, VolumeFaces

from preprocess_train_set import get_components

shape_image_size = 512
n_angles = 8 # Število vrednosti elevation in azimuth kota kamere

_, test = load_urocell_preprocessed('data/urocell')

p = pv.Plotter(off_screen = True, window_size = [shape_image_size] * 2)

def test_set_plot():
    plot_image_height = 2 * shape_image_size
    plot_image_width = 5 * shape_image_size
    plot_image_size = (plot_image_width, plot_image_height)

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

def erosion_plot():
    components_by_radius = [get_components(r) for r in range(1, 7)]

    for c in range(10):
        plot = Image.new('RGB', (3 * shape_image_size, 2 * shape_image_size), (0, 0, 0))
        best_angles = None

        for r in range(6):
            component = components_by_radius[r][c]
            mesh = pv.wrap(Trimesh(*VolumeFaces(component).get_mesh()))

            volume_actor = p.add_mesh(mesh)

            if r == 0:
                image, best_angles = bruteforce_view(p, n_angles)
            else:
                image = rotate_scene(p, *best_angles)

            p.remove_actor(volume_actor)

            k = r % 3
            j = r // 3

            plot.paste(image, box = (k * shape_image_size, j * shape_image_size))

        plot.save(f'analysis/erosion_{c + 1}.png')

def resizing_plot():
    components = get_components()

    original = components[1]
    resized_64 = resize_volume(original, 64)
    resized_32 = resize_volume(original, 32)

    plot = Image.new('RGB', (3 * shape_image_size, shape_image_size), (0, 0, 0))

    best_angles = None

    for i, volume in enumerate([original, resized_64, resized_32]):
        mesh = pv.wrap(Trimesh(*VolumeFaces(volume).get_mesh()))
        volume_actor = p.add_mesh(mesh)

        if i == 0:
            image, best_angles = bruteforce_view(p, n_angles)
        else:
            image = rotate_scene(p, *best_angles)

        plot.paste(image, box = (i * shape_image_size, 0))

        p.remove_actor(volume_actor)

    plot.save('analysis/resizing.png')

def normals_plot():
    x = test[5]
    mesh = pv.wrap(Trimesh(x.vertices, x.faces - 1))

    volume_actor = p.add_mesh(mesh)
    _, best_angles = bruteforce_view(p, n_angles)
    n = 100
    arrows_actor = p.add_arrows(x.shape_points[:n, :], x.normals[:n, :], mag = 0.2, show_scalar_bar = False)

    image = rotate_scene(p, *best_angles)
    image.save('analysis/normals.png')

    p.remove_actor(volume_actor)
    p.remove_actor(arrows_actor)

test_set_plot()
erosion_plot()
resizing_plot()
normals_plot()
