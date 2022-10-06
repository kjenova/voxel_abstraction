import numpy as np
from trimesh import Trimesh
from PIL import Image
import pyvista as pv
from load_shapenet import load_shapenet, ShapeNetShape

shapenet_dir = 'shapenet/chamferData/00'
max_n_examples = 5
dataset = load_shapenet(shapenet_dir, max_n_examples)

n_angles = 8 # Število vrednosti elevation in azimuth kota kamere
camera_radius = 4.
max_shape_radius = np.sqrt(3) / 2 + 1e-4
clipping_range = (camera_radius - max_shape_radius, camera_radius + 2 * max_shape_radius)

shape_image_size = 256
plot_image_size = 4096

images = []

for i, shape in enumerate(dataset):
    vertices, faces = shape.resized_volume_faces.get_mesh()
    mesh = pv.wrap(Trimesh(vertices, faces))

    # To je slika i-te oblike, ki je tako zarotirana,
    # da je slika čimbolj zapolnjena.
    best_image = None
    min_n_empty_pixels = float('inf')

    for e in range(n_angles):
        for a in range(n_angles):
            elevation = e * np.pi / n_angles # [0, pi)
            azimuth = a * 2 * np.pi / n_angles # [0, 2pi)

            camera = pv.Camera()
            camera.roll = 0
            camera.elevation = 0
            camera.azimuth = 0
            camera.clipping_range = clipping_range

            position = camera_radius * to_cartesian(elevation, azimuth)
            camera.position = position
            camera.focal_point = - position
            camera.up = up_vector_on_sphere(position)

            p = pv.Plotter(off_screen = True, window_size = [1024, 1024])
            p.add_mesh(mesh, color = True)
            p.store_image = True
            p.camera = camera

            image = p.screenshot(transparent_background = True)
            image = Image.fromarray(image, 'RGBA')
            depth = p.get_image_depth()
            n_empty_pixels = np.count_nonzero(~np.isnan(depth))

            if n_empty_pixels < min_n_empty_pixels:
                min_n_empty_pixels = n_empty_pixels
                best_image = image

    image.save(f'render/best_{i + 1}.png')
    images.append(image)

embedding = np.random.rand(n, 2)
embedding = embedding - embedding.min(0)
embedding = embedding / embedding.max(0)
