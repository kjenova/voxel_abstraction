import numpy as np
from PIL import Image
import pyvista as pv

def to_cartesian(elevation, azimuth):
    return np.asarray([
            np.sin(azimuth) * np.cos(elevation),
            np.sin(azimuth) * np.sin(elevation),
            np.cos(azimuth)
        ])

def up_vector_on_sphere(v):
    i = np.abs(v).argmax()
    j = (i + 1) % 3
    k = (i + 2) % 3
    up = np.ones(3)
    up[i] = (v[j] + v[k]) / - v[i]
    return up / np.linalg.norm(up)

camera_radius = 4.
max_shape_radius = np.sqrt(3) / 2 + 1e-4
clipping_range = (camera_radius - max_shape_radius, camera_radius + 2 * max_shape_radius)

# To je slika oblike 'mesh', ki je tako zarotirana,
# da je slika čimbolj zapolnjena.
def bruteforce_view(p, mesh, n_angles):
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

            mesh_actor = p.add_mesh(mesh, color = True)
            p.store_image = True
            p.camera = camera

            image = p.screenshot(transparent_background = True)
            image = Image.fromarray(image, 'RGBA')
            depth = p.get_image_depth()

            p.remove_actor(mesh_actor)

            n_empty_pixels = np.count_nonzero(~np.isnan(depth))
            if n_empty_pixels < min_n_empty_pixels:
                min_n_empty_pixels = n_empty_pixels
                best_image = image

    return best_image
