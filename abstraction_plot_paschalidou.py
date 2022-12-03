import open3d as o3d
import numpy as np
from trimesh import Trimesh
from PIL import Image
import pyvista as pv

from paschalidou.inference import inference as paschalidou_inference

from loader.load_urocell import load_urocell_preprocessed

from graphics.colors import colors

_, test = load_urocell_preprocessed("data/chamferData/urocell")

X = paschalidou_inference(test)

for i, primitives in enumerate(X):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.vstack(primitives))
    o3d.io.write_point_cloud("results/paschalidou/" + str(i + 1) + ".ply", pcd)

n_angles = 8 # Število vrednosti elevation in azimuth kota kamere
shape_image_size = 512
plot_image_height = 2 * shape_image_size
plot_image_width = 5 * shape_image_size
plot_image_size = [plot_image_width, plot_image_height]

p = pv.Plotter(off_screen = True, window_size = [shape_image_size] * 2)

volume_meshes = [pv.wrap(Trimesh(test[i].vertices, test[i].faces - 1)) for i in range(10)]
best_angles = []
for i, volume_mesh in enumerate(volume_meshes):
    volume_actor = p.add_mesh(volume_mesh)
    image, best_angle = bruteforce_view(p, n_angles)
    p.remove_actor(volume_actor)

    best_angles.append(best_angle)

plot = Image.new('RGBA' if transparent else 'RGB', (plot_image_width, plot_image_height), (0, 0, 0))

for i, volume_mesh in enumerate(volume_meshes):
    p = len(X[i])
    if p <= 0:
        continue

    n = X[i][0].shape[0]
    c = colors[:p].reshape(p, 1, 3).repeat(n, axis = 1)
    predictions_mesh = pv.PolyData(np.vstack(X[i]))
    predictions_mesh["colors"] = c

    volume_actor = p.add_mesh(volume_mesh, opacity = .5)
    predictions_actor = p.add_mesh(predictions_mesh, scalars = "colors", rgb = True)

    k = i % 5
    j = i // 5
    image = rotate_scene(p, *best_angles[i], transparent = transparent)
    plot.paste(image, box = (k * shape_image_size, j * shape_image_size))

    p.remove_actor(volume_actor)
    p.remove_actor(predictions_actor)

plot.save(f'results/paschalidou/abstraction_paschalidou.png')
