import numpy as np
from trimesh import Trimesh
from PIL import Image
import pyvista as pv
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from loader.load_urocell import load_urocell_preprocessed
from graphics.bruteforce_view import bruteforce_view, rotate_scene

n_components = 10

shape_image_size = 256
plot_image_size = 16384

validation, test = load_urocell_preprocessed('data/urocell')
# validation, test = load_urocell_preprocessed('data/urocell_contacting', contacting = True)
dataset = validation + test

p = pv.Plotter(off_screen = True, window_size = [shape_image_size] * 2)

features = np.zeros((len(dataset), n_components * (n_components - 1)))

for i, shape in enumerate(tqdm(dataset)):
    m = Trimesh(shape.vertices, shape.faces - 1)
    mesh = pv.wrap(m)

    mesh_actor = p.add_mesh(mesh, color = 'red' if shape.branched else None)
    shape.image = rotate_scene(p, 0, 0, transparent = True)
    p.remove_actor(mesh_actor)

    pca = PCA(n_components = n_components)
    pca.fit(shape.vertices)

    similarities = cosine_similarity(pca.components_)
    features[i] = similarities[~np.eye(n_components, dtype = bool)]

embedding = TSNE(n_components = 2).fit_transform(features)
embedding = embedding - embedding.min(0)
embedding = embedding / embedding.max(0)
s = plot_image_size - shape_image_size
embedding *= s
embedding = np.floor(embedding).clip(0, s - 1).astype(int)

plot = Image.new('RGBA', (plot_image_size,) * 2, (0, 0, 0, 0))

for i, shape in enumerate(dataset):
    plot.paste(shape.image, box = (embedding[i, 0], embedding[i, 1]), mask = shape.image)

image = Image.new('RGBA', (plot_image_size,) * 2, (76, 76, 76, 255))
image.paste(plot, mask = plot)
image = image.convert('RGB')

image.save('pca.png')
image.resize((2048, 2048)).save('pca_resized.png')
