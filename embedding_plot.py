import numpy as np
from trimesh import Trimesh
from PIL import Image
import pyvista as pv
from sklearn.manifold import TSNE
from tqdm import tqdm

from loader.load_urocell import load_urocell_preprocessed
from graphics.bruteforce_view import bruteforce_view, rotate_scene

from tulsiani.parameters import params
from tulsiani.inference import inference as tulsiani_inference
from yang.inference import inference as yang_inference

method = 'yang'

n_angles = 8 # Število vrednosti elevation in azimuth kota kamere
shape_image_size = 256
plot_image_size = 16384
# Da namesto parametrov primitivov vložimo notranjo predstavitev oblike v nevronski mreži:
use_internal_representation = False
existence_handling = 'existence' # [ 'existence', 'probability', 'exclude' ]
# Problem je pri mitohondrijih na robovih. Tam največjo površino tipično dobimo,
# če kamero fokusiramo na ploščat, odrezan del...
use_best_angles = False

validation, test = load_urocell_preprocessed(params.urocell_dir)
dataset = validation + test

n = len(dataset)
n_branched = sum(x.branched for x in dataset)
print(f'total: {n}')
print(f'branched: {n_branched}')

result_batches = tulsiani_inference(dataset) if method == 'tulsiani' else yang_inference(dataset, embedding_mode = True)

if use_internal_representation:
    dims = result_batches[0].outdict['z'].size()
    shape_parameters = np.zeros((n, dims[1], dims[2]))

    k = 0
    for batch in result_batches:
        representation = batch.outdict['z']
        m = representation.size(0)

        shape_parameters[k : k + m, ...] = representation.cpu().numpy()

        k += m
else:
    n_dims = 11 if existence_handling != 'exclude' else 10
    shape_parameters = np.zeros((n, params.n_primitives, n_dims))

    k = 0
    for batch in result_batches:
        m = batch.dims.size(0)

        shape_parameters[k : k + m, :, :3] = batch.dims.cpu().numpy()
        shape_parameters[k : k + m, :, 3:7] = batch.quat.cpu().numpy()
        shape_parameters[k : k + m, :, 7:10] = batch.trans.cpu().numpy()

        if existence_handling == 'existence':
            shape_parameters[k : k + m, :, 10] = batch.exist.cpu().numpy()
        elif existence_handling == 'probability':
            shape_parameters[k : k + m, :, 10] = batch.prob.cpu().numpy()

        k += m

shape_parameters.resize((shape_parameters.shape[0], shape_parameters.shape[1] * shape_parameters.shape[2]))

images = []
p = pv.Plotter(off_screen = True, window_size = [shape_image_size] * 2)

for i, shape in enumerate(tqdm(dataset)):
    m = Trimesh(shape.vertices, shape.faces - 1)
    # m.export(f'models/{i}.stl')
    mesh = pv.wrap(m)

    mesh_actor = p.add_mesh(mesh, color = 'red' if shape.branched else None)

    if use_best_angles:
        best_image, _ = bruteforce_view(p, n_angles, transparent = True)
    else:
        best_image = rotate_scene(p, 0, 0, transparent = True)

    p.remove_actor(mesh_actor)

    images.append(best_image)

# for i in range(n):
#     images[i].save(f'images/{i}.png')

embedding = TSNE(n_components = 2).fit_transform(shape_parameters)
embedding = embedding - embedding.min(0)
embedding = embedding / embedding.max(0)
s = plot_image_size - shape_image_size
embedding *= s
embedding = np.floor(embedding).clip(0, s - 1).astype(int)

plot = Image.new('RGBA', (plot_image_size,) * 2, (0, 0, 0, 0))

for i in range(n):
    plot.paste(images[i], box = (embedding[i, 0], embedding[i, 1]), mask = images[i])

image = Image.new('RGBA', (plot_image_size,) * 2, (76, 76, 76, 255))
image.paste(plot, mask = plot)
image = image.convert('RGB')

image.save(f'results/{method}/embedding_{method}_original.png')

image.resize((2048, 2048)).save(f'results/{method}/embedding_{method}_resized.png')
