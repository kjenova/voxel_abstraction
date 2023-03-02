import numpy as np
import torch

from loader.load_urocell import load_urocell_preprocessed

from tulsiani.parameters import params
from tulsiani.inference import inference as tulsiani_inference
from yang.inference import inference as yang_inference

method = 'yang'

# Da namesto parametrov primitivov vložimo notranjo predstavitev oblike v nevronski mreži:
use_internal_representation = False
existence_handling = 'existence' # [ 'existence', 'probability', 'exclude' ]

validation, test = load_urocell_preprocessed('data/urocell')
# validation, test = load_urocell_preprocessed('data/urocell_contacting', contacting = True)
dataset = validation + test

branched = [x for x in dataset if x.branched]
unbranched = [x for x in dataset if not x.branched]

n_experiments = 10

print(f'total: {len(dataset)}')
print(f'branched: {len(branched)}')

def get_results(x):
    result_batches = tulsiani_inference(x) if method == 'tulsiani' else yang_inference(x, embedding_mode = True)

    n = len(x)

    if use_internal_representation:
        # field = 'z'
        field = 'x_cuboid'

        dims = result_batches[0].outdict[field].size()
        shape_parameters = np.zeros((n, dims[1], dims[2]))

        k = 0
        for batch in result_batches:
            representation = batch.outdict[field]
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

    shape_parameters.resize((n, shape_parameters.shape[1] * shape_parameters.shape[2]))
    return shape_parameters

branched = get_results(branched)
unbranched = get_results(unbranched)

confusion_matrix = np.zeros((2, 2))

rng = np.random.default_rng(seed = 0x5EED)

def split(x):
    c = rng.choice(len(x), len(x) // 2, replace = False)
    i = np.zeros(len(x), dtype = bool)
    i[c] = True
    return torch.tensor(x[i]), torch.tensor(x[~i])

def avg_dist(a, b):
    return torch.cdist(a, b).mean(-1)

with torch.no_grad():
    for _ in range(n_experiments):
        b1, b2 = split(branched)
        u1, u2 = split(unbranched)

        bb = avg_dist(b1, b2)
        bu = avg_dist(b1, u2)

        for i in range(len(b1)):
            if bb[i] < bu[i]:
                confusion_matrix[1, 1] += 1 # true positive
            else:
                confusion_matrix[1, 0] += 1 # false negative

        ub = avg_dist(u1, b2)
        uu = avg_dist(u1, u2)

        for i in range(len(u1)):
            if uu[i] < ub[i]:
                confusion_matrix[0, 0] += 1 # true negative
            else:
                confusion_matrix[0, 1] += 1 # false positive

precision = confusion_matrix[1, 1] / confusion_matrix[:, 1].sum()
print(f'precision: {precision}')
recall = confusion_matrix[1, 1] / confusion_matrix[1, :].sum()
print(f'recall: {recall}')

confusion_matrix /= n_experiments * (len(branched) // 2 + len(unbranched) // 2)
print(confusion_matrix)
