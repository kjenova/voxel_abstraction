from glob import glob
import re
import torch

from yang.parameters import hypara, save_path
from yang.network import Network_Whole

from tulsiani.primitives import Primitives

from common.batch_provider import BatchProvider, BatchProviderParams

def inference(dataset):
    model_path = None
    max_iter = -1
    for filename in glob(f"{save_path}/*.pth"):
        it = int(re.match(r'.*?iter(\d+)', filename)[1])
        if it > max_iter:
            max_iter = it
            model_path = filename

    if model_path is None:
        return None

    model = Network_Whole(hypara).cuda()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    batch_params = BatchProviderParams(
        torch.device('cuda'),
        n_samples_per_shape = hypara['L']['L_n_samples_per_shape'],
        batch_size = hypara['L']['L_batch_size']
    )

    test_batches = BatchProvider(
        dataset,
        batch_params,
        store_on_gpu = False,
        include_normals = False,
        uses_point_sampling = hypara['L']['L_sample_points']
    )

    results = []
    with torch.no_grad():
        for _, points, _ in test_batches.get_all_batches():
            outdict = model(pc = points)

            assign_matrix = outdict['assign_matrix'] # batch_size * n_points * n_cuboids
            assigned_ratio = assign_matrix.mean(1)
            assigned_existence = (assigned_ratio > .02).to(torch.float32).detach()

            P = Primitives(
                outdict['scale'] * .5, # krat .5 zaradi kompatibilnosti s Tulsiani...
                outdict['rotate_quat'],
                outdict['pc_assign_mean'],
                assigned_existence,
                assigned_existence,
                None
            )

            results.append(P)

    return results
