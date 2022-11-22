import os
import torch

from tulsiani.parameters import params
from tulsiani.network import TulsianiNetwork

def inference(dataset):
    params.grid_size = dataset[0].resized_volume.shape[0]
    params.phase = 0 if params.use_paschalidou_loss else 1

    path = 'results/tulsiani/save.torch'
    if not os.path.exists(path):
        return None

    model = TulsianiNetwork(params)
    model.load_state_dict(torch.load(path))
    model.eval()

    results = []
    with torch.no_grad():
        for start in range(0, len(dataset), params.batch_size):
            batch = torch.stack(
                [torch.Tensor(s.resized_volume) for s in dataset[start : start + params.batch_size]]
            )
            results.append(model(batch))

    return results
