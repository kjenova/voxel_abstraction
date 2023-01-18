import os
import torch

from tulsiani.parameters import params
from tulsiani.network import TulsianiNetwork

from common.iou import iou, IoUParams

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
        total_iou = .0
        n = 0

        for start in range(0, len(dataset), params.batch_size):
            batch = torch.stack(
                [torch.Tensor(s.resized_volume) for s in dataset[start : start + params.batch_size]]
            )
            P = model(batch)

            results.append(P)

            P.exist = (P.prob > .5).long()
            total_iou += iou(batch, P, params).sum()
            n += P.dims.size(0)

        print(f'tulsiani test IoU: {total_iou / n}')

    return results
