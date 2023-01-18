import os
import random
import copy
import json
import numpy as np
from tensorboardX import SummaryWriter

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched
import torch.nn.functional as F
from torch.distributions.bernoulli import Bernoulli

from yang.parameters import hypara, save_path
from yang.network import Network_Whole
from yang.losses import loss_whole
import yang.utils_pytorch as utils_pt

from tulsiani.primitives import Primitives
from tulsiani.reinforce import ReinforceRewardUpdater

from common.batch_provider import BatchProvider, BatchProviderParams
from common.reconstruction_loss import reconstruction_loss
from common.iou import iou, IoUParams

from loader.load_preprocessed import load_preprocessed
from loader.load_urocell import load_urocell_preprocessed

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

seed = 123
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed) 

def get_primitives(out_dict, hypara):
    assign_matrix = out_dict['assign_matrix'] # batch_size * n_points * n_cuboids
    assigned_ratio = assign_matrix.mean(1)
    assigned_existence = (assigned_ratio > hypara['W']['W_min_importance_to_exist']).to(torch.float32).detach()

    # exist = out_dict_1['exist']
    # exist = F.sigmoid(exist.reshape(exist.size(0), -1))
    return Primitives(
        out_dict_1['scale'] * .5, # krat .5 zaradi kompatibilnosti s Tulsiani...
        out_dict_1['rotate_quat'],
        out_dict_1['pc_assign_mean'], # out_dict_1['trans'],
        assigned_existence
    )

class ReconstructionLossParams:
    def __init__(self, n_samples_per_primitive, use_chamfer):
        self.n_samples_per_primitive = n_samples_per_primitive
        self.use_chamfer = use_chamfer

# Ko je euclidean dual loss = True, reconstruction loss-a ne računamo po
# metodi Yang in Chen (ki upošteva normalne vektorje), temveč po "navadni"
# formuli iz Tulsiani in sod.
def compute_loss(loss_func, data, out_dict_1, out_dict_2, hypara, reinforce_updater = None):
    volume, points, closest_points, normals = data

    loss, loss_dict = loss_func(points, normals, out_dict_1, out_dict_2, hypara)

    if not hypara['W']['W_euclidean_dual_loss']:
        return loss, loss_dict

    use_reinforce = False

    if use_reinforce:
        prob = out_dict_1['exist']

        if False:
            prob = F.sigmoid(prob.reshape(prob.size(0), -1))

            distr = Bernoulli(prob)
            exist = distr.sample()
            log_prob = distr.log_prob(exist)

        P = Primitives(
            out_dict_1['scale'] * .5, # krat .5 zaradi kompatibilnosti s Tulsiani...
            out_dict_1['rotate_quat'],
            out_dict_1['trans'],
            torch.ones(prob.size(0), prob.size(1), device = prob.device),
        )

        params = ReconstructionLossParams(hypara['W']['W_n_samples_per_primitive'], hypara['W']['W_use_chamfer'])
        cov, cons = reconstruction_loss(volume, P, points, closest_points, params)

        l = (cov + cons).mean()

        if False:
            r = cov + cons

            if reinforce_updater: # Samo pri treniranju.
                existence_penalty = 8e-5
                for p in range(exist.size(1)):
                    penalty = r + existence_penalty * P.exist[:, p]
                    P.log_prob[:, p] *= reinforce_updater.update(penalty)

                l = r.mean() + P.log_prob.mean()
            else:
                l = r.mean()
    else:
        P = get_primitives(P, hypara)

        # distance = points_to_primitives_distance_squared(P, points) # batch_size * n_cuboids * n_points
        # cov = distance * assign_matrix.transpose(1, 2)
        # cov = cov.sum((1, 2))

        # cons = consistency(volume, P, closest_points, hypara['W']['W_n_samples_per_primitive'])

        params = ReconstructionLossParams(hypara['W']['W_n_samples_per_primitive'], hypara['W']['W_use_chamfer'])
        cov, cons = reconstruction_loss(volume, P, points, closest_points, params)

        l = (cov + cons).mean()

    loss_dict['eval'] += (l.data.detach().item() - loss_dict['REC']) * hypara['W']['W_REC']
    loss_dict['REC'] = l.data.detach().item()

    loss += l * hypara['W']['W_REC']

    loss_dict['ALL'] = loss.data.detach().item()

    return loss, loss_dict

def main():
    # save hyper-parameters to json
    with open(save_path + '/hypara.json', 'w') as f:
        json.dump(hypara, f)
    summary_writer = SummaryWriter(save_path + '/tensorboard')

    # Choose the CUDA device
    if 'E_CUDA' in hypara['E']:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(hypara['E']['E_CUDA'])

    train_set = load_preprocessed(hypara['E']['E_train_dir'])
    validation_set, test_set = load_urocell_preprocessed(hypara['E']['E_urocell_dir'])

    if hypara['E']['E_dont_use_split']:
        train_set += validation_set + test_set

    batch_params = BatchProviderParams(
        torch.device('cuda'),
        n_samples_per_shape = hypara['L']['L_n_samples_per_shape'],
        batch_size = hypara['L']['L_batch_size']
    )

    train_batches = BatchProvider(
        train_set,
        batch_params,
        store_on_gpu = False,
        include_normals = True,
        uses_point_sampling = hypara['L']['L_sample_points']
    )
    validation_batches = BatchProvider(
        validation_set,
        batch_params,
        store_on_gpu = False,
        include_normals = True,
        uses_point_sampling = hypara['L']['L_sample_points']
    )
    test_batches = BatchProvider(
        test_set,
        batch_params,
        store_on_gpu = False,
        include_normals = True,
        uses_point_sampling = hypara['L']['L_sample_points']
    )

    # Create Model
    Network = Network_Whole(hypara).cuda()
    Network.train()

    # Create Loss Function
    loss_func = loss_whole(hypara).cuda()

    # Create Optimizer
    optimizer = optim.Adam(
        Network.parameters(),
        lr = hypara['L']['L_base_lr'],
        betas = (hypara['L']['L_adam_beta1'], 0.999)
    )

    reinforce_updater = ReinforceRewardUpdater(.9)

    # Training Processing
    best_eval_loss = 100000
    color = utils_pt.generate_ncolors(hypara['N']['N_num_cubes'])
    num_batch = len(train_set) / hypara['L']['L_batch_size']
    batch_count = 0
    for epoch in range(hypara['L']['L_epochs']):
        for i, data in enumerate(train_batches.get_all_batches(shuffle = True)):
            optimizer.zero_grad()

            outdict = Network(pc = data[1])
            loss, loss_dict = compute_loss(loss_func, data, outdict, None, hypara, reinforce_updater)

            loss.backward()
            optimizer.step()

            utils_pt.print_text(
                loss_dict,
                save_path,
                is_train = True,
                epoch = epoch,
                i = i,
                num_batch = num_batch,
                lr = hypara['L']['L_base_lr'],
                print_freq_iter = hypara['E']['E_freq_print_iter']
            )
            batch_count += 1

            if batch_count % int(hypara['E']['E_freq_val_epoch'] * num_batch) == 0:
                utils_pt.train_summaries(summary_writer, loss_dict, batch_count * hypara['L']['L_batch_size'])

                new_best = hypara['E']['E_dont_use_split']

                if not hypara['E']['E_dont_use_split']:
                    eval_loss = validate(
                        hypara,
                        validation_batches,
                        Network,
                        loss_func,
                        save_path,
                        batch_count,
                        summary_writer,
                        color
                    )

                    if eval_loss < best_eval_loss:
                        best_eval_loss = eval_loss
                        new_best = True

                if not new_best:
                    continue

                model_name = utils_pt.create_name(batch_count, loss_dict)
                torch.save(Network.state_dict(), save_path + '/' + model_name + '.pth')

                Network.eval()

                with torch.no_grad():
                    save_points = next(test_batches.get_all_batches(shuffle = False))[1]
                    save_dict = Network(pc = save_points)

                    P = get_primitives(save_dict, hypara)
                    mean_iou = iou(data[0], P, IoUParams())
                    print(f'iter = {batch_count}, test IoU = {mean_iou}')

                Network.train()

                vertices, faces = utils_pt.generate_cube_mesh_batch(save_dict['verts_forward'], save_dict['cube_face'], hypara['L']['L_batch_size'])
                utils_pt.visualize_segmentation(save_points, color, save_dict['assign_matrix'], save_path + '/log/', 0, None)
                utils_pt.visualize_cubes(vertices, faces, color, save_path + '/log/', 0, '', None)
                utils_pt.visualize_cubes_masked(vertices, faces, color, save_dict['assign_matrix'], save_path + '/log/', 0, '', None)

                vertices_pred, faces_pred = utils_pt.generate_cube_mesh_batch(save_dict['verts_predict'], save_dict['cube_face'], hypara['L']['L_batch_size'])
                utils_pt.visualize_cubes(vertices_pred, faces_pred, color, save_path + '/log/', 0, 'pred', None)
                utils_pt.visualize_cubes_masked(vertices_pred, faces_pred, color, save_dict['assign_matrix'], save_path + '/log/', 0, 'pred', None)
                utils_pt.visualize_cubes_masked_pred(vertices_pred, faces_pred, color, save_dict['exist'], save_path + '/log/', 0, None)


def validate(hypara, validation_batches, Network, loss_func, save_path, iter, summary_writer, color):
    loss_dict = {}

    Network.eval()

    with torch.no_grad():
        for j, data in enumerate(validation_batches.get_all_batches(shuffle = False)):
            volume, points, closest_points, normals = data

            outdict = Network(pc = points)
            _, cur_loss_dict = compute_loss(loss_func, data, outdict, None, hypara)

            if loss_dict:
                for key in cur_loss_dict:
                    loss_dict[key] = loss_dict[key] + cur_loss_dict[key]
            else:
                loss_dict = cur_loss_dict

    Network.train()

    for key in loss_dict:
        loss_dict[key] = loss_dict[key] / (j+1)

    utils_pt.print_text(loss_dict, save_path, is_train = False)
    utils_pt.valid_summaries(summary_writer, loss_dict, iter * hypara['L']['L_batch_size'])

    return loss_dict['eval']

if __name__ == '__main__':
    main()
