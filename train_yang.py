import os
import random
import copy
import argparse
import json
import numpy as np
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from yang.network import Network_Whole
from yang.losses import loss_whole
import yang.utils_pytorch as utils_pt

from tulsiani.primitives import Primitives

from common.batch_provider import BatchProvider, BatchProviderParams
from common.reconstruction_loss import points_to_primitives_distance_squared, consistency

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

def parsing_hyperparas(args):
    # parsing hyper-parameters to dict
    hypara = {}
    hypara['E'] = {}
    hypara['L'] = {}
    hypara['W'] = {}
    hypara['N'] = {}
    for arg in vars(args):
        hypara[str(arg)[0]][str(arg)] = getattr(args, arg)
    # get the save_path
    save_path = hypara['E']['E_save_dir'] + '/' + hypara['E']['E_name']
    save_path = save_path + '-L'
    for key in hypara['L']:
        save_path = save_path + '_' + str(hypara['L'][key])
    save_path = save_path + '-N'
    for key in hypara['N']:
        save_path = save_path + '_' + str(hypara['N'][key])
    save_path = save_path + '-W'
    for key in hypara['W']:
        save_path = save_path + '_' + str(hypara['W'][key])
    if not os.path.exists(save_path + '/log/'): 
        os.makedirs(save_path + '/log/')
    # save hyper-parameters to json
    with open(save_path + '/hypara.json', 'w') as f:
        json.dump(hypara, f)
    summary_writer = SummaryWriter(save_path + '/tensorboard')

    return hypara, save_path, summary_writer

# Ko je euclidean dual loss = True, reconstruction loss-a ne računamo po
# metodi Yang in Chen (ki upošteva normalne vektorje), temveč po "navadni"
# formuli iz Tulsiani in sod.
def compute_loss(loss_func, data, out_dict_1, out_dict_2, hypara):
    volume, points, closest_points, normals = data

    loss, loss_dict = loss_func(points, normals, out_dict_1, out_dict_2, hypara)

    if not hypara['W']['W_euclidean_dual_loss']:
        return loss, loss_dict

    assign_matrix = out_dict_1['assign_matrix'] # batch_size * n_points * n_cuboids
    assigned_ratio = assign_matrix.mean(1)

    # exist = out_dict_1['exist']
    P = Primitives(
        out_dict_1['scale'],
        out_dict_1['rotate_quat'],
        out_dict_1['pc_assign_mean'], # out_dict_1['trans'],
        assigned_ratio # F.sigmoid(exist.reshape(exist.size(0), -1))
    )

    distance = points_to_primitives_distance_squared(P, points) # batch_size * n_cuboids * n_points
    cov = distance * assign_matrix.transpose(1, 2)
    cov = cov.sum((1, 2))

    cons = consistency(volume, P, closest_points, hypara['W']['W_n_samples_per_primitive'])

    r = (cov + cons).mean() * hypara['W']['W_REC']
    loss_dict['eval'] = (r.data.detach().item() - loss_dict['REC']) * hypara['W']['W_REC']
    loss_dict['REC'] = r.data.detach().item()

    loss += r * hypara['W']['W_REC']

    loss_dict['ALL'] = loss.data.detach().item()

    return loss, loss_dict

def main(args):
    hypara, save_path, summary_writer = parsing_hyperparas(args)

    # Choose the CUDA device
    if 'E_CUDA' in hypara['E']:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(hypara['E']['E_CUDA'])

    train_set = load_preprocessed(hypara['E']['E_train_dir'])
    validation_set, _ = load_urocell_preprocessed(hypara['E']['E_urocell_dir'])

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

    # Training Processing
    best_eval_loss = 100000
    color = utils_pt.generate_ncolors(hypara['N']['N_num_cubes'])
    num_batch = len(train_set) / hypara['L']['L_batch_size']
    batch_count = 0
    for epoch in range(hypara['L']['L_epochs']):
        for i, data in enumerate(train_batches.get_all_batches(shuffle = True)):
            optimizer.zero_grad()

            outdict = Network(pc = data[1])
            loss, loss_dict = compute_loss(loss_func, data, outdict, None, hypara)

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
                best_eval_loss = validate(
                    hypara,
                    validation_batches,
                    Network,
                    loss_func,
                    hypara['W'],
                    save_path,
                    batch_count,
                    epoch,
                    summary_writer,
                    best_eval_loss,
                    color
                )
                Network.train()

def validate(hypara, validation_batches, Network, loss_func, loss_weight, save_path, iter, epoch, summary_writer, best_eval_loss, color):
    Network.eval()
    loss_dict = {}
    for j, data in enumerate(validation_batches.get_all_batches(shuffle = False)):
        with torch.no_grad():
            volume, points, closest_points, normals = data

            outdict = Network(pc = points)
            _, cur_loss_dict = compute_loss(loss_func, data, outdict, None, hypara)
            if j == 0:
                save_points = points
                save_dict = outdict
            if loss_dict:
                for key in cur_loss_dict:
                    loss_dict[key] = loss_dict[key] + cur_loss_dict[key]
            else:
                loss_dict = cur_loss_dict

    for key in loss_dict:
        loss_dict[key] = loss_dict[key] / (j+1)

    utils_pt.print_text(loss_dict, save_path, is_train = False)
    utils_pt.valid_summaries(summary_writer, loss_dict, iter * hypara['L']['L_batch_size'])

    if loss_dict['eval'] < best_eval_loss:
        best_eval_loss = copy.deepcopy(loss_dict['eval'])
        print('eval: ',best_eval_loss)
        if epoch >= 0:
            model_name = utils_pt.create_name(iter, loss_dict)
            torch.save(Network.state_dict(), save_path + '/' + model_name + '.pth')

            torch.save(Network.state_dict(), hypara['E']['E_save_dir'] + '/save.torch')

            vertices, faces = utils_pt.generate_cube_mesh_batch(save_dict['verts_forward'], save_dict['cube_face'], hypara['L']['L_batch_size'])
            utils_pt.visualize_segmentation(save_points, color, save_dict['assign_matrix'], save_path + '/log/', 0, None)
            utils_pt.visualize_cubes(vertices, faces, color, save_path + '/log/', 0, '', None)
            utils_pt.visualize_cubes_masked(vertices, faces, color, save_dict['assign_matrix'], save_path + '/log/', 0, '', None)

            vertices_pred, faces_pred = utils_pt.generate_cube_mesh_batch(save_dict['verts_predict'], save_dict['cube_face'], hypara['L']['L_batch_size'])
            utils_pt.visualize_cubes(vertices_pred, faces_pred, color, save_path + '/log/', 0, 'pred', None)
            utils_pt.visualize_cubes_masked(vertices_pred, faces_pred, color, save_dict['assign_matrix'], save_path + '/log/', 0, 'pred', None)
            utils_pt.visualize_cubes_masked_pred(vertices_pred, faces_pred, color, save_dict['exist'], save_path + '/log/', 0, None)
    
    return best_eval_loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Experiment(E) hyper-parameters
    parser.add_argument('--E_name', default = 'EXP_1', type = str, help = 'Experiment name')
    parser.add_argument('--E_freq_val_epoch', default = 1, type = float, help = 'Frequency of validation')
    parser.add_argument('--E_freq_print_iter', default = 10, type = int, help = 'Frequency of print')
    parser.add_argument('--E_CUDA', default = 1, type = int, help = 'Index of CUDA')
    parser.add_argument('--E_train_dir', default = 'data/chamferData/01', type = str)
    parser.add_argument('--E_urocell_dir', default = 'data/chamferData/urocell', type = str)
    parser.add_argument('--E_save_dir', default = 'results/yang', type = str)

    # Learning(L) hyper-parameters
    parser.add_argument('--L_base_lr', default = 6e-4, type = float, help = 'Learning rate')
    parser.add_argument('--L_adam_beta1', default = 0.9, type = float, help = 'Adam beta1')
    parser.add_argument('--L_batch_size', default = 32, type = int, help = 'Batch size')
    parser.add_argument('--L_epochs', default = 1000, type = int, help = 'Number of epochs')
    parser.add_argument('--L_sample_points', default = True, action = 'store_true')
    parser.add_argument('--L_n_samples_per_shape', default = 1000, type = int)

    # Network(N) hyper-parameters`
    parser.add_argument('--N_if_low_dim', default = 0, type = int, help = 'DGCNN paramter: KNN manner')
    parser.add_argument('--N_k', default = 20, type = int, help = 'DGCNN paramter: K of KNN')
    parser.add_argument('--N_dim_emb', default = 1024, type = int, help = 'Dimension of global feature')
    parser.add_argument('--N_dim_z', default = 512, type = int, help = 'Dimension of latent code Z')
    parser.add_argument('--N_dim_att', default = 64, type = int, help = 'Dimension of query and key in attention')
    parser.add_argument('--N_num_cubes', default = 16, type = int, help = 'Number of cuboids')

    # Weight(W) hyper-parameters of losses
    # Če je ta flag nastavljen, se za reconstruction loss uporablja loss iz Tulsiani in sod., drugače
    # pa loss iz Yang in Chen (ki upošteva ujemanje normalnih vektorjev s ploskvami).
    parser.add_argument('--W_euclidean_dual_loss', action = 'store_true')
    # Se uporablja za reconstruction loss iz Tulsiani in sod.:
    parser.add_argument('--W_n_samples_per_primitive', default = 150, type = int)
    parser.add_argument('--W_REC', default = 1.00, type = float, help = 'REC loss weight')
    parser.add_argument('--W_std', default = 0.05, type = float, help = 'std of normal sampling')
    parser.add_argument('--W_SPS', default = 0.10, type = float, help = 'SPS loss weight')
    parser.add_argument('--W_EXT', default = 0.01, type = float, help = 'EXT loss weight')
    parser.add_argument('--W_KLD', default = 6e-6, type = float, help = 'KLD loss weight')
    parser.add_argument('--W_CST', default = 0.00, type = float, help = 'CST loss weight, this loss is only for generation application')

    args = parser.parse_args()
    main(args)
