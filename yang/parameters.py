import argparse
import os

parser = argparse.ArgumentParser()

# Experiment(E) hyper-parameters
parser.add_argument('--E_name', default = 'EXP_1', type = str, help = 'Experiment name')
parser.add_argument('--E_freq_val_epoch', default = 1, type = float, help = 'Frequency of validation')
parser.add_argument('--E_freq_print_iter', default = 10, type = int, help = 'Frequency of print')
parser.add_argument('--E_CUDA', default = 1, type = int, help = 'Index of CUDA')
parser.add_argument('--E_train_dir', default = 'data/train', type = str)
parser.add_argument('--E_urocell_dir', default = 'data/urocell', type = str)
parser.add_argument('--E_save_dir', default = 'results/yang', type = str)

parser.add_argument('--E_dont_use_split', action = 'store_true')

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
parser.add_argument('--N_num_cubes', default = 20, type = int, help = 'Number of cuboids')
parser.add_argument('--N_nonvariational_network', action = 'store_true')
# Da nimamo samo enega linearnega sloja, ki iz globalnih značilk + one-hot encoding-a indeksa primitiva
# napove parametre primitiva, ampak da tako kot v Tulsiani et al. parametre napovemo za vsak primitiv posebej.
parser.add_argument('--N_separate_primitive_encoding', action = 'store_true')

# Weight(W) hyper-parameters of losses
# Če je ta flag nastavljen, se za reconstruction loss uporablja loss iz Tulsiani in sod., drugače
# pa loss iz Yang in Chen (ki upošteva ujemanje normalnih vektorjev s ploskvami).
parser.add_argument('--W_euclidean_dual_loss', action = 'store_true')
# Ta parameter pride v poštev samo, ko je W_euclidean_dual_loss = True.
parser.add_argument('--W_use_chamfer', action = 'store_true')
# Se uporablja za reconstruction loss iz Tulsiani in sod.:
parser.add_argument('--W_n_samples_per_primitive', default = 150, type = int)
parser.add_argument('--W_REC', default = 1.00, type = float, help = 'REC loss weight')
parser.add_argument('--W_std', default = 0.05, type = float, help = 'std of normal sampling')
parser.add_argument('--W_SPS', default = 0.10, type = float, help = 'SPS loss weight')
parser.add_argument('--W_EXT', default = 0.01, type = float, help = 'EXT loss weight')
parser.add_argument('--W_KLD', default = 6e-6, type = float, help = 'KLD loss weight')
parser.add_argument('--W_CST', default = 0.00, type = float, help = 'CST loss weight, this loss is only for generation application')
parser.add_argument('--W_min_importance_to_exist', default = 0.05, type = float)

args = parser.parse_args()

# parsing hyper-parameters to dict
hypara = {}
hypara['E'] = {}
hypara['L'] = {}
hypara['W'] = {}
hypara['N'] = {}
for arg in vars(args):
    hypara[str(arg)[0]][str(arg)] = getattr(args, arg)

if hypara['W']['W_euclidean_dual_loss']:
    hypara['W']['W_EXT'] = .0
    hypara['W']['W_SPS'] = .0
    hypara['W']['W_CST'] = .0
    # Kullback-Leiblerja razdalja pa ostane.

if hypara['N']['N_nonvariational_network']:
    hypara['W']['W_KLD'] = .0

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

