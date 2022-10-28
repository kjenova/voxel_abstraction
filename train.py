import torch
import numpy as np

from tulsiani.network import TulsianiNetwork, TulsianiParams
from tulsiani.net_utils import scale_weights
from tulsiani.reinforce import ReinforceRewardUpdater
from tulsiani.stats import TulsianiStats

from common.cuboid import CuboidSurface
from common.reconstruction_losses import reconstruction_loss

from loader.load_preprocessed import load_preprocessed
from loader.load_urocell import load_urocell_preprocessed

params = TulsianiParams()

params.train_dir = 'data/chamferData/01'
params.urocell_dir = 'data/chamferData/urocell'

# Število iteracij treniranja v prvi in drugi fazi:
params.n_iterations = [20000, 30000]
# Toliko iteracij se vsak batch ponovi (t.j. po tolikšnem številu
# naložimo nov batch), glej 'params.modelIter' v referenčni implementaciji.
# (Bolj strogo gledano se na toliko iteracij ponovi batch z istimi modeli,
# ker potem še za vsako iteracijo vzorčimo podmnožico točk na površini oblike.)
params.repeat_batch_n_iterations = 2
# Na vsake toliko iteracij se shrani model, če je validacijski loss manjši:
params.save_iteration = 1000

# Ali napovedujemo prisotnost primitivov (True) ali pa kar vedno vzamemo vse (False):
params.prune_primitives = True
params.n_primitives = 20
params.grid_size = 64
# Iz vsake oblike smo med predprocesiranjem vzorčili 10.000 točk:
params.n_points_per_shape = 10000
# Vendar naenkrat bomo upoštevali samo 1000 naključnih točk:
params.n_samples_per_shape = 1000
params.n_samples_per_primitive = 150

params.batch_size = 32
params.use_batch_normalization_conv = True
params.use_batch_normalization_linear = True
params.learning_rate = 1e-3
params.reinforce_baseline_momentum = .9

# S faktorjem 'dims_factor' dosežemo, da je aktivacija za napovedovanje
# dimenzij bolj "razpotegnjena" (dims = sigmoid(dims_factor * features)).
# Tako kompenziramo nagnjenje metode k temu, da slabo ujemajoče kvadre
# samo zmanjša namesto, da bi poiskala boljše translacije in rotacije.
# Z drugimi besedami: dimenzij se učimo počasneje kot translacije in rotacije
# (https://github.com/nileshkulkarni/volumetricPrimitivesPytorch/issues/3).
# Vrednost tega faktorja je večja v drugi fazi, kjer problem zmanjšanja
# slabo ujemajočih kvadrov ni tako izrazit, saj se jim lahko samo zmanjša
# verjetnost.
params.dims_factors = [0.01, 0.5] # prva in druga faza

# Ta faktor ima isto nalogo, ampak je za napovedovanje verjetnosti.
# V prvi fazi učenja je faktor zelo majhen, saj poskusimo optimizirati ostale
# parametre, preden dovolimo, da se nad kvadrom "obupa".
params.prob_factors = [0.0001, 0.2]

# V drugi fazi tudi rahlo penaliziramo prisotnost.
params.existence_penalties = [.0, 8e-5]

# cuda:1 = Titan X
params.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

def train(network, train_batches, validation_batches, params, stats):
    optimizer = torch.optim.Adam(network.parameters(), lr = params.learning_rate)
    sampler = CuboidSurface(params.n_samples_per_primitive)
    reinforce_updater = ReinforceRewardUpdater(params.reinforce_baseline_momentum)

    best_validation_loss = float('inf')

    network.train()
    for _ in range(params.n_iterations[params.phase]):
        optimizer.zero_grad()

        (volume, sampled_points, closest_points) = train_batches.get()
        P = network(volume, params)
        cov, cons = reconstruction_loss(volume, P, sampled_points, closest_points, params)
        l = cov + cons

        total_penalty = .0

        if prune_primitives:
            # Pri nas se minimizira 'penalty', čeprav se pri REINFORCE tipično maksimizira 'reward'.
            # Edina razlika je v tem, da bi v primeru, da bi maksimizirali 'reward = - penalty', morali
            # potem še pri gradientu dodati minus.
            for p in range(n_primitives):
                penalty = l + params.existence_penalty * P.exist[:, p]
                total_penalty += penalty.mean().item()
                P.log_prob[:, p] *= reinforce_updater.update(penalty)

            # Ker je navaden loss reduciran z .mean(), tudi ta "REINFORCE loss" reduciram z .mean():
            (l.mean() + P.log_prob.mean()).backward()
        else:
            l.mean().backward()

        optimizer.step()

        i = train_batches.iteration - 1
        stats.cov[i] = cov.mean()
        stats.cons[i] = cons.mean()
        stats.prob_means[i] = P.prob.mean()
        stats.penalty_means[i] = total_penalty / n_primitives
        i += 1

        if i % params.save_iteration == 0:
            cov_mean = stats.cov[i - save_iteration : i].mean()
            cons_mean = stats.cons[i - save_iteration : i].mean()
            mean_prob = stats.prob_means[i - save_iteration : i].mean()
            mean_penalty = stats.penalty_means[i - save_iteration : i].mean()

            print(f'---- iteration {i} ----')
            print(f'    loss {cov_mean + cons_mean}, cov: {cov_mean}, cons: {cons_mean}')
            print(f'    mean prob: {mean_prob}')
            print(f'    mean penalty: {mean_penalty}')

            validation_loss = .0

            network.eval()

            with torch.no_grad():
                for (volume, sampled_points, closest_points) in validation_batches.get_all_batches():
                    P = network(volume, params)
                    cov, cons = reconstruction_loss(volume, P, sampled_points, closest_points, params)
                    validation_loss += (cov + cons).sum()

            network.train()

            validation_loss /= validation_batches.n

            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                torch.save(network.state_dict(), 'results/tulsiani/save.torch')

            j = train_batches.iteration // params.save_iteration - 1
            stats.validation_loss[j] = validation_loss

            print(f'    validation loss: {validation_loss}')
            print(f'    best validation loss: {best_validation_loss}')

if __name__ == '__main__':
    os.makedirs('results/tulsiani/graphs')

    train_set = load_preprocessed(params.train_dir)

    validation_set, _ = load_urocell_preprocessed(params.urocell_dir)

    train_batches = BatchProvider(train_set, params, store_on_gpu = False)
    validation_batches = BatchProvider(validation_set, params, store_on_gpu = False)

    stats = TulsianiStats()

    params.phase = 0
    network = TulsianiNetwork(params)
    network.to(params.device)

    train(network, train_batches, validation_batches, params, stats)

    params.phase = 1
    network = TulsianiNetwork(params)
    network.load_state_dict(torch.load('results/tulsiani/save.torch'))
    scale_weights(network)
    network.to(params.device)

    train(network, train_batches, validation_batches, params, stats)

    stats.save_plots('results/tulsiani/graphs')
