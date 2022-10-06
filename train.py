import torch
import torch.nn as nn
import numpy as np
import random
import matplotlib.pyplot as plt
from volume_encoder import VolumeEncoder
from net_utils import weights_init
from primitives import PrimitivesPrediction
from cuboid import CuboidSurface
from losses import loss
from reinforce import ReinforceRewardUpdater
from load_shapes import load_shapes, Shape
from load_shapenet import load_shapenet, ShapeNetShape
from generate_mesh import predictions_to_mesh
from write_mesh import write_volume_mesh, write_predictions_mesh

random.seed(0x5EED)

shapenet_dir = 'shapenet/chamferData/00'
# Koliko povezanih komponent vzamemo pri celičnih predelkih
# (pri ShapeNet-u vzamemo vse učne primere):
n_examples = 2000
# Število iteracij treniranja v prvi in drugi fazi:
n_iterations = [20000, 30000]
# Toliko iteracij se vsak batch ponovi (t.j. po tolikšnem številu
# naložimo nov batch), glej 'params.modelIter' v referenčni implementaciji.
# (Bolj strogo gledano se na toliko iteracij ponovi batch z istimi modeli,
# ker potem še za vsako iteracijo vzorčimo podmnožico točk na površini oblike.)
repeat_batch_n_iterations = 2
# Na vsake toliko iteracij se izpiše statistika:
output_iteration = 1000
# Na vsake toliko iteracij se shrani napovedane primitive
save_mesh_iteration = sum(n_iterations) # samo na koncu
# za toliko učnih primerov:
n_examples_for_visualization = 100
batch_size = 32
use_batch_normalization_conv = True
use_batch_normalization_linear = True
# Ali napovedujemo prisotnost primitivov (True) ali pa kar vedno vzamemo vse (False):
prune_primitives = True
n_primitives = 20
grid_size = 32
# Iz vsake oblike smo med predprocesiranjem vzorčili 10.000 točk:
n_points_per_shape = 10000
# Vendar naenkrat bomo upoštevali samo 1000 točk (na novo vzorčimo vsako epoho):
n_samples_per_shape = 1000
n_samples_per_primitive = 150
learning_rate = 1e-3
reinforce_baseline_momentum = .9

# S faktorjem 'dims_factor' dosežemo, da je aktivacija za napovedovanje
# dimenzij bolj "razpotegnjena" (dims = sigmoid(dims_factor * features)).
# Tako kompenziramo nagnjenje metode k temu, da slabo ujemajoče kvadre
# samo zmanjša namesto, da bi poiskala boljše translacije in rotacije.
# Z drugimi besedami: dimenzij se učimo počasneje kot translacije in rotacije
# (https://github.com/nileshkulkarni/volumetricPrimitivesPytorch/issues/3).
# Vrednost tega faktorja je večja v drugi fazi, kjer problem zmanjšanja
# slabo ujemajočih kvadrov ni tako izrazit, saj se jim lahko samo zmanjša
# verjetnost.
dims_factors = [0.01, 0.5] # prva in druga faza

# Ta faktor ima isto nalogo, ampak je za napovedovanje verjetnosti.
# V prvi fazi učenja je faktor zelo majhen, saj poskusim optimizirati ostale
# parametre, preden dovolimo, da se nad kvadrom "obupa".
prob_factors = [0.0001, 0.2]

# V drugi fazi tudi rahlo penaliziramo prisotnost.
existence_penalties = [.0, 8e-5]

# cuda:1 = Titan X
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

class PhaseParams:
    def __init__(self, phase):
        self.phase = phase
        self.n_iterations = n_iterations[phase]
        self.dims_factor = dims_factors[phase]
        self.prob_factor = prob_factors[phase]
        self.existence_penalty = existence_penalties[phase]
        self.prune_primitives = prune_primitives

class Network(nn.Module):
    def __init__(self, params):
        super().__init__()

        self.encoder = VolumeEncoder(5, 4, 1, use_batch_normalization_conv)
        n = self.encoder.n_out_channels

        layers = []
        for _ in range(2):
            linear = nn.Linear(n, n, bias = not use_batch_normalization_linear)
            weights_init(linear)
            layers.append(linear)

            if use_batch_normalization_linear:
                batch_norm = nn.BatchNorm1d(n)
                weights_init(batch_norm)
                layers.append(batch_norm)

            layers.append(nn.LeakyReLU(0.2, inplace = True))

        self.fc_layers = nn.Sequential(*layers)
        self.primitives = PrimitivesPrediction(n, n_primitives, params)

    def forward(self, volume, params):
        x = self.encoder(volume.unsqueeze(1))
        x = x.reshape(volume.size(0), self.encoder.n_out_channels)
        x = self.fc_layers(x)
        return self.primitives(x, params)

class BatchProvider:
    def __init__(self, shapes):
        self.volume = torch.stack([torch.from_numpy(s.resized_volume.astype(np.float32)) for s in shapes]).to(device)
        self.shape_points = torch.stack([torch.from_numpy(s.shape_points) for s in shapes]).to(device)
        self.closest_points = torch.stack([s.closest_points for s in shapes]).to(device)

        self.iteration = 0

    def load_examples(self):
        n = self.volume.size(0)
        indices = torch.randint(0, n, (min(batch_size, n),), device = device)
        self.loaded_volume = self.volume[indices]
        self.loaded_shape_points = self.shape_points[indices]
        self.loaded_closest_points = self.closest_points[indices]

    def get(self):
        if self.iteration % repeat_batch_n_iterations == 0:
            self.load_examples()

        self.iteration += 1

        [b, n] = self.loaded_shape_points.size()[:2]
        i = torch.randint(0, n, (b, n_samples_per_shape), device = device)
        i += n * torch.arange(0, b, device = device).reshape(-1, 1)
        sampled_points = self.loaded_shape_points.reshape(-1, 3)[i]
        return (self.loaded_volume, sampled_points, self.loaded_closest_points)

    def get_batches_for_visualization(self):
        batches = []
        for i in range(0, n_examples_for_visualization, batch_size):
            n = min(batch_size, n_examples_for_visualization - i)
            batches.append(self.volume[i : i + n])
        return batches

class Stats:
    def __init__(self):
        total_n_iters = sum(n_iterations)
        self.cov = np.zeros(total_n_iters)
        self.cons = np.zeros(total_n_iters)
        self.prob_means = np.zeros(total_n_iters)
        self.penalty_means = np.zeros(total_n_iters)

    def save_plots(self):
        x = np.arange(1, sum(n_iterations) + 1)

        plt.figure(figsize = (20, 5))
        plt.plot(x, self.cov)
        plt.xlabel('iteration')
        plt.ylabel('coverage')
        plt.savefig('graphs/coverage.png')

        plt.clf()
        plt.figure(figsize = (20, 5))
        plt.plot(x, self.cons)
        plt.xlabel('iteration')
        plt.ylabel('consistency')
        plt.savefig('graphs/consistency.png')

        plt.clf()
        plt.figure(figsize = (20, 5))
        plt.plot(x, self.prob_means)
        plt.xlabel('iteration')
        plt.ylabel('mean probability')
        plt.savefig('graphs/probability.png')

        plt.clf()
        plt.figure(figsize = (20, 5))
        plt.plot(x, self.penalty_means)
        plt.xlabel('iteration')
        plt.ylabel('penalty')
        plt.savefig('graphs/penalty.png')

def _scale_weights(m, f):
    r = f[0] / f[1]
    m.weight.data *= r
    m.bias.data *= r

def scale_weights(network):
    # Ker v drugi fazi učenja zamenjamo 'dims_factor', mu tudi prilagodimo cel
    # 'scale' polno povezanega sloja za napovedovanje dimenzij.
    _scale_weights(network.primitives.dims.layer, dims_factors)

    # Enako za 'prob_factor'.
    _scale_weights(network.primitives.prob.layer, prob_factors)

def train(network, batch_provider, params, stats):
    if params.phase == 1:
        scale_weights(network)

    optimizer = torch.optim.Adam(network.parameters(), lr = learning_rate)
    sampler = CuboidSurface(n_samples_per_primitive)
    reinforce_updater = ReinforceRewardUpdater(reinforce_baseline_momentum)

    network.train()
    for _ in range(params.n_iterations):
        optimizer.zero_grad()

        (volume, sampled_points, closest_points) = batch_provider.get()
        P = network(volume, params)
        cov, cons = loss(volume, P, sampled_points, closest_points, sampler)
        l = cov + cons

        total_penalty = .0

        if prune_primitives:
            # Pri nas se minimizira 'penalty', čeprav se pri REINFORCE tipično maksimizira 'reward'.
            # Edina razlika je v tem, da bi v primeru, da bi maksimizirali 'reward = - penalty', morali
            # potem še pri gradientu dodati minus.
            for p in range(n_primitives):
                # Glede na članek bi bila sledeča formula, ampak če bi sledili referenčni implementaciji
                # bi pa bilo 'l + params.existence_penalty * torch.sum(P.exist[:, p])':
                # https://github.com/nileshkulkarni/volumetricPrimitivesPytorch/blob/367d2bc3f7d2ec122c4e2066c2ee2a922cf4e0c8/experiments/cadAutoEncCuboids/primSelTsdfChamfer.py#L142
                penalty = l + params.existence_penalty * P.exist[:, p]
                total_penalty += penalty.mean().item()
                P.log_prob[:, p] *= reinforce_updater.update(penalty)

            # Ker je navaden loss reduciran z .mean(), tudi ta "REINFORCE loss" reduciram z .mean():
            (l.mean() + P.log_prob.mean()).backward()
        else:
            l.mean().backward()

        optimizer.step()

        i = batch_provider.iteration - 1
        stats.cov[i] = cov.mean()
        stats.cons[i] = cons.mean()
        stats.prob_means[i] = P.prob.mean()
        stats.penalty_means[i] = total_penalty / n_primitives

        if batch_provider.iteration % output_iteration == 0:
            i += 1
            cov_mean = stats.cov[i - output_iteration : i].mean()
            cons_mean = stats.cons[i - output_iteration : i].mean()
            mean_prob = stats.prob_means[i - output_iteration : i].mean()
            mean_penalty = stats.penalty_means[i - output_iteration : i].mean()

            print(f'---- iteration {i} ----')
            print(f'    loss {cov_mean + cons_mean}, cov: {cov_mean}, cons: {cons_mean}')
            print(f'    mean prob: {mean_prob}')
            print(f'    mean penalty: {mean_penalty}')

        if batch_provider.iteration % save_mesh_iteration == 0:
            network.eval()
            with torch.no_grad():
                parameters = np.zeros(n_examples, n_primitives, 7)

                k = 0
                for volume_batch in batch_provider.get_batches_for_visualization():
                    X = network(volume_batch, params)
                    vertices = predictions_to_mesh(X).cpu()
                    nv = vertices.size(0)

                    parameters[k : k + nv, :, :3] = X.dims
                    parameters[k : k + nv, :, 3:7] = X.quat
                    parameters[k : k + nv, :, 7:] = X.trans

                    for j in range(nv):
                        # Pri inferenci vzamemo samo kvadre z verjetnostjo prisotnosti > 0.5:
                        v = vertices[j, X.prob[j].cpu() > 0.5].numpy()
                        write_predictions_mesh(v, f'i{i}_{k + j + 1}')

                    k += nv

                np.save('shape_parameters.npy', parameters)

            network.train()

if __name__ == "__main__":
    if shapenet_dir is None:
        train_set = load_shapes(grid_size, n_examples, n_points_per_shape)
    else:
        train_set = load_shapenet(shapenet_dir)

    batch_provider = BatchProvider(train_set)
    stats = Stats()

    # Za zdaj bomo vizualizirali kar primere iz učne množice. Tako je tudi v
    # referenčni implementaciji.
    for i, shape in enumerate(train_set[:n_examples_for_visualization]):
        write_volume_mesh(shape, i + 1)

    network = Network(PhaseParams(0))
    network.to(device)

    for phase in range(2):
        train(network, batch_provider, PhaseParams(phase), stats)

    stats.save_plots()
