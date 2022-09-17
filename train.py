import torch
import torch.nn as nn
import numpy as np
import random
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

shapenet_dir = 'shapenet/chamferData/02691156' # None = mitohondriji
n_examples = 2000
train_set_ratio = .8
n_epochs = 30
visualization_each_n_epochs = 10
n_primitives_for_visualization = 5
batch_size = 32
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

class NetworkParams:
    def __init__(self, phase):
        self.phase = phase
        self.dims_factor = dims_factors[phase]
        self.prob_factor = prob_factors[phase]
        self.existence_penalty = existence_penalties[phase]

class Network(nn.Module):
    def __init__(self, params):
        super().__init__()

        # Število aktivacij iz prvega konvolucijskega sloja je 4.
        # Tretji (končni) konvolucijski sloj ima torej 4 * 2^{3 - 1} = 16
        # aktivacij. Tako je tudi v referenčni implementaciji v Python-u:
        # https://github.com/nileshkulkarni/volumetricPrimitivesPytorch/blob/367d2bc3f7d2ec122c4e2066c2ee2a922cf4e0c8/experiments/cadAutoEncCuboids/primSelTsdfChamfer.py#L172.
        # Vendar v originalni implementaciji v Lua je končno število
        # aktivacij 64. Tam je namreč 5 konvolucijskih slojev (tako kot v članku), ampak
        # ugibam, da se zadnja dva izrodita v polno povezani sloj:
        # https://github.com/shubhtuls/volumetricPrimitives/blob/3d994709925166d55aca32f1b6f448978836a05d/experiments/cadAutoEncCuboids/primSelTsdfChamfer.lua#L126
        # Pytorch tega ne dovoli, stari torch pa je izgleda dovolil
        # (drugačne privzete nastavitve za padding?). Dimenzije vhodnega
        # volumna (32^3) in arhitektura (po vsakem sloju se velikost po
        # vsaki dimenziji prepolovi z max pool slojem) sta namreč isti.
        self.encoder = VolumeEncoder(3, 4, 1)
        n = self.encoder.n_out_channels  # 16

        # Ker se mi zdi 16 aktivacij res premalo, dodam še dva nivoja,
        # ki sta funkcionalno enaka kot tista dva izrojena nivoja v Lua implementaciji:
        layers = []
        for _ in range(2):
            layers.append(nn.Linear(n, 2 * n))
            n *= 2
            layers.append(nn.BatchNorm1d(n))
            layers.append(nn.LeakyReLU(0.2, inplace = True))

        # To so pa predvideni polno povezani sloji:
        for _ in range(2):
            linear = nn.Linear(n, n)
            batch_norm = nn.BatchNorm1d(n)

            # Prejšnjim slojem, ki v originalni implementaciji sodijo v konvolucijski del,
            # pa pustimo, da se inicializirajo na privzeti način.
            weights_init(linear)
            weights_init(batch_norm)

            layers.append(linear)
            layers.append(batch_norm)
            layers.append(nn.LeakyReLU(0.2, inplace = True))

        self.fc_layers = nn.Sequential(*layers)
        self.primitives = PrimitivesPrediction(n, n_primitives, params)

    def forward(self, volume, params):
        x = self.encoder(volume.unsqueeze(1))
        x = x.reshape(volume.size(0), self.encoder.n_out_channels)
        x = self.fc_layers(x)
        return self.primitives(x, params)

class Batch:
    def __init__(self, shapes):
        self.volume = torch.stack([torch.from_numpy(s.resized_volume.astype(np.float32)) for s in shapes])
        self.shape_points = torch.stack([torch.from_numpy(s.shape_points) for s in shapes])
        self.closest_points = torch.stack([s.closest_points for s in shapes])

    def get():
        b = self.shape_points.size(0)
        sample_indices = torch.randint(0, n_points_per_shape, (b, n_samples_per_primitive))
        sample_indices += n_samples_per_primitive * torch.arange(0, b)
        sampled_points = self.shape_points[sample_indices]
        return (self.volume, sampled_points, self.closest_points)

def get_batches(shapes):
    batches = [Batch(shapes[i : i + batch_size]) for i in range(0, len(shapes), batch_size)]
    # V paketu morata biti vsaj dva elementa zaradi paketne normalizacije:
    return batches if batches[-1].volume.size(0) > 1 else batches[:-1]

def report(network, batches, epoch, params):
    sampler = CuboidSurface(n_samples_per_primitive)

    network.eval()
    with torch.no_grad():
        total_loss = .0
        i = 1
        for b in batches:
            volume = b.volume.to(device)
            closest_points = b.closest_points.to(device)

            P = network(volume, params)
            l = loss(volume, P, sampled_points, closest_points, sampler)

            total_loss += l.mean().item()

            n = b.volume.size(0)

            if epoch % visualization_each_n_epochs == 0:
                vertices = predictions_to_mesh(P).cpu().numpy()
                for j in range(n):
                    if i + j > n_primitives_for_visualization:
                        break

                    # Pri inferenci vzamemo samo kvadre z verjetnostjo prisotnosti > 0.5:
                    v = vertices[j, P.prob[j].cpu() > 0.5]
                    write_predictions_mesh(v, f'e{epoch}_{i + j}')

            i += n

        total_loss /= len(batches)
        print(f'loss: {total_loss}')

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

def train(network, train_set, validation_set, params):
    if params.phase == 1:
        scale_weights(network)

    validation_batches = get_batches(validation_set)

    optimizer = torch.optim.Adam(network.parameters(), lr = learning_rate)
    sampler = CuboidSurface(n_samples_per_primitive)
    reinforce_updater = ReinforceRewardUpdater(reinforce_baseline_momentum)
    epochs_offset = params.phase * n_epochs
    for e in range(epochs_offset + 1, epochs_offset + n_epochs + 1):
        print(f'epoch #{e}')

        random.shuffle(train_set)
        train_batches = get_batches(train_set)

        network.train()

        total_loss = .0
        total_prob = .0
        total_penalty = .0
        for b in train_batches:
            optimizer.zero_grad()

            volume = b.volume.to(device)
            sampled_points = b.sampled_points.to(device)
            closest_points = b.closest_points.to(device)

            P = network(volume, params)
            l = loss(volume, P, sampled_points, closest_points, sampler)

            total_prob += P.prob.mean().item()

            for i in range(n_primitives):
                # Pri nas se minimizira 'penalty', čeprav se pri REINFORCE tipično maksimizira 'reward'.
                # Edina razlika je v tem, da bi v primeru, da bi maksimizirali 'reward = - penalty', morali
                # potem še pri gradientu dodati minus.
                penalty = l + params.existence_penalty * P.exist[:, i] # torch.sum(P.exist[:, i])
                total_penalty += penalty.mean().item()
                P.log_prob[:, i] *= reinforce_updater.update(penalty)

            l = l.mean()
            total_loss += l.item()

            l += P.log_prob.mean() # sum()
            l.backward()
            optimizer.step()

        total_loss /= len(train_batches)
        total_prob /= len(train_batches)
        total_penalty /= n_primitives * len(train_batches)

        print(f'train loss: {total_loss}')
        print(f'avg prob: {total_prob}')
        print(f'avg penalty: {total_penalty}')

        report(network, validation_batches, e, params)

if shapenet_dir is None:
    examples = load_shapes(grid_size, n_examples, n_points_per_shape)
else:
    examples = load_shapenet(shapenet_dir)
train_set_size = int(train_set_ratio * len(examples))
train_set = examples[:train_set_size]
validation_set = examples[train_set_size:]

for i, shape in enumerate(validation_set[:n_primitives_for_visualization]):
    write_volume_mesh(shape, i + 1)

network = Network(NetworkParams(0))
network.to(device)

for phase in range(2):
    train(network, train_set, validation_set, NetworkParams(phase))
