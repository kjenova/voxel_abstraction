import torch
import torch.nn as nn
import numpy as np
import random
from volume_encoder import VolumeEncoder
from primitives import PrimitivesPrediction
from cuboid import CuboidSurface
from losses import loss
from reinforce import ReinforceRewardUpdater
from load_shapes import load_shapes, Shape
from load_shapenet import load_shapenet, ShapeNetShape
from generate_mesh import predictions_to_mesh
from write_mesh import write_volume_mesh, write_predictions_mesh

random.seed(0x5EED)

shapenet_dir = 'shapenet/chamferData/02691156' # None
n_examples = 2000
train_set_ratio = .8
n_epochs = 20
visualization_each_n_epochs = 20
batch_size = 32
n_primitives = 20
grid_size = 32
n_samples_per_shape = 1000
n_samples_per_primitive = 150
learning_rate = 1e-3
reinforce_baseline_momentum = .9
existence_penalty = 8e-5

# cuda:1 = Titan X
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

class Network(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = VolumeEncoder(3, 4, 1)
        n = self.encoder.n_out_channels

        layers = []
        for _ in range(2):
            layers.append(nn.Linear(n, n))
            layers.append(nn.BatchNorm1d(n))
            layers.append(nn.LeakyReLU(0.2, inplace = True))

        self.fc_layers = nn.Sequential(*layers)
        self.primitives = PrimitivesPrediction(n, n_primitives)

    def forward(self, volume, predict_existence):
        x = self.encoder(volume.unsqueeze(1))
        x = x.reshape(volume.size(0), self.encoder.n_out_channels)
        x = self.fc_layers(x)
        return self.primitives(x, predict_existence)

class Batch:
    def __init__(self, shapes):
        self.volume = torch.stack([torch.from_numpy(s.resized_volume.astype(np.float32)) for s in shapes])
        self.sampled_points = torch.stack([torch.from_numpy(s.sampled_points) for s in shapes])
        self.closest_points = torch.stack([s.closest_points for s in shapes])

def get_batches(shapes):
    return [Batch(shapes[i : i + batch_size]) for i in range(0, len(shapes), batch_size)]

def report(network, batches, epoch, predict_existence):
    sampler = CuboidSurface(n_samples_per_primitive)

    network.eval()
    with torch.no_grad():
        total_loss = .0
        i = 1
        for b in batches:
            volume = b.volume.to(device)
            sampled_points = b.sampled_points.to(device)
            closest_points = b.closest_points.to(device)

            P = network(volume, predict_existence)
            l = loss(volume, P, sampled_points, closest_points, sampler)

            total_loss += l.mean().item()

            n = b.volume.size(0)

            if epoch % visualization_each_n_epochs == 0:
                vertices = predictions_to_mesh(P).cpu().numpy()
                for j in range(n):
                    # Pri inferenci vzamemo samo kvadre z verjetnostjo prisotnosti > 0.5:
                    v = vertices[j, P.prob[j].cpu() > 0.5]
                    write_predictions_mesh(v, i + j)

            i += n

        total_loss /= len(batches)
        print(f'loss: {total_loss}')

def train(network, train_set, validation_set, train_existence):
    random.shuffle(train_set)
    train_batches = get_batches(train_set)

    validation_batches = get_batches(validation_set)

    optimizer = torch.optim.Adam(network.parameters(), lr = learning_rate)
    sampler = CuboidSurface(n_samples_per_primitive)
    reward_updater = ReinforceRewardUpdater(reinforce_baseline_momentum)
    epochs_offset = n_epochs if train_existence else 0
    for e in range(epochs_offset + 1, epochs_offset + n_epochs + 1):
        print(f'epoch #{e}')

        network.train()

        total_loss = .0
        for b in train_batches:
            optimizer.zero_grad()

            volume = b.volume.to(device)
            sampled_points = b.sampled_points.to(device)
            closest_points = b.closest_points.to(device)

            P = network(volume, train_existence)
            l = loss(volume, P, sampled_points, closest_points, sampler)

            if train_existence:
                p = P.exist.size(1)
                for i in range(p):
                    # Pri nas se 'reward' minimizira, čeprav se ga pri REINFORCE tipično maksimizira.
                    # Edina razlika je v tem, da bi v primeru, da bi nastavili 'reward *= -1', morali
                    # potem še pri gradientu dodati minus.
                    reward = l + existence_penalty * P.exist[:, i] # oba člena sta tenzorja dolžine B
                    reinforce_reward = reward_updater.update(reward)
                    P.log_prob[:, i] *= reinforce_reward

            l = l.mean()
            total_loss += l.item()

            if train_existence:
                l += P.log_prob.mean()

            l.backward()
            optimizer.step()

        total_loss /= len(train_batches)
        print(f'train loss: {total_loss}')

        report(network, validation_batches, e, train_existence)

if shapenet_dir is None:
    examples = load_shapes(grid_size, n_examples, n_samples_per_shape)
else:
    examples = load_shapenet(shapenet_dir)
train_set_size = int(train_set_ratio * len(examples))
train_set = examples[:train_set_size]
validation_set = examples[train_set_size:]

for i, shape in enumerate(validation_set):
    write_volume_mesh(shape, i + 1)

network = Network()
network.to(device)

train(network, train_set, validation_set, False)
train(network, train_set, validation_set, True)
