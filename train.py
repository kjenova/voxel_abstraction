import torch
import torch.nn as nn
import numpy as np
import random
from volume_encoder import VolumeEncoder
from primitives import PrimitivesPrediction
from losses import loss
from load_shapes import load_shapes, Shape
from generate_mesh import predictions_to_mesh
from write_mesh import write_volume_mesh, write_predictions_mesh

random.seed(0x5EED)

n_examples = 4 # 2000
train_set_ratio = 0.5 # 0.8
n_epochs = 1
visualization_each_n_epochs = 1
batch_size = 2 # 32
n_primitives = 4
grid_size = 32
n_samples_per_shape = 10000
n_samples_per_primitive = 150
learning_rate = 1e-4

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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

    def forward(self, volume):
        x = self.encoder(volume.unsqueeze(1))
        x = x.reshape(volume.size(0), self.encoder.n_out_channels)
        x = self.fc_layers(x)
        return self.primitives(x)

class Batch:
    def __init__(self, shapes):
        self.volume = torch.stack([torch.from_numpy(s.resized_volume.astype(np.float32)) for s in shapes])
        self.sampled_points = torch.stack([torch.from_numpy(s.sampled_points) for s in shapes])
        self.closest_points = torch.stack([s.closest_points for s in shapes])

def get_batches(shapes):
    batches = []
    for i in range(0, len(shapes), batch_size):
        batches.append(Batch(shapes[i : i + batch_size]))
    return batches

def report(network, batches, epoch):
    network.eval()
    with torch.no_grad():
        total_loss = .0
        i = 1
        for b in batches:
            volume = b.volume.to(device)
            sampled_points = b.sampled_points.to(device)
            closest_points = b.closest_points.to(device)

            p = network(volume)
            l = loss(volume, *p, sampled_points, closest_points, n_samples_per_primitive)

            total_loss += l.item()

            n = b.volume.size(0)

            if epoch % visualization_each_n_epochs == 0:
                vertices = predictions_to_mesh(*p)
                for j in range(n):
                    write_predictions_mesh(vertices[j].cpu().numpy(), i + j)

            i += n

        total_loss /= len(batches)
        print(f'loss: {total_loss}')

def train(network, train_set, validation_set):
    optimizer = torch.optim.Adam(network.parameters(), lr = learning_rate)

    train_batches = get_batches(train_set)
    random.shuffle(train_batches)

    validation_batches = get_batches(validation_set)

    for e in range(1, n_epochs + 1):
        print(f'epoch #{e}')

        network.train()

        total_loss = .0
        for b in train_batches:
            optimizer.zero_grad()

            volume = b.volume.to(device)
            sampled_points = b.sampled_points.to(device)
            closest_points = b.closest_points.to(device)

            p = network(volume)
            l = loss(volume, *p, sampled_points, closest_points, n_samples_per_primitive)

            l.backward()
            optimizer.step()

            total_loss += l.item()

        total_loss /= len(train_batches)
        print(f'train loss: {total_loss}')

        report(network, validation_batches, e)

examples = load_shapes(grid_size, n_examples, n_samples_per_shape)
train_set_size = int(train_set_ratio * len(examples))
train_set = examples[:train_set_size]
validation_set = examples[train_set_size:]

for i, shape in enumerate(validation_set):
    write_volume_mesh(shape, i + 1)

network = Network()
network.to(device)
train(network, train_set, validation_set)
