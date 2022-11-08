import torch.nn as nn
import numpy as np

from .volume_encoder import VolumeEncoder
from .net_utils import weights_init
from .primitives import PrimitivesPrediction

class TulsianiNetwork(nn.Module):
    def __init__(self, params):
        super().__init__()

        n_encoder_layers = np.around(np.log2(params.grid_size)).astype(int)
        add_coords = params.add_coordinates_to_encoder
        self.encoder = VolumeEncoder(n_encoder_layers, 8 if add_coords else 4, 1, params.use_batch_normalization_conv, add_coords)
        n = self.encoder.n_out_channels

        layers = []
        for _ in range(2):
            linear = nn.Linear(n, n, bias = not params.use_batch_normalization_linear)
            weights_init(linear)
            layers.append(linear)

            if params.use_batch_normalization_linear:
                batch_norm = nn.BatchNorm1d(n)
                weights_init(batch_norm)
                layers.append(batch_norm)

            layers.append(nn.LeakyReLU(0.2, inplace = True))

        self.fc_layers = nn.Sequential(*layers)
        self.primitives = PrimitivesPrediction(n, params)

    def forward(self, volume):
        x = self.encoder(volume.unsqueeze(1))
        x = x.reshape(volume.size(0), self.encoder.n_out_channels)
        x = self.fc_layers(x)
        return self.primitives(x)
