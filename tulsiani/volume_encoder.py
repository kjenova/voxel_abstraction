import torch
import torch.nn as nn

def voxel_center_points(n, device):
    c = torch.linspace(-.5 + .5 / n, .5 - .5 / n, n, device = device)
    return torch.cartesian_prod(c, c, c)

center_points = None

class VolumeEncoder(nn.Module):
    def __init__(self, n_layers, first_layer_n_out_channels, n_input_channels, use_batch_normalization, add_coordinates):
        super().__init__()

        self.add_coordinates = add_coordinates

        encoder = []

        c_in = n_input_channels + (3 if add_coordinates else 0)
        c_out = first_layer_n_out_channels

        for _ in range(n_layers):
            encoder.append(nn.Conv3d(c_in, c_out, kernel_size = 3, padding = 'same', bias = not use_batch_normalization))
            if use_batch_normalization:
                encoder.append(nn.BatchNorm3d(c_out))
            encoder.append(nn.LeakyReLU(0.2, inplace = True))
            encoder.append(nn.MaxPool3d(kernel_size = 2))

            c_in = c_out
            c_out *= 2

        self.encoder = nn.Sequential(*encoder)
        self.n_out_channels = c_out // 2

    def forward(self, volume):
        global center_points

        if self.add_coordinates:
            n = volume.size(2)

            if center_points is None:
                center_points = voxel_center_points(n, volume.device).transpose(0, 1)

            c = center_points.reshape(1, 3, n, n, n).repeat(volume.size(0), 1, 1, 1, 1)
            volume = torch.cat((volume, c), dim = 1)

        return self.encoder(volume)
