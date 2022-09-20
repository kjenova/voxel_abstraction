import torch.nn as nn

class VolumeEncoder(nn.Module):
    def __init__(self, n_layers, first_layer_n_out_channels, n_input_channels):
        super().__init__()

        encoder = []

        c_in = n_input_channels
        c_out = first_layer_n_out_channels

        for _ in range(n_layers):
            encoder.append(nn.Conv3d(c_in, c_out, kernel_size = 3, padding = 'same', bias = False))
            encoder.append(nn.BatchNorm3d(c_out))
            encoder.append(nn.LeakyReLU(0.2, inplace = True))
            encoder.append(nn.MaxPool3d(kernel_size = 2))

            c_in = c_out
            c_out *= 2

        self.encoder = nn.Sequential(*encoder)
        self.n_out_channels = c_out // 2

    def forward(self, volume):
        return self.encoder(volume)
