import torch
import torch.nn as nn
import torch.nn.functional as F

class ParameterPrediction(nn.Module):
    def __init__(self, n_input_channels, n_primitives, n_out_features, bias_init = None, nonlinearity = None):
        super().__init__()

        self.n_primitives = n_primitives
        self.nonlinearity = nonlinearity
        self.layer = nn.Linear(n_input_channels, n_primitives * n_out_features)

        if bias_init is not None:
            self.layer.bias.data = torch.Tensor(bias_init).view(1, n_out_features).repeat(1, n_primitives)

    def forward(self, feature):
        x = self.layer(feature)

        if self.nonlinearity is not None:
            x = self.nonlinearity(x)

        return x.view(feature.size(0), self.n_primitives, -1)

class PrimitivesPrediction(nn.Module):
    def __init__(self, n_input_channels, n_primitives):
        super().__init__()

        self.shape = ParameterPrediction(n_input_channels, n_primitives, 3, [-3, -3, -3], nn.Sigmoid())
        self.quat = ParameterPrediction(n_input_channels, n_primitives, 4, [1, 0, 0, 0])
        self.trans = ParameterPrediction(n_input_channels, n_primitives, 3, nonlinearity = nn.Tanh())
        # self.prob = ParameterPrediction(n_input_channels, n_primitives, 1, [0], nn.Sigmoid())

    def forward(self, feature):
        shape = self.shape(feature) * .5
        quat = F.normalize(self.quat(feature), dim = -1)
        trans = self.trans(feature)
        # prob = self.prob(feature)

        return shape, quat, trans

