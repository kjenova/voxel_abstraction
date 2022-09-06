import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.bernoulli import Bernoulli

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

class Primitives:
    def __init__(self, dims, quat, trans, exist, log_prob):
        self.dims = dims
        self.quat = quat
        self.trans = trans
        self.exist = exist
        self.log_prob = log_prob

class PrimitivesPrediction(nn.Module):
    def __init__(self, n_input_channels, n_primitives):
        super().__init__()

        self.dims = ParameterPrediction(n_input_channels, n_primitives, 3, [-3, -3, -3], nn.Sigmoid())
        self.quat = ParameterPrediction(n_input_channels, n_primitives, 4, [1, 0, 0, 0])
        self.trans = ParameterPrediction(n_input_channels, n_primitives, 3, nonlinearity = nn.Tanh())
        self.prob = ParameterPrediction(n_input_channels, n_primitives, 1, [0], nn.Sigmoid())

    def forward(self, feature):
        # Dimenzije že kar na tem mestu delimo z 2, kajti drugače bi jih morali na treh različnih mestih.
        # (Pa tudi v kodi, po kateri se zgledujemo, je tako: https://github.com/nileshkulkarni/volumetricPrimitivesPytorch)
        dims = self.dims(feature) * .5
        quat = F.normalize(self.quat(feature), dim = -1)
        # Ker rotacijo opravimo pred translacijo, moramo rotacijo upoštevati pri določanju maksimalne vrednosti translacije.
        # V ekstremnem primeru se kvader po i-ti dimenziji razteza od - sqrt(3) / 2 do sqrt(3) / 2. Da ga spravimo izven
        # območja našega koordinatnega sistema ([- 1 / 2, 1 / 2]^3), ga moramo po i-ti dimenziji prestaviti vsaj za delta =
        # sqrt(3) / 2 + 1 / 2 ~= 1,366. Po tej logiki bi morali 'trans' pomnožiti z delta, ampak za zdaj pustimo tako, kot
        # je v kodi od nileshkulkarni.
        trans = self.trans(feature)
        prob = self.prob(feature).squeeze()

        distr = Bernoulli(prob)
        exist = distr.sample()
        log_prob = distr.log_prob(exist)

        return Primitives(dims, quat, trans, exist, log_prob)
