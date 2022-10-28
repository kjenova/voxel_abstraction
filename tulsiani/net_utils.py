def weights_init(m):
    mean = 0
    t = str(type(m))
    if 'BatchNorm1d' in t:
        mean = 1
    elif 'Linear' not in t:
        return
    m.weight.data.normal_(mean = mean, std = 0.02)
    if m.bias is not None:
        m.bias.data.zero_()

def _scale_weights(m, f):
    r = f[0] / f[1]
    m.weight.data *= r
    m.bias.data *= r

def scale_weights(network, params):
    # Ker v drugi fazi učenja zamenjamo 'dims_factor', mu tudi prilagodimo cel
    # 'scale' polno povezanega sloja za napovedovanje dimenzij.
    _scale_weights(network.primitives.dims.layer, params.dims_factors)

    # Enako za 'prob_factor'.
    _scale_weights(network.primitives.prob.layer, params.prob_factors)
