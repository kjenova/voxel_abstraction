def weights_init(m):
    mean = 0
    t = str(type(m))
    if 'BatchNorm1d' in t:
        mean = 1
    elif 'Linear' not in t:
        return
    m.weight.data.normal_(mean = mean, std = 0.02)
    m.bias.data.zero_()
