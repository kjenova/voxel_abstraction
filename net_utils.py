def weights_init(m):
    mean = 1
    t = str(type(m))
    if 'BatchNorm1d' in t:
        mean = 0
    elif 'Linear' not in t:
        return
    m.weight.data.normal_(mean = mean, std = 0.02)
    m.bias.data.zero_()
