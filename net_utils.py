def weights_init(m):
    mean = 0 if 'BatchNorm1d' in str(type(m)) else 1
    m.weights.data.normal_(mean = mean, std = 0.02)
    m.bias.data.zero_()
