import torch
import torch.nn.functional as F

class CuboidSurface:
    def __init__(self, n_samples):
        self.n_samples = n_samples
        # Ploskvi, ki imata i-to koordinato konstantno (bodisi 1 ali -1), vzorčimo skupaj, zato 3 namesto 6.
        self.n_samples_per_face = n_samples // 3

    # dims: B x P x 3
    def area(self, dims):
        width, height, depth = dims.chunk(3, dim = -1)

        wh = width * height
        hd = height * depth
        wd = width * depth

        area = 2 * (wh + hd + wd)
        return area.repeat(1, 1, self.n_samples)

    def get_importance_weights(self, dims):
        area = self.area(dims)

        # [x, y, z] => (3 / (1/x + 1/y + 1/z)) * [ 1/x, 1/y, 1/z ]
        # Relativna "majhnost" dane dimenzije
        weights = 3 * F.normalize(1 / dims, p = 1, dim = -1)

        weights = weights.unsqueeze(-1).repeat(1, 1, 1, self.n_samples_per_face)
        [b, p] = dims.size()[:2]
        weights = weights.reshape(b, p, self.n_samples)
        return (1 / self.n_samples_per_face) * (weights * area)

    def sample_points(self, dims):
        [b, p] = dims.size()[:2]
        s = self.n_samples
        fs = self.n_samples_per_face

        # Ali naj bo konstantna dimenzija ploskve pri dani točki enaka 1 ali -1?
        constant_dimension = torch.full((3, b, p, fs), 0.5, device = dims.device).bernoulli()
        constant_dimension = 2 * constant_dimension - 1

        samples = torch.empty(b, p, 3, fs, 3, device = dims.device).uniform_(-1, 1)
        for i in range(3):
            samples[:, :, i, :, i] = constant_dimension[i]

        samples = samples.reshape(b, p, s, 3)
        return samples * dims.unsqueeze(2).repeat(1, 1, s, 1)
