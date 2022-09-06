import torch
import torch.nn.functional as F

class CuboidSurface:
    def __init__(self, n_samples):
        self.n_samples = n_samples
        # Ploskvi, ki imata i-to koordinato konstantno (bodisi 1 ali -1), vzorčimo skupaj, zato 3 namesto 6.
        self.n_samples_per_face = n_samples // 3

    # dims: B x P x 3
    def get_importance_weights(self, dims):
        width, height, depth = dims.chunk(3, dim = -1)

        hd = height * depth # vse razen prve dimenzije
        wd = width * depth # vse razen druge dimenzije
        wh = width * height # vse razen tretje dimenzije

        face_areas = torch.stack((hd, wd, wh), dim = -1)
        face_areas = face_areas.repeat(1, 1, self.n_samples_per_face, 1)

        # face_areas[..., i, j] ustreza i-temu vzorcu iz j-tega para ploskev, torej je sorazmeren ploščini
        # j-tega para ploskev.

        # Lahko bi množili z 2, saj vzorčimo iz dveh ploskev skupaj; s 4, saj so dims v bistvu polovice dimenzij,
        # in delili s self.n_samples_per_face, ampak je vseeno, ker se bodo uteži normalizirale.

        [b, p] = dims.size()[:2]
        return face_areas.reshape(b, p, self.n_samples)

    def sample_points(self, dims):
        [b, p] = dims.size()[:2]
        s = self.n_samples
        fs = self.n_samples_per_face

        # Ali naj ima konstantna dimenzija ploskve pri dani točki predznak + ali -?
        constant_dimension = torch.full((3, b, p, fs), 0.5, device = dims.device).bernoulli()
        constant_dimension = 2 * constant_dimension - 1

        samples = torch.empty(b, p, fs, 3, 3, device = dims.device).uniform_(-1, 1)
        for i in range(3):
            samples[..., i, i] = constant_dimension[i]

        # samples[..., i, j, :] = i-ti vzorec iz j-tega para ploskev.

        samples = samples.reshape(b, p, s, 3)
        return samples * dims.unsqueeze(2).repeat(1, 1, s, 1)

if __name__ == "__main__":
    def test_sampling(n_samples):
        fs = n_samples // 3

        dims = torch.Tensor([[1, 2, 3], [2, 4, 6], [3, 6, 9]]).reshape(1, 3, 3)

        surface = CuboidSurface(n_samples)
        w = surface.get_importance_weights(dims)

        true_w = torch.Tensor([[2 * 3, 1 * 3, 1 * 2], [4 * 6, 2 * 6, 2 * 4], [6 * 9, 3 * 9, 3 * 6]])
        true_w = true_w.reshape(1, 3, 3, 1).repeat(1, 1, 1, fs)
        true_w = true_w.transpose(-1, -2)
        true_w = true_w.reshape(1, 3, n_samples)

        assert torch.allclose(w, true_w), 'get_importance_weights is incorrect'

    test_sampling(6)
