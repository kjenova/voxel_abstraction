import torch
import numpy as np

def sample_points(all_points, n_samples_per_shape):
    [b, n] = all_points.size()[:2]
    i = torch.randint(0, n, (b, n_samples_per_shape), device = all_points.device)
    i += n * torch.arange(0, b, device = all_points.device).reshape(-1, 1)
    return all_points.reshape(-1, 3)[i]

class BatchProvider:
    def __init__(self, shapes, params, test = False, store_on_gpu = True):
        self.repeat_batch_n_iterations = params.repeat_batch_n_iterations
        self.n_samples_per_shape = params.n_samples_per_shape
        self.batch_size = params.batch_size

        self.batch_device = params.device
        self.store_device = self.batch_device if store_on_gpu else torch.device('cpu')

        self.volume = torch.stack(
            [torch.from_numpy(s.resized_volume.astype(np.float32)) for s in shapes]
        ).to(self.store_device)
        self.n = self.volume.size(0)

        if not test:
            self.shape_points = torch.stack(
                [torch.from_numpy(s.shape_points) for s in shapes]
            ).to(self.store_device)
            self.closest_points = torch.stack(
                [torch.from_numpy(s.closest_points) for s in shapes]
            ).to(self.store_device)
            self.iteration = 0

    def load_batch(self):
        indices = torch.randint(0, self.n, (min(self.batch_size, self.n),), device = self.store_device)
        # Če je store_on_gpu = False, na GPU pošljemo samo batche, drugače pa so že celotni tenzorji tam.
        self.loaded_volume = self.volume[indices].to(self.batch_device)
        self.loaded_shape_points = self.shape_points[indices].to(self.batch_device)
        self.loaded_closest_points = self.closest_points[indices].to(self.batch_device)

    def get(self):
        if self.iteration % self.repeat_batch_n_iterations == 0:
            self.load_batch()

        self.iteration += 1

        sampled_points = sample_points(self.loaded_shape_points, self.n_samples_per_shape)
        return (self.loaded_volume, sampled_points, self.loaded_closest_points)

    def get_all_batches(self):
        for i in range(0, self.n, self.batch_size):
            m = min(self.batch_size, self.n - i)
            sampled_points = sample_points(
                self.shape_points[i : i + m],
                self.n_samples_per_shape
            ).to(self.batch_device)
            yield (
                self.volume[i : i + m].to(self.batch_device),
                sampled_points,
                self.closest_points[i : i + m].to(self.batch_device)
            )
