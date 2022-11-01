import torch
import numpy as np

class BatchProviderParams:
    def __init__(self, device, repeat_batch_n_iterations = 1, n_samples_per_shape = 1000, batch_size = 32):
        self.device = device
        self.repeat_batch_n_iterations = repeat_batch_n_iterations
        self.n_samples_per_shape = n_samples_per_shape
        self.batch_size = batch_size

class BatchProvider:
    def __init__(
        self,
        shapes,
        params,
        test = False,
        store_on_gpu = True,
        include_normals = False,
        uses_point_sampling = True
    ):
        self.repeat_batch_n_iterations = params.repeat_batch_n_iterations
        self.n_samples_per_shape = params.n_samples_per_shape
        self.batch_size = params.batch_size

        self.include_normals = include_normals
        self.uses_point_sampling = uses_point_sampling

        self.batch_device = params.device
        self.store_device = self.batch_device if store_on_gpu else torch.device('cpu')

        self.volume = torch.stack(
            [torch.from_numpy(s.resized_volume.astype(np.float32)) for s in shapes]
        ).to(self.store_device)
        self.n = self.volume.size(0)

        if not test:
            self.shape_points = torch.stack(
                [torch.from_numpy(s.shape_points.astype(np.float32)) for s in shapes]
            ).to(self.store_device)
            self.closest_points = torch.stack(
                [torch.from_numpy(s.closest_points.astype(np.float32)) for s in shapes]
            ).to(self.store_device)

            if include_normals:
                self.normals = torch.stack(
                    [torch.from_numpy(s.normals.astype(np.float32)) for s in shapes]
                ).to(self.store_device)

            self.iteration = 0

    def load_batch(self):
        indices = torch.randint(0, self.n, (min(self.batch_size, self.n),), device = self.store_device)
        # Če je store_on_gpu = False, na GPU pošljemo samo batche, drugače pa so že celotni tenzorji tam.
        self.loaded_volume = self.volume[indices].to(self.batch_device)
        self.loaded_shape_points = self.shape_points[indices].to(self.batch_device)
        self.loaded_closest_points = self.closest_points[indices].to(self.batch_device)

    def sample_points(self, all_points, all_normals = None):
        if not self.uses_point_sampling:
            return all_points, all_normals

        [b, n] = all_points.size()[:2]
        i = torch.randint(0, n, (b, self.n_samples_per_shape), device = all_points.device)
        i += n * torch.arange(0, b, device = all_points.device).reshape(-1, 1)
        return all_points.reshape(-1, 3)[i], all_normals.reshape(-1, 3)[i] if all_normals != None else None

    def get(self):
        if self.iteration % self.repeat_batch_n_iterations == 0:
            self.load_batch()

        self.iteration += 1

        sampled_points, _ = self.sample_points(self.loaded_shape_points)
        return (self.loaded_volume, sampled_points, self.loaded_closest_points)

    def get_all_batches(self, shuffle = False):
        indices = torch.randperm(self.n) if shuffle else torch.arange(self.n)

        for i in range(0, self.n, self.batch_size):
            m = min(self.batch_size, self.n - i)
            ind = indices[i : i + m]

            sampled_points, sampled_normals = self.sample_points(
                self.shape_points[ind],
                self.normals[ind] if self.include_normals else None
            )

            result = (
                self.volume[ind].to(self.batch_device),
                sampled_points.to(self.batch_device),
                self.closest_points[ind].to(self.batch_device)
            )

            if self.include_normals:
                result += sampled_normals.to(self.batch_device),

            yield result
