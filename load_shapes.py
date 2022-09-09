import os.path
import pickle
import nibabel
import torch
import numpy as np
from scipy.ndimage.morphology import binary_erosion
from skimage import measure
from skimage.transform import resize
from tqdm import tqdm

np.random.seed(0x5EED)

def split_padding(p):
    return (p // 2, p - p // 2)

def resize_volume(volume, grid_size):
    m = max(volume.shape)
    volume = np.pad(volume, [split_padding(m - d) for d in volume.shape])
    return resize(volume, [grid_size] * 3, mode = 'constant')

def get_face_data(volume):
    [w, h, d] = volume.shape

    max_n_faces = 6 * w * h * d
    voxel_centers = np.zeros((max_n_faces, 3))
    normals = np.zeros((max_n_faces, 3))
    tangents = np.zeros((max_n_faces, 2, 3))
    n_faces = 0
    for x in range(w):
        for y in range(h):
            for z in range(d):
                if not volume[x, y, z]:
                    continue

                for side in range(6):
                    orientation = 1 - 2 * (side % 2)
                    direction = side // 2

                    u = [x, y, z]
                    u[direction] += orientation

                    if u[0] < 0 or u[0] >= w or \
                       u[1] < 0 or u[1] >= h or \
                       u[2] < 0 or u[2] >= d or \
                       not volume[u[0], u[1], u[2]]:
                        voxel_centers[n_faces] = [x, y, z]
                        normals[n_faces, direction] = orientation
                        tangents[n_faces, 0, (direction + 1) % 3] = 0.5 * orientation
                        tangents[n_faces, 1, (direction + 2) % 3] = 0.5 * orientation
                        n_faces += 1

    return voxel_centers[:n_faces] + 0.5, normals[:n_faces], tangents[:n_faces]

class VolumeFaces:
    def __init__(self, volume):
        voxel_centers, normals, tangents = get_face_data(volume)
        self.n_faces = voxel_centers.shape[0]

        face_centers = voxel_centers + 0.5 * normals

        # Domena normaliziranih koordinat je [-0.5, 0.5].

        m = max(volume.shape)
        face_centers /= m
        face_centers[..., 0] -= 0.5 * volume.shape[0] / m
        face_centers[..., 1] -= 0.5 * volume.shape[1] / m
        face_centers[..., 2] -= 0.5 * volume.shape[2] / m

        tangents /= m

        self.face_centers = face_centers
        self.tangents = tangents

    def get_mesh(self):
        quads_points = np.zeros((self.n_faces, 4, 3))
        quads_points[:, 0] = self.face_centers - self.tangents[:, 0] - self.tangents[:, 1]
        quads_points[:, 1] = self.face_centers + self.tangents[:, 0] - self.tangents[:, 1]
        quads_points[:, 2] = self.face_centers + self.tangents[:, 0] + self.tangents[:, 1]
        quads_points[:, 3] = self.face_centers - self.tangents[:, 0] + self.tangents[:, 1]

        vertices = quads_points.reshape(-1, 3)
        faces = np.repeat([[[0, 1, 2], [2, 3, 0]]], self.n_faces, axis = 0)
        faces += np.arange(0, self.n_faces * 4, 4).reshape(-1, 1, 1)
        faces = faces.reshape(-1, 3)

        return vertices, faces

    def sample(self, n_points):
        sampled_faces = np.random.randint(self.n_faces, size = n_points)
        r = np.random.uniform(-1, 1, (n_points, 1, 2))
        u = np.matmul(r, self.tangents[sampled_faces]).squeeze(1)
        return self.face_centers[sampled_faces] + u

def centers_linspace(n):
    return torch.linspace(-.5 + .5 / n, .5 - .5 / n, n)

def voxel_center_points(size):
    x = centers_linspace(size[0])
    y = centers_linspace(size[1])
    z = centers_linspace(size[2])

    return torch.cartesian_prod(x, y, z)

def closest_points_grid(volume):
    centers = voxel_center_points(volume.shape)
    centers = centers.reshape(-1, 3)
    n_points = centers.size(0)

    distances = torch.cdist(centers, centers)
    v = np.tile(volume.reshape(-1), n_points).reshape(n_points, n_points)
    # Prazni voksli naj ne prispevajo:
    distances = np.where(v, distances, 1000.0)
    inds = distances.argmin(axis = -1)

    return centers[inds, :]

class Shape:
    def __init__(self, volume, grid_size, n_sampled_points):
        self.volume = volume
        self.resized_volume = resize_volume(volume, grid_size)
        self.volume_faces = VolumeFaces(volume)
        self.resized_volume_faces = VolumeFaces(self.resized_volume)
        self.sampled_points = self.resized_volume_faces.sample(n_sampled_points)
        self.closest_points = closest_points_grid(self.resized_volume)

def load_shapes(grid_size, n_components, n_sampled_points):
    shapes_path = 'shapes.pickle'
    if os.path.exists(shapes_path):
        with open(shapes_path, 'rb') as file:
            shapes = pickle.load(file)
    else:
        path = 'components.pickle'
        if os.path.exists(path):
            with open(path, 'rb') as file:
                components = pickle.load(file)
        else:
            volume = nibabel.load('mito-endolyso.nii.gz')
            volume = volume.get_fdata() == 1

            kernel = np.ones([8] * 3, np.bool_)
            volume = binary_erosion(volume, kernel)

            labelled = measure.label(volume)
            props = measure.regionprops(labelled)
            print(f'# connected components: {len(props)}')
            props.sort(key = lambda x: x.area, reverse = True)

            components = [p.filled_image for p in props]
            with open(path, 'wb') as file:
                pickle.dump(components, file)

        shapes = []
        for v in tqdm(components[:n_components]):
            shapes.append(Shape(v, grid_size, n_sampled_points))

        with open(shapes_path, 'wb') as file:
            pickle.dump(shapes, file)

    return shapes

if __name__ == "__main__":
    def test_centers_linspace(n = 10):
        c1 = centers_linspace(n)
        c2 = (torch.arange(0, n) + .5) / n - .5
        assert torch.allclose(c1, c2), 'center_linspace is incorrect'

    def test_voxel_centers(n = 10):
        v1 = voxel_center_points([2, 3, 4])
        v2 = []
        for x in [-.25, .25]:
            for y in [-.5 + .5/3, .0, .5 - .5/3]:
                for z in [-.5 + .5 * 1/4, -.5 + .5 * 3/4, -.5 + .5 * 5/4, -.5 + .5 *  7/4]:
                    v2.append([x, y, z])
        assert torch.allclose(v1, torch.Tensor(v2)), 'voxel_center_points is incorrect'

        v1 = torch.zeros(3, 4, 5, 3)
        for x in range(3):
            for y in range(4):
                for z in range(5):
                    volume = torch.zeros(3, 4, 5, dtype = torch.bool)
                    volume[x, y, z] = True
                    voxel_centers, _, _ = get_face_data(volume)
                    assert voxel_centers.shape == (6, 3)
                    assert voxel_centers.min(0)[0] == voxel_centers.max(0)[0]
                    v1[x, y, z] = torch.from_numpy(voxel_centers[0])

        v1 = v1.reshape(-1, 3)
        v1[:, 0] /= 3
        v1[:, 1] /= 4
        v1[:, 2] /= 5
        v1 = v1 - .5

        v2 = voxel_center_points([3, 4, 5])
        assert torch.allclose(v1, v2), 'voxel_center_points is incorrect'

    from generate_mesh import predictions_to_mesh
    from write_mesh import write_helper, write_predictions_mesh

    def mesh_test(grid_size = 5):
        for i in range(3):
            volume = torch.zeros([grid_size] * 3, dtype = torch.bool)
            inds = [grid_size // 2] * 3
            inds[i] = 0
            volume[inds[0], inds[1], inds[2]] = True
            f = VolumeFaces(volume)
            write_helper(*f.get_mesh(), f'volume_mesh_{i}')

            shape = torch.Tensor([.5 / grid_size] * 3).reshape(1, 1, 3)
            quat = torch.Tensor([1, 0, 0, 0]).reshape(1, 1, 4)
            trans = (torch.Tensor(inds).reshape(1, 1, 3) + .5) / grid_size - .5
            mesh = predictions_to_mesh(shape, quat, trans).numpy()[0]
            write_predictions_mesh(mesh, f'mesh_{i}')

    test_centers_linspace()
    test_voxel_centers()
    mesh_test()
