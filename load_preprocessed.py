import torch
from scipy.io import loadmat
import glob
from load_shapes import VolumeFaces

class ShapeNetShape:
    def __init__(self, mat):
        self.volume = None
        self.resized_volume = mat['Volume']
        self.volume_faces = None
        self.resized_volume_faces = VolumeFaces(self.resized_volume)
        self.shape_points = mat['surfaceSamples']
        self.closest_points = torch.from_numpy(mat['closestPoints'])

def load_preprocessed(directory, max_n_shapes = 1000000):
    shapes = []
    for filename in glob.iglob(f'{directory}/*.mat'):
        mat = loadmat(filename)
        shapes.append(ShapeNetShape(mat))

        if len(shapes) >= max_n_shapes:
            break
    return shapes
