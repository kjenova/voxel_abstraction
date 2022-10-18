import torch
from scipy.io import loadmat
import glob
from os import path
from load_shapes import VolumeFaces

class PreprocessedShape:
    def __init__(self, mat):
        self.volume = None
        self.resized_volume = mat['Volume']
        self.volume_faces = None
        self.resized_volume_faces = VolumeFaces(self.resized_volume)
        self.shape_points = mat['surfaceSamples']
        self.closest_points = torch.from_numpy(mat['closestPoints'])

def load_preprocessed(directory, max_n_shapes = 1000000):
    shapes = []
    filenames = glob.glob(f'{directory}/*.mat')[:max_n_shapes]
    filenames.sort(key = lambda x: int(path.basename(x)[:-4]))
    return [PreprocessedShape(loadmat(f)) for f in filenames]
