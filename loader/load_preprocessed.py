from scipy.io import loadmat
import glob
from os import path

class PreprocessedShape:
    def __init__(self, mat):
        self.resized_volume = mat['Volume']
        self.shape_points = mat['surfaceSamples']
        self.closest_points = mat['closestPoints']
        self.vertices = mat['vertices']
        self.faces = mat['faces']

        if 'normals' in mat:
            self.normals = mat['normals']

def load_preprocessed(directory, max_n_shapes = 1000000):
    shapes = []
    filenames = glob.glob(f'{directory}/*.mat')[:max_n_shapes]
    filenames.sort(key = lambda x: int(path.basename(x)[:-4]))
    return [PreprocessedShape(loadmat(f)) for f in filenames]
