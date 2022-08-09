from scipy.io import loadmat
import glob
from load_shapes import VolumeFaces

class ShapeNetShape:
    def __init__(self, mat):
        self.volume = None
        self.resized_volume = mat['Volume']
        self.volume_faces = None
        self.resized_volume_faces = VolumeFaces(self.resized_volume)
        self.sampled_points = mat['surfaceSamples']
        self.closest_points = mat['closestPoints']

def load_shapenet(directory):
    shapes = []
    for filename in glob.iglob(f'{directory}/*.mat'):
        mat = loadmat(filename)
        shapes.append(ShapeNetShape(mat))
    return shapes

