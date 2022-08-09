from scipy.io import loadmat
import glob
import os
import numpy as np

class ShapeNetShape:
    def __init__(mat):
        self.resized_volume = mat['Volume']
        self.sampled_points = mat['surfaceSamples']
        self.closest_points = mat['closestPoints']
        print(self.resized_volume.shape)
        print(self.sampled_points.shape)
        print(self.closest_points.shape)

def load_shapenet(directory):
    shapes = []
    for filename in glob.iglob(f'{directory}/*.mat'):
        mat = loadmat(filename)
        shapes.append(ShapeNetShape(mat))
    return shapes

load_shapenet('...')
