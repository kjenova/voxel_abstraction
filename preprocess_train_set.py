import nibabel
import os
from skimage import measure
from scipy.ndimage.morphology import binary_erosion, ball

from loader.preprocess import preprocess

def get_components(radius = 3):
    volume = nibabel.load('mito-endolyso.nii.gz')
    volume = volume.get_fdata() == 1

    kernel = ball(radius)
    volume = binary_erosion(volume, kernel)

    labelled = measure.label(volume)
    props = measure.regionprops(labelled)
    props.sort(key = lambda x: x.area, reverse = True)

    print(f'# connected components: {len(props)} (radius = {radius})')

    return [p.filled_image for p in props]

def preprocess_train_set():
    components = get_components()
    data = [c, f'data/train/{i + 1}.mat' for i, c in enumerate(components[:2000])]
    preprocess(data)

if __name__ == "__main__":
    preprocess_train_set()
