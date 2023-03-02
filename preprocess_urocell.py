import nibabel
import os
from skimage import measure

from loader.preprocess import preprocess

test_data = [
    'fib1-0-0-0.nii.gz',
    'fib1-1-0-3.nii.gz',
    'fib1-3-2-1.nii.gz',
    'fib1-3-3-0.nii.gz',
    'fib1-4-3-0.nii.gz'
]

def preprocess_urocell(basedir, suffix = ''):
    shapes = []

    for file in test_data:
        volume = nibabel.load(f'{basedir}/{file}')
        volume = volume.get_fdata()

        volume_name = file.replace('.nii.gz', '')

        for label in [1, 2]:
            labelled = measure.label(volume == label)
            props = measure.regionprops(labelled)
            props.sort(key = lambda x: x.area, reverse = True)

            dir = f'data/urocell{suffix}/{volume_name}/{label}'
            os.makedirs(dir, exist_ok = True)

            for i, p in enumerate(props):
                if p.area < 2:
                    break

                shapes.append((p.filled_image, f'{dir}/{i + 1}.mat'))

    print(len(shapes))

    preprocess(shapes)

if __name__ == "__main__":
    preprocess_urocell('urocell/branched')
    preprocess_urocell('urocell/contacting', '_contacting')
