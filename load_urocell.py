import nibabel
from skimage import measure

test_data = [
    ('fib1-1-0-3.nii.gz', [[0], [1, 2]]),
    ('fib1-3-2-1.nii.gz', [[0, 1], [18]]),
    ('fib1-3-3-0.nii.gz', [[0, 3], [0]]),
    ('fib1-4-3-0.nii.gz', [[0], []])
]

def load_validation_and_test():
    validation = []
    test = []

    for file, indices_by_label in test_data:
        volume = nibabel.load(f'{basedir}/{file}')
        volume = volume.get_fdata()

        for label, indices in enumerate(indices_by_label):
            v = volume == (label + 1)
            labelled = measure.label(volume)
            props = measure.regionprops(labelled)
            props.sort(key = lambda x: x.area, reverse = True)

            for j, component in enumerate(p.filled_image for p in props):
                if j in indices:
                    test.append(component)
                else:
                    validation.append(component)

    return validation, test
