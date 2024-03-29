import nibabel
from .load_preprocessed import load_preprocessed

test_data = [
    # ('fib1-0-0-0.nii.gz', [[], []]), # Tegale sem pozabil prej, tako da ni trenirano na njem. Mislim, da nima nobenih zanimivih mitohondrijev...
    ('fib1-1-0-3.nii.gz', [[0], [1, 2]]),
    ('fib1-3-2-1.nii.gz', [[0, 1], [18]]),
    ('fib1-3-3-0.nii.gz', [[0, 3], [0]]),
    ('fib1-4-3-0.nii.gz', [[0], []])
]

# Samo tale funkcija se uporablja:
def load_urocell_preprocessed(basedir, contacting = False):
    validation = []
    test = []

    for subvolume_dir, indices_by_label in test_data:
        for label, indices in enumerate(indices_by_label):
            loaded = load_preprocessed(basedir + '/' + subvolume_dir.replace(".nii.gz", "") + '/' + str(label + 1))
            
            for i in range(len(loaded)):
                loaded[i].branched = label == int(contacting)

                if i in indices_by_label[label]:
                    test.append(loaded[i])
                else:
                    validation.append(loaded[i])

    return validation, test

"""
    from .load_shapes import resize_volume, VolumeFaces, closest_points_grid
    from skimage import measure

    class UroCellShape:
        def __init__(self, volume, grid_size, n_points_per_shape):
            self.volume = volume
            self.resized_volume = resize_volume(volume, grid_size)
            self.volume_faces = VolumeFaces(volume)
            # self.shape_points = self.volume_faces.sample(n_points_per_shape)
            # self.closest_points_grid = closest_points_grid(self.resized_volume, self.volume_faces)

    def load_urocell(basedir, grid_size, n_points_per_shape = 10000, discard_validation = False):
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
                        test.append(UroCellShape(component, grid_size, n_points_per_shape))
                    elif not discard_validation:
                        validation.append(UroCellShape(component, grid_size, n_points_per_shape))

        return validation, test
"""
