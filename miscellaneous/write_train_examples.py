from loader.load_preprocessed import load_preprocessed
from loader.load_shapes import VolumeFaces
from graphics.write_mesh import write_helper

train_dir = 'data/chamferData/01'
train_set = load_preprocessed(train_dir)[:50]

for i, x in enumerate(train_set):
    write_helper(x.vertices, x.faces - 1, f'full_{i + 1}')
    v = VolumeFaces(x.resized_volume)
    write_helper(*v.get_mesh(), f'resized_{i + 1}')
