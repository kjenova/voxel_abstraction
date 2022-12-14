import numpy as np
import trimesh
from trimesh import Trimesh

from .colors import colors

def write_helper(vertices, faces, filename):
    # xyz => xzy
    vertices_xzy = vertices.copy()
    vertices_xzy[..., 1] = vertices[..., 2]
    vertices_xzy[..., 2] = vertices[..., 1]
    triangles = Trimesh(vertices_xzy, faces)
    triangles.export(f'results/{filename}.stl')

def write_volume_mesh(shape, name):
    if shape.volume_faces is not None:
        write_helper(*shape.volume_faces.get_mesh(), f'{name}_volume')
    write_helper(*shape.resized_volume_faces.get_mesh(), f'{name}_resized')

cuboid_faces = np.asarray([ \
    [2, 3, 1, 0], \
    [5, 4, 0, 1], \
    [7, 5, 1, 3], \
    [3, 2, 6, 7], \
    [4, 6, 2, 0], \
    [4, 5, 7, 6] \
])

def write_predictions_mesh(vertices, name, directory):
    mtl_lines = []
    p = vertices.shape[0]

    for i in range(min(p, colors.shape[0])):
        c = colors[i]
        mtl_lines.append(f'newmtl m{i}\nKd {c[0]} {c[1]} {c[2]}\nKa 0 0 0\n')

    filename = f'{name}_predictions.mtl'
    mtl_lines.append(f'mtllib {filename}\n')

    for i in range(p):
        mtl_lines.append(f'usemtl m{i % colors.shape[0]}\n')

        for j in range(8):
            v = vertices[i, j]
            mtl_lines.append(f'v {v[0]} {v[1]} {- v[2]}\n')

        for j in range(6):
            f = cuboid_faces[j] + 8 * i + 1
            mtl_lines.append(f'f {f[0]} {f[1]} {f[2]} {f[3]}\n')

    with open(f'{directory}/{filename}', 'w') as f:
        f.writelines(mtl_lines)

if __name__ == "__main__":
    def test():
        vertices = np.zeros((8, 3))

        vertices[[0, 1, 4, 5], 0] = 1
        vertices[[2, 3, 6, 7], 0] = -1

        vertices[::2, 1] = 5
        vertices[1::2, 1] = -5

        vertices[:4, 2] = 10
        vertices[4:, 2] = -10

        write_predictions_mesh(vertices.reshape(1, 8, 3), 'test')

    test()
