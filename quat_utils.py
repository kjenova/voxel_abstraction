import torch

signed_inds = torch.LongTensor([[0, -1, -2, -3], [1, 0, 3, -2], [2, -3, 0, 1], [3, 2, -1, 0]])
inds = torch.abs(signed_inds)
# Hack to make 0 as positive sign: add 0.01 to all the values...
signs = torch.sign(signed_inds + 0.01)

def hamilton_product(q1, q2):
    q1_q2_prods = []
    for i in range(4):
        q2_permute = [q2[..., inds[i][j]] * signs[i][j] for j in range(4)]
        q2_permute = torch.stack(q2_permute, dim = -1)
        q1_q2_prods.append(torch.sum(q1 * q2_permute, dim = -1))

    return torch.stack(q1_q2_prods, dim = -1)

conj_multiplier = torch.Tensor([1, -1, -1, -1])

# Pri enotskem kvaternionu je konjugacija enaka inverzu.
# Kvaternione smo normalizirali ob napovedi.
def quat_conjugate(quat):
    return quat * conj_multiplier.to(quat.device)

# points: B x P x N x 3, quat/quat_conj: B x P x 4
def rotate(points, quat, conj):
    [b, p, n] = points.size()[:3]
    points = torch.cat([torch.zeros(b, p, n, 1, device = points.device), points], dim = -1)
    quat = quat.unsqueeze(2).repeat(1, 1, n, 1)
    conj = conj.unsqueeze(2).repeat(1, 1, n, 1)

    mult = hamilton_product(quat, points)
    mult = hamilton_product(mult, conj)
    return mult[..., 1:4]

def quat_rotate(points, quat):
    return rotate(points, quat, quat_conjugate(quat))

def quat_inverse_rotate(points, quat):
    return rotate(points, quat_conjugate(quat), quat)

from pyquaternion import Quaternion

def get_random_quat(b = 1, p = 1):
    q = Quaternion.random()
    quat = torch.Tensor(q.elements).float().view(1, 1, 4)
    return quat.repeat(b, p, 1), q

if __name__ == "__main__":
    import numpy as np

    def conjugate_test_helper(prod):
        dot_p = prod[..., 0] - prod[..., 1:4].sum(-1)
        return (1 - dot_p.mean()).abs().item() < 1e-4

    def test_quat_conjugate():
        quat, _ = get_random_quat(10, 3)
        conj = quat_conjugate(quat)
        assert conjugate_test_helper(quat * conj), 'Conjugate is incorrect'
        prod = hamilton_product(quat, conj)
        assert conjugate_test_helper(prod), 'Conjugate/Hamilton product is incorrect'

    def test_hamilton_product():
        quat1, q1 = get_random_quat()
        quat2, q2 = get_random_quat()
        quat_prod = hamilton_product(quat1, quat2).squeeze()
        q_prod = torch.Tensor((q1 * q2).elements).float()
        assert (quat_prod - q_prod).abs().mean() < 1e-4, 'Hamilton product is incorrect'

    def test_rotate():
        points = torch.empty(10, 3, 4, 3).uniform_(-1, 1)
        quat, _ = get_random_quat(10, 3)
        points_2 = quat_inverse_rotate(quat_rotate(points, quat), quat)
        assert (points_2 - points).abs().mean() < 1e-4, 'Rotation/Inverse rotation is incorrect'

        points = torch.empty(1, 1, 1, 3).uniform_(-1, 1)
        quat, q = get_random_quat()
        quat_rotated = quat_rotate(points, quat).squeeze()
        q_rotated = torch.Tensor(q.rotate(points.squeeze().numpy())).float()
        assert (quat_rotated - q_rotated).abs().mean() < 1e-4, 'Rotation is incorrect'

    test_quat_conjugate()
    test_hamilton_product()
    test_rotate()

