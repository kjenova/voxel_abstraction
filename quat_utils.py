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

if __name__ == "__main__":
    from pyquaternion import Quaternion
    import numpy as np

    def get_random_quat():
        q = Quaternion.random()
        quat = torch.Tensor(q.elements).float().view(1, 1, 4)
        return quat, q

    def test_quat_conjugate():
        quat, _ = get_random_quat()
        conj = quat_conjugate(quat)
        dot_p = quat[..., 0] * conj[..., 0] - (quat[..., 1:4] * conj[..., 1:4]).sum(-1)
        assert (1 - dot_p.mean()).abs().item() < 1e-4, 'Conjugate is incorrect'

    test_quat_conjugate()

