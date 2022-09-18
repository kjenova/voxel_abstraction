class ReinforceRewardUpdater:
    def __init__(self, baseline_momentum):
        self.m = baseline_momentum
        self.baseline = .0

    # V referenčni implementaciji se od 'reward' odšteje najnovejši 'baseline', t.j.
    # tisti, v katerem je 'reward' že upoštevan:
    # https://github.com/nileshkulkarni/volumetricPrimitivesPytorch/blob/367d2bc3f7d2ec122c4e2066c2ee2a922cf4e0c8/modules/primitives.py#L151
    # To po mojem razumevanju članka "Simple Statistical Gradient-Following Algorithms
    # for Connectionist Reinforcement Learning" ni korektno ("Suppose further that the
    # reinforcement baseline b_{ij} is conditionally independent of y_i..."). Zato sem
    # naredil tako, da se najprej odšteje prejšnji 'baseline'. Pri dovoljšnjem številu
    # iteracij sicer ne bi smelo biti razlike...
    def update(self, reward):
        # tenzor dolžine B minus Python float
        result = reward - self.baseline
        # .item() zato, da reward.mean() "odklopimo" od avtomatskega odvajanja
        # (dobimo navaden Python float).
        self.baseline = self.m * self.baseline + (1 - self.m) * reward.mean().item()
        return result
