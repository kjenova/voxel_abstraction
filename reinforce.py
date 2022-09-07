class ReinforceRewardUpdater:
    def __init__(self, baseline_momentum):
        self.m = baseline_momentum
        self.baseline = .0

    def update(self, reward):
        # tenzor dolžine B minus Python float
        result = reward - self.baseline
        # .item() zato, da reward.mean() "odklopimo" od avtomatskega odvajanja
        # (dobimo navaden Python float).
        self.baseline = self.m * self.baseline + (1 - self.m) * reward.mean().item()
        return result
