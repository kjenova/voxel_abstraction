class ReinforceRewardUpdater:
    def __init__(self, baseline_momentum):
        self.m = baseline_momentum
        self.baseline = 0

    def update(self, rewards):
        result = rewards - self.baseline
        self.baseline = self.m * self.baseline + (1 - self.m) * rewards.mean()
        return result
