import numpy as np
import matplotlib.pyplot as plt

class TulsianiStats:
    def __init__(self, params):
        self.n = sum(params.n_iterations)
        self.cov = np.zeros(self.n)
        self.cons = np.zeros(self.n)
        self.prob_means = np.zeros(self.n)
        self.penalty_means = np.zeros(self.n)

        self.use_paschalidou_loss = params.use_paschalidou_loss
        self.parsimony = np.zeros(self.n)

        self.m = self.n // params.save_iteration
        self.validation_loss = np.zeros(self.m)

    def save_plots(self, directory):
        x = np.arange(1, self.n + 1)

        plt.figure(figsize = (20, 5))
        plt.plot(x, self.cov)
        plt.xlabel('iteration')
        plt.ylabel('coverage')
        plt.savefig(f'{directory}/coverage.png')

        plt.clf()
        plt.figure(figsize = (20, 5))
        plt.plot(x, self.cons)
        plt.xlabel('iteration')
        plt.ylabel('consistency')
        plt.savefig(f'{directory}/consistency.png')

        if self.use_paschalidou_loss:
            plt.clf()
            plt.figure(figsize = (20, 5))
            plt.plot(x, self.parsimony)
            plt.xlabel('iteration')
            plt.ylabel('parsimony')
            plt.savefig(f'{directory}/parsimony.png')

        plt.figure(figsize = (20, 5))
        plt.plot(x[save_iteration:], (self.cov + self.cons + self.parsimony)[save_iteration:], label = 'train')
        v = np.arange(save_iteration, self.n + 1, save_iteration)
        plt.plot(v, self.validation_loss, label = 'validation')
        plt.xlabel('iteration')
        plt.ylabel('loss')
        plt.legend()
        plt.savefig(f'{directory}/loss.png')

        plt.clf()
        plt.figure(figsize = (20, 5))
        plt.plot(x, self.prob_means)
        plt.xlabel('iteration')
        plt.ylabel('mean probability')
        plt.savefig(f'{directory}/probability.png')

        if not self.use_paschalidou_loss:
            plt.clf()
            plt.figure(figsize = (20, 5))
            plt.plot(x, self.penalty_means)
            plt.xlabel('iteration')
            plt.ylabel('penalty')
            plt.savefig(f'{directory}/penalty.png')
