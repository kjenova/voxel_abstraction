import numpy as np
import matplotlib.pyplot as plt

class TulsianiStats:
    def __init__(self, params):
        self.n = params.total_iterations
        self.cov = np.zeros(self.n)
        self.cons = np.zeros(self.n)
        self.prob_means = np.zeros(self.n)
        self.dim_means = np.zeros(self.n)
        self.trans_stds = np.zeros(self.n)
        self.penalty_means = np.zeros(self.n)

        self.use_paschalidou_loss = params.use_paschalidou_loss
        self.parsimony = np.zeros(self.n)

        self.save_iteration = params.save_iteration
        self.m = self.n // self.save_iteration
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
        plt.plot(x[self.save_iteration:], (self.cov + self.cons + self.parsimony)[self.save_iteration:], label = 'train')
        v = np.arange(self.save_iteration, self.n + 1, self.save_iteration)
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

        plt.clf()
        plt.figure(figsize = (20, 5))
        plt.plot(x, self.dim_means)
        plt.xlabel('iteration')
        plt.ylabel('mean dimensions')
        plt.savefig(f'{directory}/dimensions.png')

        plt.clf()
        plt.figure(figsize = (20, 5))
        plt.plot(x, self.trans_stds)
        plt.xlabel('iteration')
        plt.ylabel('standard deviation of translations')
        plt.savefig(f'{directory}/translations.png')

        if not self.use_paschalidou_loss:
            plt.clf()
            plt.figure(figsize = (20, 5))
            plt.plot(x, self.penalty_means)
            plt.xlabel('iteration')
            plt.ylabel('penalty')
            plt.savefig(f'{directory}/penalty.png')
