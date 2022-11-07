import torch
import numpy as np
import os

from tulsiani.parameters import params
from tulsiani.network import TulsianiNetwork
from tulsiani.net_utils import scale_weights
from tulsiani.reinforce import ReinforceRewardUpdater
from tulsiani.stats import TulsianiStats

from common.batch_provider import BatchProvider
from common.reconstruction_loss import reconstruction_loss

from loader.load_preprocessed import load_preprocessed
from loader.load_urocell import load_urocell_preprocessed

def train(network, train_batches, validation_batches, params, stats):
    optimizer = torch.optim.Adam(network.parameters(), lr = params.learning_rate)
    reinforce_updater = ReinforceRewardUpdater(params.reinforce_baseline_momentum)

    best_validation_loss = float('inf')

    network.train()
    for _ in range(params.n_iterations[params.phase]):
        optimizer.zero_grad()

        volume, sampled_points, closest_points = train_batches.get()
        P = network(volume)
        cov, cons = reconstruction_loss(volume, P, sampled_points, closest_points, params.n_samples_per_primitive)
        loss = cov + cons

        total_penalty = .0

        if params.prune_primitives:
            # Pri nas se minimizira 'penalty', čeprav se pri REINFORCE tipično maksimizira 'reward'.
            # Edina razlika je v tem, da bi v primeru, da bi maksimizirali 'reward = - penalty', morali
            # potem še pri gradientu dodati minus.
            for p in range(params.n_primitives):
                penalty = loss + params.existence_penalty * P.exist[:, p]
                total_penalty += penalty.mean().item()
                P.log_prob[:, p] *= reinforce_updater.update(penalty)

            # Ker je navaden loss reduciran z .mean(), tudi ta "REINFORCE loss" reduciram z .mean():
            (loss.mean() + P.log_prob.mean()).backward()
        else:
            loss.mean().backward()

        optimizer.step()

        i = train_batches.iteration - 1
        stats.cov[i] = cov.mean()
        stats.cons[i] = cons.mean()
        stats.prob_means[i] = P.prob.mean()
        stats.penalty_means[i] = total_penalty / params.n_primitives
        i += 1

        if i % params.save_iteration == 0:
            cov_mean = stats.cov[i - params.save_iteration : i].mean()
            cons_mean = stats.cons[i - params.save_iteration : i].mean()
            mean_prob = stats.prob_means[i - params.save_iteration : i].mean()
            mean_penalty = stats.penalty_means[i - params.save_iteration : i].mean()

            print(f'---- iteration {i} ----')
            print(f'    loss {cov_mean + cons_mean}, cov: {cov_mean}, cons: {cons_mean}')
            print(f'    mean prob: {mean_prob}')
            print(f'    mean penalty: {mean_penalty}')

            validation_loss = .0

            network.eval()

            with torch.no_grad():
                for (volume, sampled_points, closest_points) in validation_batches.get_all_batches():
                    P = network(volume)
                    cov, cons = reconstruction_loss(volume, P, sampled_points, closest_points, params)
                    validation_loss += (cov + cons).sum() # .sum() zato, ker kasneje delimo.

            network.train()

            validation_loss /= validation_batches.n

            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                torch.save(network.state_dict(), 'results/tulsiani/save.torch')

            j = train_batches.iteration // params.save_iteration - 1
            stats.validation_loss[j] = validation_loss

            print(f'    validation loss: {validation_loss}')
            print(f'    best validation loss: {best_validation_loss}')

try:
    os.makedirs('results/tulsiani/graphs')
except FileExistsError:
    pass

train_set = load_preprocessed(params.train_dir, 32)

params.grid_size = train_set[0].resized_volume.shape[0]

validation_set, _ = load_urocell_preprocessed(params.urocell_dir)

train_batches = BatchProvider(train_set, params, store_on_gpu = False)
validation_batches = BatchProvider(validation_set, params, store_on_gpu = False)

stats = TulsianiStats(params)

params.phase = 0
network = TulsianiNetwork(params)
network.to(params.device)

train(network, train_batches, validation_batches, params, stats)

params.phase = 1
network = TulsianiNetwork(params)
network.load_state_dict(torch.load('results/tulsiani/save.torch'))
scale_weights(network, params)
network.to(params.device)

train(network, train_batches, validation_batches, params, stats)

stats.save_plots('results/tulsiani/graphs')
