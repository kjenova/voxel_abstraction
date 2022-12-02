#!/usr/bin/env python
"""Script used to train the network for representing a 3D object as a set of
primitives
"""
import argparse
from itertools import izip
import json
import random
import os
import string
import sys

import numpy as np
import torch

from paschalidou.scripts.arguments import add_voxelizer_parameters, add_nn_parameters,\
    add_dataset_parameters, add_training_parameters,\
    add_regularizer_parameters, add_sq_mesh_sampler_parameters,\
    add_gaussian_noise_layer_parameters, voxelizer_shape,\
    add_loss_options_parameters, add_loss_parameters, get_loss_options
from paschalidou.scripts.output_logger import get_logger

from paschalidou.learnable_primitives.equal_distance_sampler_sq import get_sampler
from paschalidou.learnable_primitives.models import NetworkParameters, train_on_batch, \
    optimizer_factory
from paschalidou.learnable_primitives.loss_functions import euclidean_dual_loss

from common.batch_provider import BatchProvider, BatchProviderParams

from loader.load_preprocessed import load_preprocessed
from loader.load_urocell import load_urocell_preprocessed

def moving_average(prev_val, new_val, b):
    return (prev_val*b + new_val) / (b+1)


def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
   return ''.join(random.choice(chars) for _ in range(size))


def lr_schedule(optimizer, current_epoch, init_lr, factor, reductions):
    def inner(epoch):
        for i, e in enumerate(reductions):
            if epoch < e:
                return init_lr*factor**(-i)
        return init_lr*factor**(-len(reductions))

    for param_group in optimizer.param_groups:
        param_group['lr'] = inner(current_epoch)

    return optimizer


def get_weight(w, epoch, current_epoch):
    if current_epoch < epoch:
        return 0.0
    else:
        return w


def get_regularizer_terms(args, current_epoch):
    # Create a dictionary with the regularization terms if there are any
    regularizer_terms = {
        "regularizer_type": args.regularizer_type,
        "bernoulli_regularizer_weight": args.bernoulli_regularizer_weight,
        "entropy_bernoulli_regularizer_weight":
            args.entropy_bernoulli_regularizer_weight,
        "parsimony_regularizer_weight": args.parsimony_regularizer_weight,
        "sparsity_regularizer_weight": args.sparsity_regularizer_weight,
        "overlapping_regularizer_weight": args.overlapping_regularizer_weight,
        "minimum_number_of_primitives": args.minimum_number_of_primitives,
        "maximum_number_of_primitives": args.maximum_number_of_primitives,
        "w1": args.w1,
        "w2": args.w2
    }

    return regularizer_terms


def save_experiment_params(args, experiment_tag, directory):
    t = vars(args)
    params = {k: str(v) for k, v in t.iteritems()}

    git_head_hash = "foo"
    params["git-commit"] = git_head_hash
    params["experiment_tag"] = experiment_tag
    for k, v in params.items():
        if v == "":
            params[k] = None
    with open(os.path.join(directory, "params.json"), "w") as f:
        json.dump(params, f, indent=4)


def main(argv):
    parser = argparse.ArgumentParser(
        description="Train a network to predict primitives"
    )
    parser.add_argument(
        "--train_dir",
        default="data/chamferData/01",
        help="Path to the directory containing the dataset"
    )
    parser.add_argument(
        "--urocell_dir",
        default="data/chamferData/urocell"
    )
    parser.add_argument(
        "--output_directory",
        default="results/paschalidou",
        help="Save the output files in that directory"
    )

    parser.add_argument(
        "--weight_file",
        default=None,
        help=("The path to a previously trainined model to continue"
              " the training from")
    )
    parser.add_argument(
        "--continue_from_epoch",
        default=0,
        type=int,
        help="Continue training from epoch (default=0)"
    )
    parser.add_argument(
        "--n_primitives",
        type=int,
        default=16, # 32, # Za 32 je premalo VRAM-a.
        help="Number of primitives"
    )
    parser.add_argument(
        "--use_deformations",
        action="store_true",
        help="Use Superquadrics with deformations as the shape configuration"
    )
    parser.add_argument(
        "--probs_only",
        action="store_true",
        help="Optimize only using the probabilities"
    )

    parser.add_argument(
        "--experiment_tag",
        default=None,
        help="Tag that refers to the current experiment"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=27,
        help="Seed for the PRNG"
    )

    add_nn_parameters(parser)
    add_dataset_parameters(parser)
    add_voxelizer_parameters(parser)
    add_training_parameters(parser)
    add_sq_mesh_sampler_parameters(parser)
    add_regularizer_parameters(parser)
    add_gaussian_noise_layer_parameters(parser)
    # Parameters related to the loss function and the loss weights
    add_loss_parameters(parser)
    # Parameters related to loss options
    add_loss_options_parameters(parser)
    args = parser.parse_args(argv)

    device = torch.device("cuda:1")

    # Check if output directory exists and if it doesn't create it
    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)

    # Create an experiment directory using the experiment_tag
    if False:
        if args.experiment_tag is None:
            experiment_tag = id_generator(9)
        else:
            experiment_tag = args.experiment_tag
    else:
        max_experiment = 0
        for name in os.listdir(args.output_directory):
            if os.path.isdir(os.path.join(args.output_directory, name)) and name.isdigit():
                max_experiment = max(int(name), max_experiment)

        experiment_tag = str(max_experiment + 1)

    experiment_directory = os.path.join(
        args.output_directory,
        experiment_tag
    )
    if not os.path.exists(experiment_directory):
        os.makedirs(experiment_directory)

    # Store the parameters for the current experiment in a json file
    save_experiment_params(args, experiment_tag, experiment_directory)
    print "Save experiment statistics in %s" %(experiment_tag, )

    # Create two files to store the training and test evolution
    train_stats = os.path.join(experiment_directory, "train.txt")
    val_stats = os.path.join(experiment_directory, "val.txt")
    if args.weight_file is None:
        train_stats_f = open(train_stats, "w")
    else:
        train_stats_f = open(train_stats, "a+")
    train_stats_f.write((
        "epoch loss pcl_to_prim_loss prim_to_pcl_loss bernoulli_regularizer "
        "entropy_bernoulli_regularizer parsimony_regularizer "
        "overlapping_regularizer sparsity_regularizer\n"
    ))

    # Set the random seed
    np.random.seed(args.seed)
    torch.manual_seed(np.random.randint(np.iinfo(np.int32).max))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(np.random.randint(np.iinfo(np.int32).max))

    # Create an object that will sample points in equal distances on the
    # surface of the primitive
    sampler = get_sampler(
        args.use_cuboids,
        args.n_points_from_sq_mesh,
        args.D_eta,
        args.D_omega
    )

    train_set = load_preprocessed(args.train_dir)
    validation_set, _ = load_urocell_preprocessed(args.urocell_dir)

    batch_params = BatchProviderParams(
        device,
        n_samples_per_shape = args.n_points_from_mesh,
        batch_size = args.batch_size
    )

    train_batches = BatchProvider(
        train_set,
        batch_params,
        store_on_gpu = False,
        uses_point_sampling = True
    )
    validation_batches = BatchProvider(
        validation_set,
        batch_params,
        store_on_gpu = False,
        uses_point_sampling = True
    )

    network_params = NetworkParameters.from_options(args)
    # Build the model to be used for training
    model = network_params.network(network_params)

    # Move model to the device to be used
    model.to(device)
    # Check whether there is a weight file provided to continue training from
    if args.weight_file is not None:
        model.load_state_dict(torch.load(args.weight_file))
    model.train()

    # Build an optimizer object to compute the gradients of the parameters
    optimizer = optimizer_factory(args, model)

    # Loop over the dataset multiple times
    pcl_to_prim_losses = []
    prim_to_pcl_losses = []
    losses = []
    min_val_loss = float('inf')
    for i in range(args.epochs):
        bar = get_logger(
            "euclidean_dual_loss",
            i+1,
            args.epochs,
            args.steps_per_epoch
        )
        for b in range(args.steps_per_epoch):
            X, y_target = train_batches.get()[:2]
            X = X.unsqueeze(1)

            # Train on batch
            batch_loss, metrics, debug_stats = train_on_batch(
                model,
                lr_schedule(
                    optimizer, i, args.lr, args.lr_factor, args.lr_epochs
                ),
                euclidean_dual_loss,
                X,
                y_target,
                get_regularizer_terms(args, i),
                sampler,
                get_loss_options(args)
            )

            # Get the regularizer terms
            reg_values = debug_stats["regularizer_terms"]
            sparsity_regularizer = reg_values["sparsity_regularizer"]
            overlapping_regularizer = reg_values["overlapping_regularizer"]
            parsimony_regularizer = reg_values["parsimony_regularizer"] 
            entropy_bernoulli_regularizer = reg_values["entropy_bernoulli_regularizer"]
            bernoulli_regularizer = reg_values["bernoulli_regularizer"]

            # The lossess
            pcl_to_prim_loss = debug_stats["pcl_to_prim_loss"].item()
            prim_to_pcl_loss = debug_stats["prim_to_pcl_loss"].item()
            bar.loss = moving_average(bar.loss, batch_loss, b)
            bar.pcl_to_prim_loss = \
                moving_average(bar.pcl_to_prim_loss, pcl_to_prim_loss, b)
            bar.prim_to_pcl_loss = \
                moving_average(bar.prim_to_pcl_loss, prim_to_pcl_loss, b)

            losses.append(bar.loss)
            prim_to_pcl_losses.append(bar.prim_to_pcl_loss)
            pcl_to_prim_losses.append(bar.pcl_to_prim_loss)

            bar.bernoulli_regularizer =\
                (bar.bernoulli_regularizer * b + bernoulli_regularizer) / (b+1)
            bar.parsimony_regularizer =\
                (bar.parsimony_regularizer * b + parsimony_regularizer) / (b+1)
            bar.overlapping_regularizer =\
                (bar.overlapping_regularizer * b + overlapping_regularizer) / (b+1)
            bar.entropy_bernoulli_regularizer = \
                (bar.entropy_bernoulli_regularizer * b +
                 entropy_bernoulli_regularizer) / (b+1)
            bar.sparsity_regularizer =\
                (bar.sparsity_regularizer * b + sparsity_regularizer) / (b+1)

            bar.exp_n_prims = metrics[0].sum(-1).mean()
            # Update the file that keeps track of the statistics
            train_stats_f.write(
                ("%d %.8f %.8f %.8f %.6f %.6f %.6f %.6f %.6f") %(
                i, bar.loss, bar.pcl_to_prim_loss, bar.prim_to_pcl_loss,
                bar.bernoulli_regularizer,
                bar.entropy_bernoulli_regularizer,
                bar.parsimony_regularizer,
                bar.overlapping_regularizer,
                bar.sparsity_regularizer
                )
            )
            train_stats_f.write("\n")
            train_stats_f.flush()

            bar.next()

        # Finish the progress bar and save the model after every epoch
        bar.finish()

        val_loss_sum = .0
        val_set_size = 0
        for batch in validation_batches.get_all_batches(shuffle = False):
            X, y_target = batch[:2]
            X = X.unsqueeze(1)

            y_hat = model(X)
            loss, _ = euclidean_dual_loss(
                y_hat,
                y_target,
                get_regularizer_terms(args, i),
                sampler,
                get_loss_options(args)
            )

            val_loss_sum += loss.item() * X.size(0)
            val_set_size += X.size(0)

        val_loss = val_loss_sum / val_set_size
        print(val_loss)
        if val_loss < min_val_loss:
            min_val_loss = val_loss

            torch.save(
                model.state_dict(),
                os.path.join(
                    experiment_directory,
                    "model.torch" # _%d" % (i + args.continue_from_epoch,)
                )
            )

    print [
        sum(losses[args.steps_per_epoch:]) / float(args.steps_per_epoch),
        sum(losses[:args.steps_per_epoch]) / float(args.steps_per_epoch),
        sum(pcl_to_prim_losses[args.steps_per_epoch:]) / float(args.steps_per_epoch),
        sum(pcl_to_prim_losses[:args.steps_per_epoch]) / float(args.steps_per_epoch),
        sum(prim_to_pcl_losses[args.steps_per_epoch:]) / float(args.steps_per_epoch),
        sum(prim_to_pcl_losses[:args.steps_per_epoch]) / float(args.steps_per_epoch),
    ]


if __name__ == "__main__":
    main(sys.argv[1:])
