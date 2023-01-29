import argparse
import os
import sys

import numpy as np
import torch

from paschalidou.scripts.arguments import add_voxelizer_parameters, add_nn_parameters,\
     add_dataset_parameters, add_gaussian_noise_layer_parameters, \
     voxelizer_shape, add_loss_options_parameters, add_loss_parameters

from paschalidou.learnable_primitives.equal_distance_sampler_sq import\
    EqualDistanceSamplerSQ
from paschalidou.learnable_primitives.models import NetworkParameters
from paschalidou.learnable_primitives.loss_functions import euclidean_dual_loss
from paschalidou.learnable_primitives.primitives import\
    euler_angles_to_rotation_matrices, quaternions_to_rotation_matrices

from paschalidou.scripts.visualization_utils import points_on_sq_surface, points_on_cuboid

from loader.load_urocell import load_urocell_preprocessed

from common.batch_provider import BatchProvider, BatchProviderParams

from train_paschalidou import iou

def get_shape_configuration(use_cuboids):
    if use_cuboids:
        return points_on_cuboid
    else:
        return points_on_sq_surface

def inference(dataset):
    basedir = "results/paschalidou"
    max_experiment = 0
    for name in os.listdir(basedir):
        if os.path.isdir(os.path.join(basedir, name)) and name.isdigit():
            max_experiment = max(int(name), max_experiment)

    if max_experiment <= None:
        return None

    model_path = os.path.join(basedir, str(max_experiment), "model.torch")

    parser = argparse.ArgumentParser(
        description="Do the forward pass and estimate a set of primitives"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Number of samples in a batch (default=32)"
    )
    parser.add_argument(
        "--n_primitives",
        type=int,
        default=20,
        help="Number of primitives"
    )
    parser.add_argument(
        "--prob_threshold",
        type=float,
        default=0.5,
        help="Probability threshold"
    )
    parser.add_argument(
        "--use_deformations",
        action="store_true",
        help="Use Superquadrics with deformations as the shape configuration"
    )
    parser.add_argument(
        "--add_coordinates_to_encoder",
        action="store_true"
    )
    parser.add_argument(
        "--iou_n_points",
        default=100000,
        type=int
    )

    add_dataset_parameters(parser)
    add_nn_parameters(parser)
    add_voxelizer_parameters(parser)
    add_gaussian_noise_layer_parameters(parser)
    add_loss_parameters(parser)
    add_loss_options_parameters(parser)
    args = parser.parse_args(sys.argv[1:])

    # A sampler instance
    e = EqualDistanceSamplerSQ(200)

    device = torch.device("cuda:1")

    batch_params = BatchProviderParams(
        device,
        n_samples_per_shape = args.n_points_from_mesh,
        batch_size = 1 # args.batch_size
    )

    test_batches = BatchProvider(
        dataset,
        batch_params,
        store_on_gpu = True,
        uses_point_sampling = True
    )

    network_params = NetworkParameters.from_options(args)
    # Build the model to be used for testing
    model = network_params.network(network_params)
    # Move model to device to be used
    model.to(device)
    model.load_state_dict(
        torch.load(model_path, map_location=device)
    )
    model.eval()

    points = []
    probability = []

    total_iou = .0
    n = 0

    with torch.no_grad():
        for volume, _, _ in test_batches.get_all_batches():
            # Do the forward pass and estimate the primitive parameters
            y_hat = model(volume.unsqueeze(1))

            M = args.n_primitives  # number of primitives
            probs = y_hat[0].to("cpu").detach().numpy()
            # Transform the Euler angles to rotation matrices
            if y_hat[2].shape[1] == 3:
                R = euler_angles_to_rotation_matrices(
                    y_hat[2].view(-1, 3)
                ).to("cpu").detach()
            else:
                R = quaternions_to_rotation_matrices(
                        y_hat[2].view(-1, 4)
                    ).to("cpu").detach()
                # get also the raw quaternions
                quats = y_hat[2].view(-1, 4).to("cpu").detach().numpy()
            translations = y_hat[1].to("cpu").view(args.n_primitives, 3)
            translations = translations.detach().numpy()

            shapes = y_hat[3].to("cpu").view(args.n_primitives, 3).detach().numpy()
            epsilons = y_hat[4].to("cpu").view(
                args.n_primitives, 2
            ).detach().numpy()
            taperings = y_hat[5].to("cpu").view(
                args.n_primitives, 2
            ).detach().numpy()

            primitives_points = []
            for i in range(args.n_primitives):
                _, _, _, p = get_shape_configuration(args.use_cuboids)(
                    shapes[i, 0],
                    shapes[i, 1],
                    shapes[i, 2],
                    epsilons[i, 0],
                    epsilons[i, 1],
                    R[i].numpy(),
                    translations[i].reshape(-1, 1),
                    taperings[i, 0],
                    taperings[i, 1]
                )

                primitives_points.append(p.transpose())

            points.append(np.stack(primitives_points))
            probability.append(probs[0, :])

            total_iou += iou(volume, y_hat, args).sum().item()
            n += volume.size(0)

    points = np.stack(points)
    probability = np.stack(probability)

    with open("results/paschalidou/points.npy", "wb") as f:
        np.save(f, points)

    with open("results/paschalidou/probability.npy", "wb") as f:
        np.save(f, probability)

    mean_iou = total_iou / n
    print("mean IoU = " + str(mean_iou))

_, test = load_urocell_preprocessed("data/urocell")
inference(test)

