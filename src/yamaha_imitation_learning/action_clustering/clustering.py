"""
Perform some basic clustering on action data via kmeans or something
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse

from torch_mpc.models.steer_setpoint_kbm import SteerSetpointKBM

from maxent_irl_costmaps.dataset.maxent_irl_dataset import MaxEntIRLDataset

from yamaha_imitation_learning.action_clustering.splines import get_vel_steer

def min_dispersion_cluster(dataset, model, k=16):
    """
    Cluster the action sequences in the dataset

    I'm probably going to start with a max dispersion thing

    For max dispersion, fine to just do it the stupid way
    """
    actions = []
    for i in range(len(dataset)):
        dpt = dataset[i]
        acts = get_vel_steer(dpt)
        actions.append(acts)

    actions = torch.stack(actions, dim=0)

    x0 = torch.zeros(5)
    U0 = torch.zeros_like(actions[0])
    tau_0 = model.rollout(x0, U0)

    dataset_rollouts = model.rollout(torch.tile(x0.view(1, 5), (len(actions), 1)), actions)

    clusters = [U0]
    cluster_rollouts = [tau_0]
    mask = torch.zeros(len(actions))

    for i in range(k-1):
        mindists = torch.ones_like(mask) + 1e10
        for c_rollout in cluster_rollouts:
            dists = torch.linalg.norm((c_rollout.unsqueeze(0) - dataset_rollouts)[..., :2], dim=-1).sum(dim=-1)
            mindists = torch.minimum(mindists, dists)

        idx = torch.argmax(mindists - mask*1e10)
        clusters.append(actions[idx])
        cluster_rollouts.append(dataset_rollouts[idx])
        mask[idx] = 1.

    clusters = torch.stack(clusters, dim=0)
    return clusters

if __name__ == '__main__':
    train_bag_fp = '/home/atv/Desktop/datasets/yamaha_maxent_irl/big_gridmaps/rosbags_train'
    train_preprocess_fp = '/home/atv/Desktop/datasets/yamaha_maxent_irl/big_gridmaps/torch_train_h75'
    train_dataset = MaxEntIRLDataset(bag_fp=train_bag_fp, preprocess_fp=train_preprocess_fp)

    x = torch.zeros(5)
    model = SteerSetpointKBM(L=3.0, v_target_lim=[0.0, 15.0], steer_lim=[-0.52, 0.52], steer_rate_lim=0.2, dt=0.1)

    act_clusters = min_dispersion_cluster(train_dataset, model, k=512)

    torch.save(act_clusters, 'act_clusters.pt')

    fig, axs = plt.subplots(1, 3, figsize=(12, 6))
    for seq in act_clusters:
        traj = model.rollout(x, seq)

        axs[0].plot(seq[:, 0])
        axs[0].set_title('Cluster Vels')
        axs[1].plot(seq[:, 1])
        axs[1].set_title('Cluster Steers')
        axs[2].plot(traj[:, 0], traj[:, 1])
        axs[2].set_title('Cluster Rollouts')

    plt.show()
