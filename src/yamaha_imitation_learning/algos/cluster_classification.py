import torch
import numpy as np
import matplotlib.pyplot as plt

from torch_mpc.models.steer_setpoint_kbm import SteerSetpointKBM

from maxent_irl_costmaps.dataset.maxent_irl_dataset import MaxEntIRLDataset

from yamaha_imitation_learning.action_clustering.splines import get_vel_steer
from yamaha_imitation_learning.feature_extraction.pretrained_resnet import PretrainedResnetFeatureExtractor

class ClusterClassifier:
    """
    Get a policy by performing classification on a set of clusters
    """
    def __init__(self, dataset, clusters, n_clusters=16, device='cpu'):
        """
        TODO: pass in the net and opt
        """
        self.dataset = dataset
        self.act_clusters = clusters[:n_clusters]
        self.norm = torch.tensor([0.5, 10.0])

        self.labels = self.generate_labels()
        self.device = device

    def generate_labels(self):
        """
        Generate labels based on min distance to normalized action cluster
        """
        labels = []
        for i in range(len(self.dataset)):
            dpt = self.dataset[i]
            acts = get_vel_steer(dpt)
            cluster_diffs = acts.unsqueeze(0) - self.act_clusters
            norm_cluster_diffs = cluster_diffs * self.norm.view(1, 1, -1)
            norm_cluster_dists = norm_cluster_diffs.abs().sum(dim=-1).sum(dim=-1)
            label = norm_cluster_dists.argmin()
            labels.append(label)
        return torch.stack(labels)

    def visualize(self, idx=-1):
        #pick a random sample if idx not given
        if idx == -1:
            idx = np.random.choice(len(self.dataset))

        query = self.dataset[idx]
        gt_acts = get_vel_steer(query)
        label = self.labels[idx]
        model = SteerSetpointKBM(L=3.0, v_target_lim=[0.0, 15.0], steer_lim=[-0.52, 0.52], steer_rate_lim=0.2, dt=0.1).to(self.device)

        fig, axs = plt.subplots(2, 2, figsize=(12, 12))
        axs = axs.flatten()

        #Show camera image
        axs[0].imshow(query['image'].permute(1, 2, 0)[:, :, [2, 1, 0]])
        axs[0].set_title('FPV')

        #Show model
        for i in range(len(self.act_clusters)):
            act_cluster = self.act_clusters[i]
            x = torch.zeros(5).to(self.device)
            tau = model.rollout(x, act_cluster)

            axs[1].plot(tau[:, 0], tau[:, 1], c='r' if i == label else 'k', alpha=1.0 if i == label else 0.2)
        axs[1].set_title('Model rollout')

        #Show vels
        for i in range(len(self.act_clusters)):
            act_cluster = self.act_clusters[i]

            axs[2].plot(act_cluster[:, 0], c='r' if i == label else 'k', alpha=1.0 if i == label else 0.2)
        axs[2].plot(gt_acts[:, 0], c='c')
        axs[2].set_title('Vel')

        #Show steers
        for i in range(len(self.act_clusters)):
            act_cluster = self.act_clusters[i]

            axs[3].plot(act_cluster[:, 1], c='r' if i == label else 'k', alpha=1.0 if i == label else 0.2)
        axs[3].plot(gt_acts[:, 1], c='c')
        axs[3].set_title('Steer')

    def to(self, device):
        self.device = device
        self.dataset = self.dataset.to(device)
        self.act_clusters = self.act_clusters.to(device)
        self.norm = self.norm.to(device)
        self.labels = self.labels.to(device)
        return self

if __name__ == '__main__':
    torch.set_printoptions(sci_mode=False)
    device = 'cuda'

    train_bag_fp = '/home/atv/Desktop/datasets/yamaha_maxent_irl/big_gridmaps/rosbags_train'
    train_preprocess_fp = '/home/atv/Desktop/datasets/yamaha_maxent_irl/big_gridmaps/torch_train_h75'
    train_dataset = MaxEntIRLDataset(bag_fp=train_bag_fp, preprocess_fp=train_preprocess_fp)

    net = torch.load('../pretraining/resnet_byol_25000.pt')
    img_feature_extractor = PretrainedResnetFeatureExtractor(D=64, C=3, project=False, net=net)

    act_seqs = torch.load('../action_clustering/act_clusters.pt')

    classifier = ClusterClassifier(train_dataset, act_seqs, n_clusters=64)

    test_bag_fp = '/home/atv/Desktop/datasets/yamaha_maxent_irl/big_gridmaps/rosbags_test'
    test_preprocess_fp = '/home/atv/Desktop/datasets/yamaha_maxent_irl/big_gridmaps/torch_test_h75'
    test_dataset = MaxEntIRLDataset(bag_fp=test_bag_fp, preprocess_fp=test_preprocess_fp).to(device)

    for i in range(100):
        classifier.visualize()
        plt.show()
