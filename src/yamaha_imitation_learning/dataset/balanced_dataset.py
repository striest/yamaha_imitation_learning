import torch
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset

from maxent_irl_costmaps.dataset.maxent_irl_dataset import MaxEntIRLDataset

from yamaha_imitation_learning.action_clustering.splines import get_vel_steer
from yamaha_imitation_learning.feature_extraction.pretrained_resnet import PretrainedResnetFeatureExtractor

class BalancedDataset(Dataset):
    """
    A dataset that attempts to balance the distribution of samples

    I'm currently messing with two methods:
        1. balance based on control effort
        2. balance based on image emebeddings
        3. some weighted combo of the two
    """
    def __init__(self, base_dataset, image_extractor, distill_p=0.1, device='cpu'):
        self.dataset = base_dataset
        self.img_extractor = image_extractor
        self.distill_p = distill_p

        self.preprocess()
        self.device = device

    def __len__(self):
        return self.sample_idxs.shape[0]

    def __getitem__(self, idx):
        return self.dataset[self.sample_idxs[idx]]

    def preprocess(self):
        """
        compute control effort and image embeddings
        """
        image_embeddings = []
        actions = []

        print('preprocessing...')
        for i in range(len(self.dataset)):
            print('{}/{}'.format(i, len(self.dataset)), end='\r')
            batch = self.dataset[i]
            img = batch['image']
            acts = get_vel_steer(batch)
            with torch.no_grad():
                img_embedding = self.img_extractor.get_features(img)

            image_embeddings.append(img_embedding)
            actions.append(acts)

            if i >= len(self.dataset)-1:
                break

        self.image_embeddings = torch.stack(image_embeddings, dim=0) #[B x Ni]
        self.actions = torch.stack(actions, dim=0) #[B x T x 2]

        """
        Distill dataset into a smaller set of balanced examples
        For now, I'm using a weighted score of:
            1. Velocity effort
            2. Steer effort
            3. Distance to other embeddings
        """
        velocity_effort = (self.actions[..., 1:, 0] - self.actions[..., :-1, 0]).abs().sum(dim=-1)
        steer_effort = (self.actions[..., 1:, 1] - self.actions[..., :-1, 1]).abs().sum(dim=-1)

        embedding_dists = []
        #I'm looping in case the dataset is really big
        for i in range(len(self.image_embeddings)):
            dists = torch.linalg.norm(self.image_embeddings[[i]] - self.image_embeddings, dim=-1)
            embedding_dists.append(dists.mean())

        embedding_dists = torch.stack(embedding_dists, dim=0)

        self.normalized_velocity_effort = (velocity_effort - velocity_effort.min()) / (velocity_effort.max() - velocity_effort.min())

        self.normalized_steer_effort = (steer_effort - steer_effort.min()) / (steer_effort.max() - steer_effort.min())

        self.normalized_embedding_dists = (embedding_dists - embedding_dists.min()) / (embedding_dists.max() - embedding_dists.min())

        self.distill_scores = self.normalized_velocity_effort + self.normalized_steer_effort + self.normalized_embedding_dists

        #ok I think we can afford to loop through temporally to add a distance penalty
        n_samples = int(self.distill_p * len(self.dataset))
        sample_idxs = []
        sample_dists = torch.ones_like(self.distill_scores) + len(self.dataset)
        mask = torch.zeros_like(self.distill_scores)
        for i in range(n_samples):
            normalized_sample_dists = (sample_dists - sample_dists.min()) / (sample_dists.max() - sample_dists.min() + 1e-8)
            sidx = torch.argmax(self.distill_scores + normalized_sample_dists + mask)

            #set up next sample
            new_sample_dists = (torch.arange(len(self.dataset), device=sidx.device) - sidx).abs()
            sample_dists = torch.min(new_sample_dists, sample_dists)
            mask[sidx] = -1e10 #prevent the same idx being picked twice
            sample_idxs.append(sidx)

        self.sample_idxs = torch.stack(sample_idxs, dim=0)
        self.sample_idxs, _ = torch.sort(self.sample_idxs)

    def to(self, device):
        self.dataset = self.dataset.to(device)
        self.img_extractor = self.img_extractor.to(device)
        self.image_embeddings = self.image_embeddings.to(device)
        self.actions = self.actions.to(device)
        self.normalized_velocity_effort = self.normalized_velocity_effort.to(device)
        self.normalized_steer_effort = self.normalized_steer_effort.to(device)
        self.normalized_embedding_dists = self.normalized_embedding_dists.to(device)
        self.distill_scores = self.distill_scores.to(device)
        return self

if __name__ == '__main__':
    torch.set_printoptions(sci_mode=False)
    device = 'cuda'

    train_bag_fp = '/home/atv/Desktop/datasets/yamaha_maxent_irl/big_gridmaps/rosbags_train'
    train_preprocess_fp = '/home/atv/Desktop/datasets/yamaha_maxent_irl/big_gridmaps/torch_train_h75'
    train_dataset = MaxEntIRLDataset(bag_fp=train_bag_fp, preprocess_fp=train_preprocess_fp)

    net = torch.load('../pretraining/resnet_byol_25000.pt')
    img_feature_extractor = PretrainedResnetFeatureExtractor(D=64, C=3, project=False, net=net)

    balanced_dataset = BalancedDataset(train_dataset, img_feature_extractor, distill_p=0.1)

    print('Dataset size {} -> {}'.format(len(train_dataset), len(balanced_dataset)))

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    plt.show(block = False)
    for _ in range(10):
        for i in range(len(balanced_dataset)):
            for ax in axs:
                ax.cla()

            fig.suptitle('Frame {}/{}'.format(i+1, len(balanced_dataset)))

            dpt = balanced_dataset[i]
            axs[0].imshow(dpt['image'].permute(1, 2, 0)[:, :, [2, 1, 0]].cpu())
            axs[0].set_title('FPV')

            acts = get_vel_steer(dpt)
            acts[..., 0] /= 15
            acts[..., 1] += 0.5
            acts[..., 1] *= 0.5
            axs[1].plot(acts[:, 0].cpu(), label='vel')
            axs[1].plot(acts[:, 1].cpu(), label='steer')
            axs[1].legend()
            axs[1].set_title('Controls')
            plt.pause(0.5)
            
