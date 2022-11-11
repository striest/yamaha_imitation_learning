import torch
import numpy as np
import matplotlib.pyplot as plt

from maxent_irl_costmaps.dataset.maxent_irl_dataset import MaxEntIRLDataset

from yamaha_imitation_learning.action_clustering.splines import get_vel_steer
from yamaha_imitation_learning.feature_extraction.pretrained_resnet import PretrainedResnetFeatureExtractor

class SanityCheck:
    """
    Implement some random baselines to sanity check
    """
    def __init__(self, dataset, dataset_skip=10, k=10, aggregate='random', device='cpu'):
        """
        Args:
            dataset: The dataset containing the images/actions to extract
            dataset_skip: Process every n-th frame
            aggregate: The method for selecting actions. One of the following:
                random: Grab a random action from the dataset
                mean: Grab the mean action from the dataset
        """
        self.dataset = dataset
        self.dataset_skip = dataset_skip
        self.k = k
        self.aggregate = aggregate

        self.action_embeddings = torch.zeros(0)
        self.actions = torch.zeros(0)

        self.device = device

    def process_dataset(self):
        action_embeddings = []
        actions = [] #store the action splines for viz

        print('preprocessing...')
        for i in range(0, len(self.dataset), self.dataset_skip):
            print('{}/{}'.format(i, len(self.dataset)), end='\r')
            batch = self.dataset[i]
            img = batch['image']
            acts = get_vel_steer(batch)
            actions.append(acts)

            if i >= len(self.dataset)-1:
                break

        self.actions = torch.stack(actions, dim=0) #[B x T x 2]

    def predict(self, query):
        if self.aggregate == 'random':
            idx = np.random.choice(np.arange(len(self.actions)))
            return self.actions[idx]

        elif self.aggregate == 'mean':
            return self.actions.mean(dim=0)
        else:
            print('Unsupported aggregation {}'.format(self.aggregate))
            exit(1)

    def predictk(self, query):
        if self.aggregate == 'random':
            idxs = np.random.choice(np.arange(len(self.actions)), (self.k, ))
            return self.actions[idxs]

        elif self.aggregate == 'mean':
            return torch.tile(self.actions.mean(dim=0, keepdim=True), [self.k, 1, 1])
        else:
            print('Unsupported aggregation {}'.format(self.aggregate))
            exit(1)

    def visualize(self, query=None, idx=-1):
        #get a sample to visualize if not provided
        if query is None:
            #pick a random sample if idx not given
            if idx == -1:
                idx = np.random.choice(len(self.dataset))

            query = self.dataset[idx]

        query_acts = get_vel_steer(query)
        query_acts[..., 0] /= 15
        query_acts[..., 1] += 0.5
        query_acts[..., 1] *= 0.5

        acts = self.predict(query)
        acts[..., 0] /= 15
        acts[..., 1] += 0.5
        acts[..., 1] *= 0.5

        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].imshow(query['image'].permute(1, 2, 0)[:, :, [2, 1, 0]].cpu())
        axs[1].plot(query_acts[:, 0].cpu(), label='vel')
        axs[1].plot(query_acts[:, 1].cpu(), label='steer')
        axs[2].plot(acts[:, 0].cpu(), label='pred vel')
        axs[2].plot(acts[:, 1].cpu(), label='pred steer')

        axs[0].set_title('query image')
        axs[1].set_title('gt actions')
        axs[2].set_title('pred actions')

        for ax in axs:
            ax.legend()

    def to(self, device):
        self.device = device
        self.dataset = self.dataset.to(device)

        self.actions = self.actions.to(device)

        return self

if __name__ == '__main__':
    torch.set_printoptions(sci_mode=False)
    device = 'cpu'

    train_bag_fp = '/home/atv/Desktop/datasets/yamaha_maxent_irl/big_gridmaps/rosbags_debug'
    train_preprocess_fp = '/home/atv/Desktop/datasets/yamaha_maxent_irl/big_gridmaps/torch_debug'
    train_dataset = MaxEntIRLDataset(bag_fp=train_bag_fp, preprocess_fp=train_preprocess_fp)

    knn_predictor = SanityCheck(train_dataset, dataset_skip=10, aggregate='random').to(device)
    knn_predictor2 = SanityCheck(train_dataset, dataset_skip=10, aggregate='mean').to(device)
    knn_predictor.process_dataset()
    knn_predictor2.process_dataset()

    test_bag_fp = '/home/atv/Desktop/datasets/yamaha_maxent_irl/big_gridmaps/rosbags_test'
    test_preprocess_fp = '/home/atv/Desktop/datasets/yamaha_maxent_irl/big_gridmaps/torch_test_h75'
    test_dataset = MaxEntIRLDataset(bag_fp=test_bag_fp, preprocess_fp=test_preprocess_fp).to(device)

    knn_predictor.visualize()
    plt.show()
    knn_predictor.visualize(idx=120)
    plt.show()
    knn_predictor.visualize(query=test_dataset[23])
    plt.show()

    for i in range(5):
        x = test_dataset[0]
        a1 = knn_predictor.predict(x)
        a2 = knn_predictor2.predict(x)

        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        axs[0].imshow(x['image'].permute(1, 2, 0)[:, :, [2, 1, 0]])
        axs[1].plot(a1[:, 0], c='b', label='throttle (random)')
        axs[1].plot(a2[:, 0], c='r', label='throttle (mean)')
        axs[2].plot(a1[:, 1], c='b', label='steer (random)')
        axs[2].plot(a2[:, 1], c='r', label='steer (mean)')
        for ax in axs:
            ax.legend()
        plt.show()
        
