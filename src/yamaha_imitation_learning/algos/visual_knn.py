import torch
import numpy as np
import matplotlib.pyplot as plt

from maxent_irl_costmaps.dataset.maxent_irl_dataset import MaxEntIRLDataset

from yamaha_imitation_learning.action_clustering.splines import get_vel_steer
from yamaha_imitation_learning.feature_extraction.pretrained_resnet import PretrainedResnetFeatureExtractor

class VisualKNN:
    """
    Method that computes actions using KNN to image features from a pretrained resnet
    """
    def __init__(self, dataset, img_extractor, k=5, dataset_skip=10, aggregate='best', device='cpu'):
        """
        Args:
            dataset: The dataset containing the images/actions to extract
            img_extractor: The extractor for images
            k: The number of neighbors to consider for the query
            dataset_skip: Process every n-th frame
            aggregate: Means of combining knn acts into single prediction. One of [best, mean, weighted, expweighted]
                best: Take the top action sequence
                mean: Take the mean of the topk
                weight: Take a mean, but weighted by knn distance
                expweight: Take a mean, weighted by the exp of knn distance
        """
        self.dataset = dataset
        self.img_extractor = img_extractor
        self.k = k
        self.dataset_skip = dataset_skip
        self.aggregate = aggregate

        self.image_embeddings = torch.zeros(0)
        self.action_embeddings = torch.zeros(0)
        self.actions = torch.zeros(0)

        self.device = device

    def process_dataset(self):
        image_embeddings = []
        action_embeddings = []
        actions = [] #store the action splines for viz

        print('preprocessing...')
        for i in range(0, len(self.dataset), self.dataset_skip):
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

    def predict(self, query):
        embeddings, dists, idxs = self.get_knn(query)
        acts = self.actions[idxs]

        if self.aggregate == 'best':
            return acts[0]

        elif self.aggregate == 'mean':
            return acts.mean(dim=0)

        elif self.aggregate == 'weight':
            #do some math to get all pos dists, with closest sample having highest weight
            dists2 = 2*dists.min(dim=0)[0] - dists
            weights = dists2 / dists2.sum(dim=0)
            return (acts * weights.view(-1, 1, 1)).sum(dim=0)

        elif self.aggregate == 'expweight':
            dists2 = dists.min(dim=0)[0] - dists
            weights = dists2.exp() / dists2.exp().sum(dim=0)
            return (acts * weights.view(-1, 1, 1)).sum(dim=0)
            
        else:
            print('Unsupported aggregation {}'.format(self.aggregate))
            exit(1)

    def predictk(self, query):
        embeddings, dists, idxs = self.get_knn(query)
        return self.actions[idxs]

    def get_knn(self, query):
        img = query['image']
        img_embedding = self.img_extractor.get_features(img)
        dists = torch.linalg.norm(
            self.image_embeddings - img_embedding.view(1, -1),
        dim=-1)

        mindists, idxs = torch.topk(dists, largest=False, k=self.k)
        return self.image_embeddings[idxs], dists[idxs], idxs

    def get_rand(self, query, k=5):
        img = query['image']
        img_embedding = self.img_extractor.get_features(img)
        dists = torch.linalg.norm(
            self.image_embeddings - img_embedding.view(1, -1),
        dim=-1)

        idxs = torch.randint(self.image_embeddings.shape[0], (k,))
        
        return self.image_embeddings[idxs], dists[idxs], idxs

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

        knn_feats, knn_dists, knn_idxs = self.get_knn(query)
        rand_feats, rand_dists, rand_idxs = self.get_rand(query)

        mosaic = """
        AACDEFG
        AAHIJKL
        BBMNOPQ
        BBRSTUV
        """

        fig, ax_dict = plt.subplot_mosaic(mosaic, figsize=(14, 8))
        axs = list(ax_dict.values())

        query_axs = [ax_dict[k] for k in 'AB']
        knn_img_axs = [ax_dict[k] for k in 'CDEFG']
        knn_act_axs = [ax_dict[k] for k in 'HIJKL']
        rand_img_axs = [ax_dict[k] for k in 'MNOPQ']
        rand_act_axs = [ax_dict[k] for k in 'RSTUV']

        query_axs[0].imshow(query['image'].permute(1, 2, 0)[:, :, [2, 1, 0]].cpu())
        query_axs[1].plot(query_acts[:, 0].cpu(), label='vel')
        query_axs[1].plot(query_acts[:, 1].cpu(), label='steer')

        query_axs[0].set_title('query image')
        query_axs[1].set_title('actions')

        for i, knn_idx in enumerate(knn_idxs[:5]):
            knn_pt = self.dataset[knn_idx]
            knn_img = knn_pt['image']
            knn_acts = get_vel_steer(knn_pt)

            #normalize actions for plotting
            knn_acts[..., 0] /= 15.
            knn_acts[..., 1] += 0.5
            knn_acts[..., 1] *= 0.5

            knn_img_axs[i].imshow(knn_img.permute(1, 2, 0)[:, :, [2, 1, 0]].cpu())
            knn_act_axs[i].plot(knn_acts[:, 0].cpu(), label='vel')
            knn_act_axs[i].plot(knn_acts[:, 1].cpu(), label='steer')

            knn_img_axs[i].set_title('knn image {} (d={:.2f})'.format(i+1, knn_dists[i]))
            knn_act_axs[i].set_title('actions')

        for i, rand_idx in enumerate(rand_idxs):
            rand_pt = self.dataset[rand_idx]
            rand_img = rand_pt['image']
            rand_acts = get_vel_steer(rand_pt)

            #normalize actions for plotting
            rand_acts[..., 0] /= 15.
            rand_acts[..., 1] += 0.5
            rand_acts[..., 1] *= 0.5

            rand_img_axs[i].imshow(rand_img.permute(1, 2, 0)[:, :, [2, 1, 0]].cpu())
            rand_act_axs[i].plot(rand_acts[:, 0].cpu(), label='vel')
            rand_act_axs[i].plot(rand_acts[:, 1].cpu(), label='steer')

            rand_img_axs[i].set_title('rand image {} (d={:.2f})'.format(i+1, rand_dists[i]))
            rand_act_axs[i].set_title('actions')

        for ax in axs:
            ax.legend()

        for ax in [query_axs[1]] + knn_act_axs + rand_act_axs:
            ax.set_ylim(0., 1.)

    def to(self, device):
        self.device = device
        self.dataset = self.dataset.to(device)
        self.img_extractor = self.img_extractor.to(device)

        self.image_embedings = self.image_embeddings.to(device)
        self.actions = self.actions.to(device)

        return self

if __name__ == '__main__':
    torch.set_printoptions(sci_mode=False)
    device = 'cuda'

    train_bag_fp = '/home/atv/Desktop/datasets/yamaha_maxent_irl/big_gridmaps/rosbags_debug'
    train_preprocess_fp = '/home/atv/Desktop/datasets/yamaha_maxent_irl/big_gridmaps/torch_debug'
    train_dataset = MaxEntIRLDataset(bag_fp=train_bag_fp, preprocess_fp=train_preprocess_fp)

    net = torch.load('../pretraining/resnet_byol_25000.pt')
#    net = torch.load('../pretraining/resnet_byol_0.pt')
    img_feature_extractor = PretrainedResnetFeatureExtractor(D=64, C=3, project=False, net=net)

    knn_predictor = VisualKNN(train_dataset, img_feature_extractor, dataset_skip=10).to(device)
    knn_predictor.process_dataset()

    test_bag_fp = '/home/atv/Desktop/datasets/yamaha_maxent_irl/big_gridmaps/rosbags_test'
    test_preprocess_fp = '/home/atv/Desktop/datasets/yamaha_maxent_irl/big_gridmaps/torch_test_h75'
    test_dataset = MaxEntIRLDataset(bag_fp=test_bag_fp, preprocess_fp=test_preprocess_fp).to(device)

    import pdb;pdb.set_trace()
    knn_predictor.visualize()
    plt.show()
    knn_predictor.visualize(idx=120)
    plt.show()
    knn_predictor.visualize(query=test_dataset[23])
    plt.show()
