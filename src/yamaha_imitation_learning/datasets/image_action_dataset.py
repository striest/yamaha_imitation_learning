import torch
import numpy as np
import matplotlib.pyplot as plt

from maxent_irl_costmaps.dataset.maxent_irl_dataset import MaxEntIRLDataset

from yamaha_imitation_learning.action_clustering.splines import ActionSpline, get_vel_steer
from yamaha_imitation_learning.feature_extraction.pretrained_resnet import PretrainedResnetFeatureExtractor

class ImageActionDataset:
    """
    Dataset containing pairs of (image features, action spline fit)
    """
    def __init__(self, dataset, img_extractor, action_extractor, device='cpu'):
        """
        Args:
            dataset: The dataset containing the images/actions to extract
            img_extractor: The extractor for images
            action_extractor: The extractor for action sequences
        """
        self.dataset = dataset
        self.img_extractor = img_extractor
        self.action_extractor = action_extractor

        self.image_embeddings = torch.zeros(0)
        self.action_embeddings = torch.zeros(0)
        self.actions = torch.zeros(0)

        self.device = device

    def process_dataset(self):
        image_embeddings = []
        action_embeddings = []
        actions = [] #store the action splines for viz

        print('preprocessing...')
        for i, batch in enumerate(self.dataset):
            print('{}/{}'.format(i, len(self.dataset)), end='\r')
            img = batch['image']
            acts = get_vel_steer(batch)
            with torch.no_grad():
                img_embedding = self.img_extractor.get_features(img)
                action_embedding, act_seq = self.action_extractor.fit_regularized_spline(acts)

            image_embeddings.append(img_embedding)
            action_embeddings.append(action_embedding)
            actions.append(act_seq)

            if i >= len(self.dataset)-1:
                break

        self.image_embeddings = torch.stack(image_embeddings, dim=0) #[B x Ni]
        self.action_embeddings = torch.stack(action_embeddings, dim=0) #[B x Na x 2]
        self.actions = torch.stack(actions, dim=0) #[B x T x 2]

    def get_knn(self, query, k=5):
        img = query['image']
        img_embedding = self.img_extractor.get_features(img)
        dists = torch.linalg.norm(
            self.image_embeddings - img_embedding.view(1, -1),
        dim=-1)

        print('mindist = {:.4f}, maxdist = {:.4f}'.format(dists.min(), dists.max()))

        mindists, idxs = torch.topk(dists, largest=False, k=k)
        return self.image_embeddings[idxs], idxs

    def get_rand(self, query, k=5):
        idxs = torch.randint(len(self.dataset)-1, (k,))
        return self.image_embeddings[idxs], idxs

    def to(self, device):
        self.device = device
        self.dataset = self.dataset.to(device)
        self.img_extractor = self.img_extractor.to(device)
        self.action_extractor = self.action_extractor.to(device)

        self.image_embedings = self.image_embeddings.to(device)
        self.action_embedings = self.action_embeddings.to(device)
        self.actions = self.actions.to(device)

        return self

if __name__ == '__main__':
    torch.set_printoptions(sci_mode=False)
    device = 'cuda'

    train_bag_fp = '/home/atv/Desktop/datasets/yamaha_maxent_irl/big_gridmaps/rosbags_train'
    train_preprocess_fp = '/home/atv/Desktop/datasets/yamaha_maxent_irl/big_gridmaps/torch_train_h75'
    train_dataset = MaxEntIRLDataset(bag_fp=train_bag_fp, preprocess_fp=train_preprocess_fp)

    net = torch.load('../pretraining/resnet_byol_25000.pt')
#    net = torch.load('../pretraining/resnet_base.pt')
    img_feature_extractor = PretrainedResnetFeatureExtractor(D=64, C=3, project=False, net=net)
    action_feature_extractor = ActionSpline(order=7)

    embedding_dataset = ImageActionDataset(train_dataset, img_feature_extractor, action_feature_extractor).to(device)
    embedding_dataset.process_dataset()

    test_bag_fp = '/home/atv/Desktop/datasets/yamaha_maxent_irl/big_gridmaps/rosbags_test'
    test_preprocess_fp = '/home/atv/Desktop/datasets/yamaha_maxent_irl/big_gridmaps/torch_test_h75'
    test_dataset = MaxEntIRLDataset(bag_fp=test_bag_fp, preprocess_fp=test_preprocess_fp).to(device)

    k = 5

    for i in range(100):
        idx = np.random.choice(np.arange(len(test_dataset)))
        dpt = test_dataset[idx]
        dpt_img = dpt['image']
        dpt_acts = get_vel_steer(dpt)

        #normalize actions for plotting
        dpt_acts[..., 0] /= 15.
        dpt_acts[..., 1] += 0.5
        dpt_acts[..., 1] *= 0.5

        knn_feats, knn_idxs = embedding_dataset.get_knn(dpt, k=k)
        rand_feats, rand_idxs = embedding_dataset.get_rand(dpt, k=k)

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

        query_axs[0].imshow(dpt['image'].permute(1, 2, 0)[:, :, [2, 1, 0]].cpu())
        query_axs[1].plot(dpt_acts[:, 0].cpu(), label='vel')
        query_axs[1].plot(dpt_acts[:, 1].cpu(), label='steer')

        query_axs[0].set_title('query image')
        query_axs[1].set_title('actions')

        for i, knn_idx in enumerate(knn_idxs):
            knn_pt = train_dataset[knn_idx]
            knn_img = knn_pt['image']
            knn_acts = get_vel_steer(knn_pt)

            #normalize actions for plotting
            knn_acts[..., 0] /= 15.
            knn_acts[..., 1] += 0.5
            knn_acts[..., 1] *= 0.5

            knn_img_axs[i].imshow(knn_img.permute(1, 2, 0)[:, :, [2, 1, 0]].cpu())
            knn_act_axs[i].plot(knn_acts[:, 0].cpu(), label='vel')
            knn_act_axs[i].plot(knn_acts[:, 1].cpu(), label='steer')

            knn_img_axs[i].set_title('knn image {}'.format(i+1))
            knn_act_axs[i].set_title('actions')

        for i, rand_idx in enumerate(rand_idxs):
            rand_pt = train_dataset[rand_idx]
            rand_img = rand_pt['image']
            rand_acts = get_vel_steer(rand_pt)

            #normalize actions for plotting
            rand_acts[..., 0] /= 15.
            rand_acts[..., 1] += 0.5
            rand_acts[..., 1] *= 0.5

            rand_img_axs[i].imshow(rand_img.permute(1, 2, 0)[:, :, [2, 1, 0]].cpu())
            rand_act_axs[i].plot(rand_acts[:, 0].cpu(), label='vel')
            rand_act_axs[i].plot(rand_acts[:, 1].cpu(), label='steer')

            rand_img_axs[i].set_title('rand image {}'.format(i+1))
            rand_act_axs[i].set_title('actions')

        for ax in axs:
            ax.legend()

        for ax in [query_axs[1]] + knn_act_axs + rand_act_axs:
            ax.set_ylim(0., 1.)

        plt.show()
                
