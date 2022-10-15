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
    def __init__(self, dataset, img_extractor, action_extractor):
        """
        Args:
            dataset: The dataset containing the images/actions to extract
            img_extractor: The extractor for images
            action_extractor: The extractor for action sequences
        """
        self.dataset = dataset
        self.img_extractor = img_extractor
        self.action_extractor = action_extractor

        self.process_dataset()

    def process_dataset(self):
        image_embeddings = []
        action_embeddings = []
        actions = [] #store the action splines for viz

        print('preprocessing...')
        for i, batch in enumerate(self.dataset):
            print('{}/{}'.format(i, len(self.dataset)), end='\r')
            img = batch['image']
            acts = get_vel_steer(batch)
            img_embedding = self.img_extractor.get_features(img)
            action_embedding, act_seq = self.action_extractor.fit_spline(acts)

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

        mindists, idxs = torch.topk(dists, largest=False, k=k)
        return self.image_embeddings[idxs], idxs

if __name__ == '__main__':
    torch.set_printoptions(sci_mode=False)

    train_bag_fp = '/home/striest/Desktop/datasets/yamaha_maxent_irl/big_gridmaps/rosbags_train'
    train_preprocess_fp = '/home/striest/Desktop/datasets/yamaha_maxent_irl/big_gridmaps/torch_train_h75'
    train_dataset = MaxEntIRLDataset(bag_fp=train_bag_fp, preprocess_fp=train_preprocess_fp)

    img_feature_extractor = PretrainedResnetFeatureExtractor(D=64, C=3, project=True)
    action_feature_extractor = ActionSpline(order=7)

    embedding_dataset = ImageActionDataset(train_dataset, img_feature_extractor, action_feature_extractor)

    test_bag_fp = '/home/striest/Desktop/datasets/yamaha_maxent_irl/big_gridmaps/rosbags_test'
    test_preprocess_fp = '/home/striest/Desktop/datasets/yamaha_maxent_irl/big_gridmaps/torch_test_h75'
    test_dataset = MaxEntIRLDataset(bag_fp=test_bag_fp, preprocess_fp=test_preprocess_fp)

    k = 5

    for i in range(100):
        idx = np.random.choice(np.arange(len(test_dataset)))
        dpt = test_dataset[idx]
        dpt_img = dpt['image']
        dpt_acts = get_vel_steer(dpt)
        dpt_embedding, dpt_smoothed_acts = embedding_dataset.action_extractor.fit_spline(dpt_acts)
        knn_feats, knn_idxs = embedding_dataset.get_knn(dpt, k=k)

        fig, axs = plt.subplots(3, k+1, figsize=((k+1) * 4, 12))
        axs1 = axs[:, 0]
        axs2 = axs[:, 1:]

        axs1[0].imshow(dpt['image'].permute(1, 2, 0)[:, :, [2, 1, 0]])
        axs1[1].plot(dpt_acts[:, 0], label='vel')
        axs1[1].plot(dpt_acts[:, 1], label='steer')
        axs1[2].plot(dpt_smoothed_acts[:, 0], label='vel')
        axs1[2].plot(dpt_smoothed_acts[:, 1], label='steer')

        axs1[0].set_title('query image')
        axs1[1].set_title('actions')
        axs1[2].set_title('embedding')

        for i, knn_idx in enumerate(knn_idxs):
            knn_pt = train_dataset[knn_idx]
            knn_img = knn_pt['image']
            knn_acts = get_vel_steer(knn_pt)
            knn_act_embedding, knn_smoothed_acts = embedding_dataset.action_extractor.fit_spline(knn_acts)

            axs2[0, i].imshow(knn_img.permute(1, 2, 0)[:, :, [2, 1, 0]])
            axs2[1, i].plot(knn_acts[:, 0], label='vel')
            axs2[1, i].plot(knn_acts[:, 1], label='steer')
            axs2[2, i].plot(knn_smoothed_acts[:, 0], label='vel')
            axs2[2, i].plot(knn_smoothed_acts[:, 1], label='steer')

            axs2[0, i].set_title('knn image {}'.format(i+1))
            axs2[1, i].set_title('actions')
            axs2[2, i].set_title('embedding')

        for ax in axs.flatten():
            ax.legend()

        plt.show()
                
