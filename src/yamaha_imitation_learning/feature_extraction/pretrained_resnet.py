import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models

from maxent_irl_costmaps.dataset.maxent_irl_dataset import MaxEntIRLDataset

class PretrainedResnetFeatureExtractor:
    """
    Get features from images/lidar maps via a pretrained resnet
    """
    def __init__(self, D=32, C=3, project=True): 
        """
        Args:
            D: The dimension of the output embedding
            C: The number of input channels
            project: If false, don't project output and just use net output
        """
        net = models.resnet50(pretrained=True)
        modules = list(net.children())[:-1]

        self.net = torch.nn.Sequential(*modules)
        self.net_D = self.net.forward(torch.rand(1, 3, 244, 244)).shape[-3]
        self.D = D if project else self.net_D

        self.C = C
        self.inp_proj = torch.nn.Linear(self.C, 3) if C != 3 else torch.nn.Identity()

        self.project = project
        self.proj = torch.nn.Linear(self.net_D, D)

    def get_features(self, img):
        with torch.no_grad():
            _x = img.unsqueeze(0)
            _x = self.inp_proj.forward(_x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            _x = self.net.forward(_x)[:, :, 0, 0]
            if self.project:
                _x = self.proj.forward(_x)

        return _x[0]

if __name__ == '__main__':
    torch.set_printoptions(sci_mode=False)

    bag_fp = '/home/striest/Desktop/datasets/yamaha_maxent_irl/big_gridmaps/rosbags_test'
    preprocess_fp = '/home/striest/Desktop/datasets/yamaha_maxent_irl/big_gridmaps/torch_test_h75'

    dataset = MaxEntIRLDataset(bag_fp=bag_fp, preprocess_fp=preprocess_fp)

    img_feature_extractor = PretrainedResnetFeatureExtractor(D=64, C=3, project=True)
    map_feature_extractor = PretrainedResnetFeatureExtractor(D=64, C=len(dataset.feature_keys), project=True)

    fig, axs = plt.subplots(2, 2, figsize=(12, 12))
    plt.show(block=False)

    img_feats_hist = []
    map_feats_hist = []
    hlen = 100
    se = 10

    for i in range(len(dataset)-1):
        print('{}/{}'.format(i, len(dataset)-1), end='\r')

        img = dataset[i]['image']
        map_feats = dataset[i]['map_features']
        img_feats = img_feature_extractor.get_features(img)
        map_feats_feats = map_feature_extractor.get_features(map_feats)

        if i % se == 0:
            map_feats_hist.append(map_feats_feats)
            if len(map_feats_hist) > hlen:
                map_feats_hist = map_feats_hist[1:]

            img_feats_hist.append(img_feats)
            if len(img_feats_hist) > hlen:
                img_feats_hist = img_feats_hist[1:]

        for ax in axs.flatten():
            ax.cla()

        axs[0, 1].set_ylim(-1., 1.)
        axs[1, 1].set_ylim(-1., 1.)

        axs[0, 0].imshow(img.permute(1, 2, 0)[..., [2, 1, 0]])
        axs[1, 0].imshow(map_feats[3])

        axs[0, 0].set_title('{}/{}'.format(i, len(dataset)-1))
        axs[0, 1].set_title('image features')
        axs[1, 1].set_title('map features')

        for j, feat in enumerate(img_feats_hist):
            alpha = j / len(img_feats_hist)
            axs[0, 1].plot(torch.arange(len(feat)), feat, marker='.', c='r', alpha = alpha)

        for j, feat in enumerate(map_feats_hist):
            alpha = j / len(map_feats_hist)
            axs[1, 1].plot(torch.arange(len(feat)), feat, marker='.', c='r', alpha = alpha)

        axs[0, 1].plot(torch.arange(len(img_feats)), img_feats, marker='.', c='b', alpha = 1.0)
        axs[1, 1].plot(torch.arange(len(map_feats_feats)), map_feats_feats, marker='.', c='b', alpha = 1.0)

        plt.pause(1e-1)
