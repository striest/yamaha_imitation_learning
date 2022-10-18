import torch
import numpy as np
import matplotlib.pyplot as plt

from maxent_irl_costmaps.dataset.maxent_irl_dataset import MaxEntIRLDataset

from yamaha_imitation_learning.action_clustering.splines import ActionSpline, get_vel_steer
from yamaha_imitation_learning.feature_extraction.pretrained_resnet import PretrainedResnetFeatureExtractor
from yamaha_imitation_learning.datasets.image_action_dataset import ImageActionDataset

class BCTrainer:
    """
    Train behavioral cloning from a pretrained resnet. Probably just need a linear set of weights at the end
    """
    def __init__(self, dataset, img_feature_extractor, action_feature_extractor, freeze_net=True):
        self.img_extractor = img_feature_extractor
        self.action_extractor = action_feature_extractor
        self.proj = torch.nn.Linear(self.img_extractor.D, 2*(self.action_extractor.order+1))
        self.opt = torch.optim.Adam(self.proj.parameters())

if __name__ == '__main__':
    torch.set_printoptions(sci_mode=False)
    device = 'cuda'

    train_bag_fp = '/home/atv/Desktop/datasets/yamaha_maxent_irl/big_gridmaps/rosbags_debug'
    train_preprocess_fp = '/home/atv/Desktop/datasets/yamaha_maxent_irl/big_gridmaps/torch_debug'
    train_dataset = MaxEntIRLDataset(bag_fp=train_bag_fp, preprocess_fp=train_preprocess_fp)

    net = torch.load('../pretraining/resnet_base.pt')
    img_feature_extractor = PretrainedResnetFeatureExtractor(D=64, C=3, project=False, net=net)
    action_feature_extractor = ActionSpline(order=9, lam=1e-3)

    test_bag_fp = '/home/atv/Desktop/datasets/yamaha_maxent_irl/big_gridmaps/rosbags_test'
    test_preprocess_fp = '/home/atv/Desktop/datasets/yamaha_maxent_irl/big_gridmaps/torch_test_h75'
    test_dataset = MaxEntIRLDataset(bag_fp=test_bag_fp, preprocess_fp=test_preprocess_fp).to(device)

    import pdb;pdb.set_trace()
    bc_trainer = BCTrainer(train_dataset, img_feature_extractor, action_feature_extractor)
