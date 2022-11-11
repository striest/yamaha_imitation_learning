import torch
import numpy as np
import matplotlib.pyplot as plt

from torch.nn import functional as F
from torchvision import models
from torchvision import transforms as T
from byol_pytorch import BYOL

from maxent_irl_costmaps.dataset.maxent_irl_dataset import MaxEntIRLDataset

from yamaha_imitation_learning.action_clustering.splines import ActionSpline, get_vel_steer
from yamaha_imitation_learning.feature_extraction.pretrained_resnet import PretrainedResnetFeatureExtractor

#some code borrowed directly from: https://github.com/lucidrains/byol-pytorch/blob/master/byol_pytorch/byol_pytorch.py

# augmentation utils

class RandomApply(torch.nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p
    def forward(self, x):
        if torch.rand(1,).item() > self.p:
            return x
        return self.fn(x)

class BYOLTrainer:
    """
    Fine-tune a network using BYOL
    """
    def __init__(self, dataset, batch_size=32, use_flip=True):
        self.dataset = dataset.to('cuda')
        self.batch_size = batch_size

        image_size = self.dataset[0]['image'].shape[-1]

        if use_flip:
            aug = torch.nn.Sequential(
            RandomApply(
                T.ColorJitter(0.8, 0.8, 0.8, 0.2),
                p = 0.3
            ),
            T.RandomGrayscale(p=0.2),
            T.RandomHorizontalFlip(),
            RandomApply(
                T.GaussianBlur((3, 3), (1.0, 2.0)),
                p = 0.2
            ),
            T.RandomResizedCrop((image_size, image_size)),
            T.Normalize(
                mean=torch.tensor([0.485, 0.456, 0.406]),
                std=torch.tensor([0.229, 0.224, 0.225])),
            )
        else:
            aug = torch.nn.Sequential(
            RandomApply(
                T.ColorJitter(0.8, 0.8, 0.8, 0.2),
                p = 0.3
            ),
            T.RandomGrayscale(p=0.2),
            RandomApply(
                T.GaussianBlur((3, 3), (1.0, 2.0)),
                p = 0.2
            ),
            T.RandomResizedCrop((image_size, image_size)),
            T.Normalize(
                mean=torch.tensor([0.485, 0.456, 0.406]),
                std=torch.tensor([0.229, 0.224, 0.225])),
            )

        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).to('cuda')
        self.byol = BYOL(
            self.resnet,
            image_size = image_size,
            hidden_layer = 'avgpool',
            augment_fn = aug
        )
        self.opt = torch.optim.Adam(self.byol.parameters(), lr=3e-4)

    def train(self, itrs):
        for i in range(itrs):
            print('{}/{}'.format(i+1, itrs), end='\r')
            self.step()

    def step(self):
        images = self.get_samples()
        loss = self.byol(images)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

    def get_samples(self):
        idxs = torch.randint(len(self.dataset)-1, (self.batch_size, ))
        return torch.stack([self.dataset[i]['image'] for i in idxs], dim=0)

if __name__ == '__main__':
    train_bag_fp = '/home/atv/Desktop/datasets/yamaha_maxent_irl/big_gridmaps/rosbags_train'
    train_preprocess_fp = '/home/atv/Desktop/datasets/yamaha_maxent_irl/big_gridmaps/torch_train_h75'
    train_dataset = MaxEntIRLDataset(bag_fp=train_bag_fp, preprocess_fp=train_preprocess_fp)

    itrs = 25000
    trainer = BYOLTrainer(train_dataset, use_flip=True)
    trainer.train(itrs = itrs)
    torch.save(trainer.resnet.cpu(), 'resnet_byol_{}.pt'.format(itrs))
