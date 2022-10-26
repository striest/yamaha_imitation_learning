import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn

from yamaha_imitation_learning.action_clustering.splines import ActionSpline, get_vel_steer
from yamaha_imitation_learning.feature_extraction.pretrained_resnet import PretrainedResnetFeatureExtractor

class ResnetSplinePredictor(nn.Module):
    """
    Take in images and produce spline parameters
    """
    def __init__(self, net=None, insize=(3, 244, 244), outsize=(2, 10), device='cpu'):
        """
        Args:
            img_extractor: the resnet backbone that takes in images and outputs a latent representation
            outsize: The number of spline parameters to predict
        """
        super(ResnetSplinePredictor, self).__init__()
        self.insize = insize
        self.outsize = outsize

        if net is None:
            net = models.resnet50(pretrained=True)
            modules = list(net.children())[:-1]
            self.net = torch.nn.Sequential(*modules)
        else:
            modules = list(net.children())[:-1]
            self.net = torch.nn.Sequential(*modules)

        self.net_outsize = self.net.forward(torch.rand(1, *insize)).shape[-3]
        self.proj = torch.nn.Linear(self.net_outsize, np.prod(outsize))
        self.device = device

    def forward(self, x, freeze_backbone=False):
        _x = x
        if freeze_backbone:
            with torch.no_grad():
                _x = self.net.forward(_x)
        else:
            _x = self.net.forward(_x)[..., 0, 0] #image -> vec

        _x = self.proj.forward(_x)
        return _x.view(-1, *self.outsize)

    def to(self, device):
        self.device = device
        self.net = self.net.to(device)
        self.proj = self.proj.to(device)
        return self
