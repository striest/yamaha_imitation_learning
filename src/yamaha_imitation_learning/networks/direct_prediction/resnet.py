import torch
import numpy as np
import matplotlib.pyplot as plt

from torch import nn
from torchvision.models import resnet50, ResNet50_Weights

from maxent_irl_costmaps.networks.mlp import MLP

class ResnetRNNFPV(nn.Module):
    """
    Same as the regular resnet but be smarter and use an RNN to predict actions over times
    """
    def __init__(self, net=None, insize=(3, 244, 244), outsize=(75, 2), nf=10, mlp_hiddens=[512, ], rnn_hidden_dim=64, rnn_layers=2, freeze_backbone=False, device='cpu'):
        super(ResnetRNNFPV, self).__init__()
        self.insize = insize
        self.outsize = outsize
        self.nf = nf
        self.rnn_layers = rnn_layers

        if net is None:
            net = resnet50(weights=ResNet50_Weights.DEFAULT)
            modules = list(net.children())[:-1]
            self.cnn = torch.nn.Sequential(*modules)
        else:
            modules = list(net.children())[:-1]
            self.cnn = torch.nn.Sequential(*modules)

        self.net_outsize = self.cnn.forward(torch.rand(1, *insize)).shape[-3]

        self.mlp_encoder = MLP(self.net_outsize + 2 * 4 * nf, rnn_hidden_dim, mlp_hiddens)

        self.fa = torch.randn(size=(nf, ))
        self.fb = torch.randn(size=(nf, ))

        self.freeze_backbone = freeze_backbone

        self.rnn = torch.nn.GRU(self.outsize[-1], rnn_hidden_dim, batch_first=True, num_layers=rnn_layers)
        self.decoder = MLP(rnn_hidden_dim, self.outsize[-1], [16, ])

        self.device = device

    def forward(self, x):
        goal = x['goal']
        state = x['dynamics']
        img = x['image']
        a0 = x['cmd'][:, 0]
        if self.freeze_backbone:
            with torch.no_grad():
                _img = self.cnn.forward(img)[..., 0, 0]
        else:
            _img = self.cnn.forward(img)[..., 0, 0] #image -> vec

        vec = torch.cat([state, goal], dim=-1)
        fvec = self.fourier(vec)

        _h = self.mlp_encoder(torch.cat([_img, fvec], dim=-1))
        _h = torch.stack(self.rnn_layers * [_h], dim=0)

        res = [a0]
        for _ in range(self.outsize[0]):
            _x, _h = self.rnn.forward(res[-1].unsqueeze(1), _h)
            act = self.decoder.forward(_x).squeeze(1).tanh()
            res.append(act)

        acts = torch.stack(res, dim=1)
        return acts[:, :-1]

    def fourier(self, x):
        temp = 2 * np.pi * self.fb.view(1, 1, -1) * x.unsqueeze(-1)
        out1 = self.fa * temp.sin()
        out2 = self.fa * temp.cos()

        return torch.cat([out1, out2], dim=-1).view(x.shape[0], -1)

    def to(self, device):
        self.device = device
        self.cnn = self.cnn.to(device)
        self.mlp_encoder = self.mlp_encoder.to(device)
        self.rnn = self.rnn.to(device)
        self.decoder = self.decoder.to(device)
        self.fa = self.fa.to(device)
        self.fb = self.fb.to(device)
        return self

class ResnetFPV(nn.Module):
    """
    Take in images and produce spline parameters
    """
    def __init__(self, net=None, insize=(3, 244, 244), outsize=(75, 2), nf = 10, mlp_hiddens=[512, ], freeze_backbone=False, device='cpu'):
        """
        Args:
            img_extractor: the resnet backbone that takes in images and outputs a latent representation
            outsize: The number of spline parameters to predict
            nf: number of Fourier expansions of the vec inputs
        """
        super(ResnetFPV, self).__init__()
        self.insize = insize
        self.outsize = outsize
        self.nf = nf

        if net is None:
            net = resnet50(weights=ResNet50_Weights.DEFAULT)
            modules = list(net.children())[:-1]
            self.net = torch.nn.Sequential(*modules)
        else:
            modules = list(net.children())[:-1]
            self.net = torch.nn.Sequential(*modules)

        self.net_outsize = self.net.forward(torch.rand(1, *insize)).shape[-3]

        self.mlp = MLP(self.net_outsize + 2 * 4 * nf, int(np.prod(outsize)), mlp_hiddens)

        self.fa = torch.randn(size=(nf, ))
        self.fb = torch.randn(size=(nf, ))

        self.freeze_backbone = freeze_backbone

        self.device = device

    def forward(self, x, freeze_backbone=False):
        """
        grab the goal, state, and image
        """
        goal = x['goal']
        state = x['dynamics']
        img = x['image']
        if self.freeze_backbone:
            with torch.no_grad():
                _img = self.net.forward(img)[..., 0, 0]
        else:
            _img = self.net.forward(img)[..., 0, 0] #image -> vec

        vec = torch.cat([state, goal], dim=-1)
        fvec = self.fourier(vec)

        mlp_in = torch.cat([_img, fvec], dim=-1)
        mlp_out = self.mlp.forward(mlp_in)

        return mlp_out.view(-1, *self.outsize).tanh()

    def fourier(self, x):
        temp = 2 * np.pi * self.fb.view(1, 1, -1) * x.unsqueeze(-1)
        out1 = self.fa * temp.sin()
        out2 = self.fa * temp.cos()

        return torch.cat([out1, out2], dim=-1).view(x.shape[0], -1)

    def to(self, device):
        self.device = device
        self.net = self.net.to(device)
        self.mlp = self.mlp.to(device)
        self.fa = self.fa.to(device)
        self.fb = self.fb.to(device)
        return self
