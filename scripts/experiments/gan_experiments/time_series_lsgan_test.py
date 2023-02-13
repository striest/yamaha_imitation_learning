import torch
import numpy as np
import matplotlib.pyplot as plt

from numpy import pi
from torch import sin, cos, tan

from maxent_irl_costmaps.networks.mlp import MLP

def sample_time_series(amp_dist, freq_dist, n=100, d=50):
    amps = amp_dist.sample((n, )).view(n, 1)
    freqs = freq_dist.sample((n, )).view(n, 1)

    t = torch.linspace(0, 2*pi, d).view(1, d)

    samples = -amps * (freqs * t).sin()
    return samples

class GRUGenerator(torch.nn.Module):
    """
    implement a time-series generator as a GRU
    """
    def __init__(self, insize=256, outsize=1, encoder_hiddens=[128, 128], decoder_hiddens=[16, ], rnn_hidden_dim=128, rnn_layers=1, activation=torch.nn.ReLU):
        super(GRUGenerator, self).__init__()

        self.insize = insize
        self.outsize = outsize
        self.rnn_layers = rnn_layers

        # 10 for state w.o. position + 1 for steer
        self.encoder = MLP(insize, rnn_hidden_dim, encoder_hiddens, hidden_activation=activation)
        self.rnn = torch.nn.GRU(outsize, rnn_hidden_dim, batch_first=True, num_layers=rnn_layers)
        self.decoder = MLP(rnn_hidden_dim, outsize, decoder_hiddens, hidden_activation=activation)

    def forward(self, x):
        h = self.encoder.forward(x)
        _h = torch.stack(self.rnn_layers * [h], dim=0)
        res = [torch.zeros(x.shape[0], 1, self.outsize, device=x.device)]
        for i in range(50):
            _x, _h = self.rnn.forward(res[-1], _h)
            _x = self.decoder(_x) 
            res.append(_x)
        return torch.cat(res, dim=1).squeeze(-1)[..., :-1]

class GRUDiscriminator(torch.nn.Module):
    """
    implement a time-series discriminator as a GRU
    """
    def __init__(self, insize=256, outsize=1, encoder_hiddens=[128, 128], decoder_hiddens=[16, ], rnn_hidden_dim=64, rnn_layers=2, activation=torch.nn.ReLU):
        super(GRUDiscriminator, self).__init__()

        self.insize = insize
        self.outsize = outsize
        self.rnn_layers = rnn_layers

        # 10 for state w.o. position + 1 for steer
        self.encoder = MLP(insize, rnn_hidden_dim, encoder_hiddens, hidden_activation=activation)
        self.rnn = torch.nn.GRU(outsize, rnn_hidden_dim, batch_first=True, num_layers=rnn_layers)
        self.decoder = MLP(rnn_hidden_dim, outsize, decoder_hiddens, hidden_activation=activation)

    def forward(self, x):
        h = self.encoder.forward(x)
        _h = torch.stack(self.rnn_layers * [h], dim=0)
        res = [torch.zeros(x.shape[0], 1, self.outsize)]
        for i in range(100):
            _x, _h = self.rnn.forward(res[-1], _h)
            _x = self.decoder(_x) 
            res.append(_x)
        return torch.cat(_x, dim=-1)

if __name__ == '__main__':
    amp_dist = torch.distributions.Normal(loc=0.5, scale=0.1)
    freq_dist = torch.distributions.Normal(loc=1.0, scale=1e-4)

#    samples = make_time_series(amp_dist, freq_dist)

#    plt.plot(samples.T)
#    plt.show()

    zn = 256
    generator = GRUGenerator().cuda()
    generator_opt = torch.optim.Adam(generator.parameters())

    nf = 20
    nd = 1
    discriminators = []
    for i in range(nd):
        discriminator = MLP(insize=50 * 2 * nf, outsize=1, hiddens=[128, 128], hidden_activation=torch.nn.ReLU).cuda()
        discriminator_opt = torch.optim.Adam(discriminator.parameters())
        discriminators.append([discriminator, discriminator_opt])

    ## idk fourier features I guess ##
    fa = torch.randn(nf).view(1, 1, nf).cuda()
    fb = torch.randn(nf).view(1, 1, nf).cuda()

    n_steps = 50000
    batch_size = 512
    for i in range(n_steps+1):
        z = torch.randn(batch_size, zn).cuda()
        fake_samples = generator.forward(z)

        real_samples = sample_time_series(amp_dist, freq_dist, batch_size).cuda()

        fake_samples_f1 = fa * (fb * fake_samples.unsqueeze(-1)).sin()
        fake_samples_f2 = fa * (fb * fake_samples.unsqueeze(-1)).cos()
        fake_samples2 = torch.stack([fake_samples_f1, fake_samples_f2], dim=-1).view(batch_size, -1)

        real_samples_f1 = fa * (fb * real_samples.unsqueeze(-1)).sin()
        real_samples_f2 = fa * (fb * real_samples.unsqueeze(-1)).cos()
        real_samples2 = torch.stack([real_samples_f1, real_samples_f2], dim=-1).view(batch_size, -1)

        idx = np.random.randint(nd)
        discriminator, discriminator_opt = discriminators[idx]

        fake_sample_logits = discriminator.forward(fake_samples2)
        real_sample_logits = discriminator.forward(real_samples2)

        ## discriminator update is min[(D(x) - 1)^2 + (D(G(z)) + 1)^2]##
        if i % 2 == 0:
            disc_objective = (real_sample_logits - 1.).pow(2) + (fake_sample_logits + 0.).pow(2)
            loss = disc_objective.mean()
            discriminator_opt.zero_grad()
            loss.backward()
            discriminator_opt.step()
        ## generator update is min[D(G(z))^2]##
        else:
            gen_objective = (fake_sample_logits - 1.).pow(2)
            loss = gen_objective.mean()
            generator_opt.zero_grad()
            loss.backward()
            generator_opt.step()

        ## log ##
        print('{}/{}'.format(i, n_steps))
        print('Avg. Disc. fake scores: {:.4f}'.format(fake_sample_logits.mean()))
        print('Avg. Disc. real scores: {:.4f}'.format(real_sample_logits.mean()))

        ## visualize ##
        if i % 10000 == 0:
            with torch.no_grad():
                fig, axs = plt.subplots(1, 2, figsize=(16, 8))
                axs[0].plot(real_samples.detach().cpu().T, label='real', c='b', alpha=0.3)
                axs[1].plot(fake_samples.detach().cpu().T, label='fake', c='r', alpha=0.3)
                plt.show()
