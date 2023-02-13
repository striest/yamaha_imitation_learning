"""
Generate trajectory samples that require terrain and bc understanding
Do so by:
    1. Generating a KBM trajectory via a decreasing velocity and some sinusoidal steering
    2. Build a feature map around this by computing distance to the traj as a feature
    3. Return the traj, map data and map metadata (also do this in batch)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

from torch_mpc.models.steer_setpoint_throttle_kbm import SteerSetpointThrottleKBM

def generate_samples(throttle_dist, steer_dist, model, x0, H, map_metadata, n=12):
    ## first generate control signals ##
    throttles = throttle_dist.sample((n, H))
    amps = steer_dist['amp_dist'].sample((n, 1))
    freqs = steer_dist['freq_dist'].sample((n, 1))
    t = torch.linspace(0., 2*np.pi, H).view(1, H).to(amps.device)
    steers = amps * (freqs * t).sin()
    cmds = torch.stack([throttles, steers], dim=-1)

    ## next make trajectories by rolling out through the model ##
    x0 = torch.stack([x0] * n, dim=0)

    trajs = model.rollout(x0, cmds)

    ## now make feature maps based on distance to traj ##
    xmin = map_metadata['origin'][0]
    xmax = xmin + map_metadata['width']
    ymin = map_metadata['origin'][1]
    ymax = ymin + map_metadata['height']
    nx = int(map_metadata['width']/map_metadata['resolution'])
    ny = int(map_metadata['height']/map_metadata['resolution'])
    rad = (map_metadata['width']**2 + map_metadata['height']**2) ** 0.5

    xs = torch.linspace(xmin, xmax, nx).to(trajs.device)
    ys = torch.linspace(ymin, ymax, ny).to(trajs.device)
    xs, ys = torch.meshgrid(xs, ys, indexing='xy')

    xs = torch.stack([xs] * n, dim=0).view(n, nx, ny, 1)
    ys = torch.stack([ys] * n, dim=0).view(n, nx, ny, 1)

    txs = trajs[..., 0].view(n, 1, 1, H)
    tys = trajs[..., 1].view(n, 1, 1, H)

    dists = torch.hypot(txs-xs, tys-ys) 
    mindists = dists.min(dim=-1)[0]

    feats = (mindists > (rad/15.)).float().unsqueeze(1)

    return {
        'cmd': cmds.unsqueeze(1),
        'traj': trajs.unsqueeze(1),
        'map_features': feats,
        'metadata': [map_metadata]* n
    }

if __name__ == '__main__':
    device = 'cuda'

    model = SteerSetpointThrottleKBM(L=3.0, throttle_lim=[0.0, 1.0], steer_lim=[-0.52, 0.52], steer_rate_lim=0.3, k_throttle=1.5, k_fric=0.00, k_drag=0.1, dt=0.1, w_Kp=10.).to(device)

    throttle_dist = torch.distributions.Normal(loc=torch.tensor(0.01).to(device), scale=torch.tensor(1e-10).to(device))
    steer_dist = {
        'amp_dist': torch.distributions.Normal(loc=torch.tensor(0.5).to(device), scale=torch.tensor(0.1).to(device)),
        'freq_dist': torch.distributions.Normal(loc=torch.tensor(1.0).to(device), scale=torch.tensor(1e-4).to(device))
    }

    x0 = torch.tensor([0., 0., 0., 5., 0.]).to(device)
    H = 100

    map_metadata = {
        'width': 100.,
        'height': 100.,
        'resolution': 0.25,
        'origin': torch.tensor([-50., -50.]).to(device)
    }

    samples = generate_samples(throttle_dist, steer_dist, model, x0, H, map_metadata)

    ## plot ##
    for i in range(12):
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        axs[0].plot(samples['cmd'][i, :, 0].cpu())
        axs[1].plot(samples['cmd'][i, :, 1].cpu())
        axs[2].plot(samples['traj'][i, :, 0].cpu(), samples['traj'][i, :, 1].cpu())

        xmin = samples['metadata'][i]['origin'][0].item()
        xmax = xmin + samples['metadata'][i]['width']
        ymin = samples['metadata'][i]['origin'][1].item()
        ymax = ymin + samples['metadata'][i]['height']

        axs[2].imshow(samples['map_features'][i, 0].cpu(), origin='lower', extent=(xmin, xmax, ymin, ymax))
        plt.show()
