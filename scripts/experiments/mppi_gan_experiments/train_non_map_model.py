"""
Key diff is that we train a model directly on traj features
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

from maxent_irl_costmaps.networks.mlp import MLP

from torch_mpc.models.steer_setpoint_throttle_kbm import SteerSetpointThrottleKBM
from torch_mpc.algos.batch_mppi import BatchMPPI

from torch_mpc.cost_functions.generic_cost_function import CostFunction
from torch_mpc.cost_functions.cost_terms.base import CostTerm

#forward terms
from torch_mpc.cost_functions.cost_terms.costmap_projection import CostmapProjection
from torch_mpc.cost_functions.cost_terms.unknown_map_projection import UnknownMapProjection

#backward terms
from torch_mpc.cost_functions.cost_terms.euclidean_distance_to_goal import EuclideanDistanceToGoal
from torch_mpc.cost_functions.cost_terms.interpolated_cost_to_goal import InterpolatedCostToGoal

from make_gt_samples import *

def get_feature_counts(traj, map_features, map_metadata):
    """
    Given a (set) of trajectories and map features, compute the features of that trajectory
    """
    xs = traj[...,0]
    ys = traj[...,1]
    res = map_metadata['resolution']
    ox = map_metadata['origin'][0]
    oy = map_metadata['origin'][1]

    xidxs = ((xs - ox) / res).long()
    yidxs = ((ys - oy) / res).long()

    valid_mask = (xidxs >= 0) & (xidxs < map_features.shape[2]) & (yidxs >= 0) & (yidxs < map_features.shape[1])

    xidxs[~valid_mask] = 0
    yidxs[~valid_mask] = 0

    # map data is transposed
    features = map_features[:, yidxs, xidxs]

    return features.moveaxis(0, -1)

## need to make a cost model class ##
class EBMCost(CostTerm):
    """
    Get the NN to spit out costs for MPPI
    """
    def __init__(self, ebm, use_rew=False, device='cpu'):
        self.ebm = ebm
        self.use_rew = use_rew
        self.device = device

    def get_data_keys(self):
        return ['map_features', 'map_metadata']

    def cost(self, states, actions, data, traj_cost=True):
        dpt = {
            'traj': states,
            'cmd': actions,
            'map_features': data['map_features'],
            'metadata': data['map_metadata']
        }

        inp = self.make_training_input(dpt)

        with torch.no_grad():
            cost = 1.-self.ebm.forward(inp.flatten(start_dim=-2)).squeeze(-1)

        if self.use_rew:
            cost = -cost

        return cost.sum(dim=-1) if traj_cost else cost

    def to(self, device):
        self.device = device
        self.ebm = self.ebm.to(device)
        return self

    def __repr__(self):
        return 'EBM'

    def make_training_input(
        self,
        dpt,
        use_map_features = True,
        use_traj_features = False,
        use_act_features = False
    ):
        """
        Make the nn input to train on 
        """
        trajs = dpt['traj']
        cmds = dpt['cmd']
        map_feats = dpt['map_features']
        metadata = dpt['metadata']
        n = len(metadata)

        out = []
        for i, (traj, cmd, map_features, map_metadata) in enumerate(zip(trajs, cmds, map_feats, metadata)):
            res = torch.zeros(*traj.shape[:-1], 0).to(traj.device)

            if use_map_features:
                feats = get_feature_counts(traj, map_features, map_metadata)
                res = torch.cat([res, feats], dim=-1)

            if use_traj_features:
                rad = (map_metadata['width']**2 + map_metadata['height']**2) ** 0.5
                feats = traj.clone()
                feats[..., :2] /= rad
                feats[..., 3] /= 5.
                res = torch.cat([res, feats], dim=-1)

            if use_act_features:
                feats = cmd.clone()
                res = torch.cat([res, feats], dim=-1)

            out.append(res)
        out = torch.stack(out, dim=0)
        return out

class ShapedEBMCost(EBMCost):
    """
    Use the shaped rew from Peng et al. for lsgan:
        cost = -max(0, 1 - 0.25(D(x) - 1)^2)
    """
    def cost(self, states, actions, data, traj_cost=True):
        dpt = {
            'traj': states,
            'cmd': actions,
            'map_features': data['map_features'],
            'metadata': data['map_metadata']
        }

        inp = self.make_training_input(dpt)

        with torch.no_grad():
            rew = self.ebm.forward(inp.flatten(start_dim=-2)).squeeze(-1)

        shaped_rew = torch.maximum(torch.zeros_like(rew), 1 - 0.25 * (rew - 1) ** 2)
        cost = -shaped_rew

        return cost.sum(dim=-1) if traj_cost else cost

if __name__ == '__main__':
    import os
    device = 'cuda'
    exp_name = 'map_traj_wgan_map_only'
    os.mkdir(exp_name)

    model = SteerSetpointThrottleKBM(L=3.0, throttle_lim=[0.0, 1.0], steer_lim=[-0.52, 0.52], steer_rate_lim=0.3, k_throttle=1.5, k_fric=0.00, k_drag=0.1, dt=0.1, w_Kp=10.).to(device)

    throttle_dist = torch.distributions.Normal(loc=torch.tensor(0.3).to(device), scale=torch.tensor(1e-10).to(device))
    steer_dist = {
        'amp_dist': torch.distributions.Normal(loc=torch.tensor(0.0).to(device), scale=torch.tensor(0.2).to(device)),
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


    ## MPPI params ##
    mppi_params = {
        'sys_noise':torch.tensor([0.5, 0.5]),
        'temperature':0.02,
        'use_ou':True,
        'ou_alpha': 0.9,
        'ou_scale': 10.0,
        'd_ou_scale': 6.0
    }

    H = 100

    ## set up network ##
    # for now I'm starting with the dumb thing
    ebm = MLP(insize=H * 1, outsize=H, hiddens=[256, 128], hidden_activation=torch.nn.ReLU).to(device)
    ebm_opt = torch.optim.Adam(ebm.parameters())
    ebm_cost = EBMCost(ebm=ebm)
#    ebm_cost = ShapedEBMCost(ebm=ebm)

    cfn = CostFunction(
        cost_terms=[
            (1.0, EuclideanDistanceToGoal()),
            (100.0/H, ebm_cost)
#            (1.0/H, ebm_cost)
        ]
    )

    ## training params ##
    batch_size = 12
    train_steps = 1000
    mppi_itrs = 10

    mppi = BatchMPPI(model=model, cost_fn=cfn, num_samples=2048, num_timesteps=H, control_params=mppi_params, num_uniform_samples=0, batch_size=batch_size, num_safe_timesteps=0).to(device)

    for ti in range(train_steps):
        print('{}/{}'.format(ti + 1, train_steps))

        ## get "expert" demonstrations ##
        samples = generate_samples(throttle_dist, steer_dist, model, x0, H, map_metadata)

        ## solve with mppi ##
        mppi.reset()
        mppi.cost_fn.data['goals'] = [x[0, [-1], :2] for x in samples['traj']]
        mppi.cost_fn.data['map_features'] = samples['map_features']
        mppi.cost_fn.data['map_metadata'] = samples['metadata']

        X = torch.stack([x0] * batch_size, dim=0)

        for mi in range(mppi_itrs):
            u = mppi.get_control(X, step=False)

        ## set up learner input ##
        minidxs = mppi.costs.argmin(dim=-1)
        best_trajs = mppi.noisy_states[torch.arange(batch_size), minidxs].unsqueeze(1)
        best_controls = mppi.noisy_controls[torch.arange(batch_size), minidxs].unsqueeze(1)

        expert_inp = ebm_cost.make_training_input(samples)
        learner_inp = ebm_cost.make_training_input({
#            'traj': best_trajs,
#            'cmd': best_controls,
            'traj': mppi.noisy_states,
            'cmd': mppi.noisy_controls,
            'map_features': samples['map_features'],
            'metadata': samples['metadata']
        })

        ## supervision ##
        #need to try a bunch of things here

        ## lsgan loss ##
#        expert_logits = ebm.forward(expert_inp.flatten(start_dim=-2))
#        learner_logits = ebm.forward(learner_inp.flatten(start_dim=-2))
#        objective = (expert_logits - 1.).pow(2) + (learner_logits + 0.).pow(2)
#        loss = objective.mean()

        ##TODO experiment with cost = -(max(0, 1 - 0.25(ebm(x) - 1)^2))

        ## wgan loss ##
        expert_logits = ebm.forward(expert_inp.flatten(start_dim=-2))
        learner_logits = ebm.forward(learner_inp.flatten(start_dim=-2))
        gp_lam = 5.0
        alpha = torch.rand(batch_size, 1, 1, 1, device=expert_inp.device)
        interp = (alpha * expert_inp + (1.-alpha) * learner_inp).flatten(start_dim=-2)
        interp = torch.autograd.Variable(interp, requires_grad=True)
        d_interp = ebm.forward(interp)
        grad = torch.autograd.grad(
            outputs=d_interp,
            inputs=interp,
            grad_outputs=torch.ones_like(d_interp),
                create_graph=True,
                retain_graph=True,
                only_inputs=True
        )[0]
        grad_penalty = (torch.linalg.norm(grad, dim=-1) - 1.) ** 2

        objective = -expert_logits + learner_logits
        loss = objective.mean() + gp_lam * grad_penalty.mean()

        ## info-NCE loss ##
#        expert_logits = ebm.forward(expert_inp.flatten(start_dim=-2))
#        learner_logits = ebm.forward(learner_inp.flatten(start_dim=-2))

#        expert_exp_energy = -expert_logits.exp().sum(dim=-2)
#        learner_exp_energy = -learner_logits.exp().sum(dim=-2)

#        objective = expert_exp_energy / (expert_exp_energy + learner_exp_energy)
#        loss = objective.mean()

        ebm_opt.zero_grad()
        loss.backward()
        ebm_opt.step()

        avg_expert_scores = expert_logits.mean().detach().cpu()
        avg_learner_scores = learner_logits.mean().detach().cpu()
        top_1p_learner_scores = torch.quantile(learner_logits, 0.99).detach()
        avg_diff = avg_expert_scores - avg_learner_scores

        print('Avg. expert scores:     {:.4f}'.format(avg_expert_scores.item()))
        print('Avg. learner scores:    {:.4f}'.format(avg_learner_scores.item()))
        print('top 1%. learner scores: {:.4f}'.format(top_1p_learner_scores.item()))
        print('Avg. Disc. real - fake: {:.4f}'.format(avg_diff.item()))
#        print('Avg. Disc. grad norm:   {:.4f}'.format(torch.linalg.norm(grad, dim=-1).mean()))

#        ## info-NCE loss (https://arxiv.org/pdf/2109.00137.pdf)##


        ## get individual costs for analysis ##
        cost_term_costs = {}
        for weight, cost_term in zip(mppi.cost_fn.cost_weights, mppi.cost_fn.cost_terms):
            c = cost_term.cost(mppi.noisy_states, mppi.noisy_controls, mppi.cost_fn.data)
            cost_term_costs[str(cost_term)] = weight * c

        if ti % 100 == 0:
            os.mkdir(os.path.join(exp_name, 'itr_{}'.format(ti)))
            ## debug viz ##
            for i in range(3):
                fig1, axs1 = plt.subplots(2, 4, figsize=(16, 8))
                fig1.suptitle(str(mppi.cost_fn))
                axs1 = axs1.flatten()

                best_idx = mppi.costs[i].argmin()

                for fi in range(expert_inp.shape[-1]):
                    axs1[fi].plot(expert_inp[i, 0, :, fi].cpu(), label='expert')
                    axs1[fi].plot(learner_inp[i, best_idx, :, fi].cpu(), label='mppi')
                    axs1[fi].legend()

                ## get features ##
                title_str = ''
                for k,v in cost_term_costs.items():
                    c = v[i, best_idx]
                    c2 = v[i].mean()
                    title_str += '{}: {:.2f}, avg {:.2f}, '.format(k, c, c2)

                fig2, axs2 = plt.subplots(2, 4, figsize=(24, 12))
                fig2.suptitle(title_str)
                axs2 = axs2.flatten()
                axs2[0].plot(samples['cmd'][i, 0, :, 0].cpu())
                axs2[1].plot(samples['cmd'][i, 0, :, 1].cpu())
                axs2[2].plot(samples['traj'][i, 0, :, 0].cpu(), samples['traj'][i, 0, :, 1].cpu(), label='expert')
                axs2[2].plot(best_trajs[i, 0, :, 0].cpu(), best_trajs[i, 0, :, 1].cpu(), label='mppi')

                xmin = samples['metadata'][i]['origin'][0].item()
                xmax = xmin + samples['metadata'][i]['width']
                ymin = samples['metadata'][i]['origin'][1].item()
                ymax = ymin + samples['metadata'][i]['height']

                axs2[2].imshow(samples['map_features'][i, 0].cpu(), origin='lower', extent=(xmin, xmax, ymin, ymax))

                idxs = torch.argsort(mppi.costs[i])[:10]
                best_costs = cost_term_costs['EBM'][i][idxs]
                best_mincost = best_costs.min()
                best_maxcost = best_costs.max()
                axs2[3].set_xlim(xmin, xmax)
                axs2[3].set_ylim(ymin, ymax)
                for si in idxs:
                    traj = mppi.noisy_states[i, si]
                    cost = cost_term_costs['EBM'][i, si]
                    v = ((cost - best_mincost) / (best_maxcost - best_mincost)).item()
                    c = [v, 0, 1-v, 1]
                    axs2[3].plot(traj[:, 0].cpu(), traj[:, 1].cpu(), color=c)
                axs2[3].scatter(samples['traj'][i, 0, -1, 0].cpu(), samples['traj'][i, 0, -1, 1].cpu(), c='y', marker='x', s=100)
                axs2[3].imshow(samples['map_features'][i, 0].cpu(), origin='lower', extent=(xmin, xmax, ymin, ymax))
                axs2[3].set_title('spread = {:.4f}'.format((best_maxcost-best_mincost).item()))

                ## expert states ##
                for li, label in enumerate(['x', 'y', 'th', 'v', 'd']):
                    axs2[4].plot(samples['traj'][i, 0, :, li].cpu(), label='expert ' + label)
                    axs2[5].plot(best_trajs[i, 0, :, li].cpu(), label='learner ' + label)

                for li, label in enumerate(['throttle', 'steer']):
                    axs2[6].plot(samples['cmd'][i, 0, :, li].cpu(), label='expert ' + label)
                    axs2[7].plot(best_controls[i, 0, :, li].cpu(), label='mppi ' + label)

                axs2[2].legend()
                axs2[4].legend()
                axs2[5].legend()
                axs2[6].legend()
                axs2[7].legend()

                ## print some mppi EBM cost statistics ##
                fig3, axs3 = plt.subplots(1, 2, figsize=(12, 6))
                ## plot avg EBM cost over time ##
                ebm_state_costs = ebm_cost.cost(mppi.noisy_states, mppi.noisy_controls, mppi.cost_fn.data, traj_cost=False)[i]

                axs3[0].set_title('EBM cost vs. H')
                axs3[0].plot(ebm_state_costs.mean(dim=0).cpu())

                ebm_min_cost = ebm_state_costs.min()
                ebm_max_cost = ebm_state_costs.max()
                vals = (ebm_state_costs - ebm_min_cost) / (ebm_max_cost - ebm_min_cost)
                colors = torch.stack([
                    vals,
                    torch.zeros_like(vals),
                    1. - vals,
                    torch.ones_like(vals)
                ], dim=-1).view(-1, 4)

                mppi_poses = mppi.noisy_states[i, ..., :2].view(-1, 2)

                axs3[1].scatter(mppi_poses[..., 0].cpu(), mppi_poses[..., 1].cpu(), alpha=0.1, s=1., c=colors.cpu())
                axs3[1].imshow(samples['map_features'][i, 0].cpu(), origin='lower', extent=(xmin, xmax, ymin, ymax))
                axs3[1].set_xlim(xmin, xmax)
                axs3[1].set_ylim(ymin, ymax)

                ## log ##
                for fi, fig in enumerate([fig1, fig2, fig3]):
                    fp = os.path.join(exp_name, 'itr_{}'.format(ti), 'fig_{}_{}.png'.format(i, fi))
                    fig.savefig(fp)

            ## eval ##
            res = []
            for i in range(30):
                print('{}/{}'.format(i+1, 30), end='\r')
                ## get "expert" demonstrations ##
                samples = generate_samples(throttle_dist, steer_dist, model, x0, H, map_metadata)

                ## solve with mppi ##
                mppi.reset()
                mppi.cost_fn.data['goals'] = [x[0, [-1], :2] for x in samples['traj']]
                mppi.cost_fn.data['map_features'] = samples['map_features']
                mppi.cost_fn.data['map_metadata'] = samples['metadata']

                X = torch.stack([x0] * batch_size, dim=0)

                for mi in range(mppi_itrs):
                    u = mppi.get_control(X, step=False)

                ## set up learner input ##
                minidxs = mppi.costs.argmin(dim=-1)
                best_trajs = mppi.noisy_states[torch.arange(batch_size), minidxs].unsqueeze(1)
                traj_err = torch.linalg.norm(best_trajs - samples['traj'])
                res.append(traj_err)

            res = torch.stack(res).mean()

            ## log ##
            torch.save(
                {
                    'traj_error': res,
                    'train_expert_scores': avg_expert_scores,
                    'train_learner_scores': avg_learner_scores,
                    'top_1p_learner_scores': top_1p_learner_scores,
                    'train_diff_scores': avg_diff
                },
                os.path.join(exp_name, 'itr_{}'.format(ti), '_metrics.pt')
            )

