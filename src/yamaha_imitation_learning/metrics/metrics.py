"""
Metrics for evaluating imitation performance
"""
import torch

from torch_mpc.cost_functions.cost_terms.utils import world_to_grid

from yamaha_imitation_learning.action_clustering.splines import get_vel_steer

#individual error metrics
def velocity_error(data, metadata):
    learner_actions = data['learner_actions']
    gt_actions = data['gt_actions']
    return (learner_actions - gt_actions).abs()[..., 0].mean()

def topk_velocity_error(data, metadata):
    learner_topk_actions = data['learner_topk_actions']
    gt_actions = data['gt_actions']
    return (learner_topk_actions - gt_actions.unsqueeze(0)).abs()[..., 0].mean(dim=-1).min()

def steer_error(data, metadata):
    learner_actions = data['learner_actions']
    gt_actions = data['gt_actions']
    return (learner_actions - gt_actions).abs()[..., 1].mean()

def topk_steer_error(data, metadata):
    learner_topk_actions = data['learner_topk_actions']
    gt_actions = data['gt_actions']
    return (learner_topk_actions - gt_actions.unsqueeze(0)).abs()[..., 1].mean(dim=-1).min()

def scaled_action_error(data, metadata):
    learner_actions = data['learner_actions']
    gt_actions = data['gt_actions']
    scale = metadata['action_scaling']
    n_lead_dims = len(gt_actions.shape) - 1
    lead_dims = [1] * n_lead_dims

    return ((learner_actions - gt_actions).abs() * scale.view(*lead_dims, -1)).mean()

def topk_scaled_action_error(data, metadata):
    learner_topk_actions = data['learner_topk_actions']
    gt_actions = data['gt_actions']
    scale = metadata['action_scaling']
    n_lead_dims = len(learner_topk_actions.shape) - 1
    lead_dims = [1] * n_lead_dims

    return ((learner_topk_actions - gt_actions.unsqueeze(0)).abs() * scale.view(*lead_dims, -1)).mean(dim=-1).mean(dim=-1).min()

def model_position_error(data, metadata):
    gt_states = data['gt_states']
    learner_states = data['learner_states']
    return torch.linalg.norm((gt_states - learner_states)[..., :2], dim=-1).mean()

def topk_model_position_error(data, metadata):
    gt_states = data['gt_states']
    learner_topk_states = data['learner_topk_states']
    return torch.linalg.norm((gt_states.unsqueeze(0) - learner_topk_states)[..., :2], dim=-1).mean(dim=-1).min()

def expert_cost(data, metadata):
    map_metadata = data['costmap_metadata']
    costmap = data['costmap']
    traj = data['gt_states']

    cost = get_cost_from_costmap(traj, costmap, map_metadata)
    return cost

def integrated_cost(data, metadata):
    map_metadata = data['costmap_metadata']
    costmap = data['costmap']
    traj = data['learner_states']

    cost = get_cost_from_costmap(traj, costmap, map_metadata)
    return cost

def topk_integrated_cost(data, metadata):
    map_metadata = data['costmap_metadata']
    costmap = data['costmap']
    traj = data['learner_topk_states']

    cost = get_cost_from_costmap(traj, costmap, map_metadata)
    return cost.min()

def get_cost_from_costmap(poses, costmap, metadata):
    res = metadata['resolution']
    nx = metadata['height']/res
    ny = metadata['width']/res
    ox = metadata['origin'][0]
    oy = metadata['origin'][1]

    gx = (poses[..., 0] - ox) / res
    gy = (poses[..., 1] - oy) / res

    grid_pos = torch.stack([gx, gy], dim=-1).long()
    invalid_mask = (grid_pos[..., 0] < 0) | (grid_pos[..., 1] < 0) | (grid_pos[..., 0] >= nx) | (grid_pos[..., 1] >= ny)

    # Switch grid axes to align with robot centric axes: +x forward, +y left
    grid_pos = grid_pos[..., [1, 0]]
    grid_pos[invalid_mask] = 0
    grid_pos = grid_pos.long()

    new_costs = torch.clone(costmap[grid_pos[..., 0], grid_pos[..., 1]])
    new_costs[invalid_mask] = 0.
    cost = new_costs.sum(dim=-1)

    return cost

#main driver method
def evaluate_metrics(dpt, learner, metrics, model, cost_net, metadata):
    gt_states = model.get_observations({
        'state': dpt['traj'],
        'steer_angle': dpt['steer'].unsqueeze(1)
    })

    gt_actions = get_vel_steer(dpt)
    gt_states = model.rollout(gt_states[0], gt_actions)

    learner_actions = learner.predict(dpt)
    learner_states = model.rollout(gt_states[0], learner_actions)

    learner_topk_actions = learner.predictk(dpt)
    learner_topk_states = model.rollout(torch.tile(gt_states[0], [learner.k, 1]), learner_topk_actions)

    #get the costmap
    costmap = cost_net.network.forward(dpt['map_features'].unsqueeze(0))['costmap'].squeeze()

    data = {
        'gt_states': gt_states,
        'gt_actions': gt_actions,
        'learner_states': learner_states,
        'learner_actions': learner_actions,
        'learner_topk_states': learner_topk_states,
        'learner_topk_actions': learner_topk_actions,
        'costmap': costmap,
        'costmap_metadata': dpt['metadata']
    }

    res = {}
    for k,v in metrics.items():
        res[k] = v(data, metadata)

    return res, data
