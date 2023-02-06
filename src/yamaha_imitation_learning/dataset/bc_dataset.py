import torch
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset

from maxent_irl_costmaps.dataset.maxent_irl_dataset import MaxEntIRLDataset

from torch_mpc.models.steer_setpoint_throttle_kbm import SteerSetpointThrottleKBM

from yamaha_imitation_learning.action_clustering.splines import get_vel_steer

class BCDataset(Dataset):
    """
    Wrap the IRL dataset for BC. Changes are:
        1. Add a goal feature that is range/bearing to final state (random sample this eventually)
        2. Provide dynamics part of state (i.e. non-position stuff in KBM)
    """
    def __init__(self, base_dataset, goal_sample=True, device='cpu'):
        """
        Args:
            base_dataset: A MaxEntIRL dataaset instance to process
            goal_sample: Whether to take final traj state, or uniformly sample goal from the traj
        """
        self.dataset = base_dataset
        self.model = SteerSetpointThrottleKBM(device=device)
        self.goal_sample = goal_sample
        self.goal_range_norm = 50. #scale goal range values appropriately

        self.device = device

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        dpt = self.dataset[idx]

        #create KBM traj
        traj = self.model.get_observations({'state': dpt['traj'], 'steer_angle': dpt['steer'].unsqueeze(-1)})

        #subsample a goal
        start = traj[0]
        goal = traj[torch.randint(traj.shape[0], size=())] if self.goal_sample else traj[-1]

        goal_angle = torch.atan2(goal[1]-start[1], goal[0]-start[0]) - start[2]
        goal_dist = torch.linalg.norm(start[:2] - goal[:2]) / self.goal_range_norm

        state = start[3:]

        dpt['goal'] = torch.stack([goal_dist, goal_angle]).to(self.device)
        dpt['dynamics'] = state.to(self.device)
        dpt['kbm_traj'] = traj.to(self.device)

        return dpt

    def to(self, device):
        self.device =  device
        self.dataset = self.dataset.to(device)
        return self

    def visualize(self, idx=-1, fig=None, axs=None):
        if fig is None or axs is None:
            fig, axs = plt.subplots(1, 4, figsize=(24, 6))
        if idx == -1:
            idx = torch.randint(len(self), size=(1, )).item()

        dpt = self[idx]

        xmin = dpt['metadata']['origin'][0].item()
        xmax = xmin + dpt['metadata']['width'].item()
        ymin = dpt['metadata']['origin'][1].item()
        ymax = ymin + dpt['metadata']['height'].item()

        axs[0].imshow(dpt['image'].permute(1, 2, 0)[..., [2, 1, 0]].cpu())

        axs[1].imshow(dpt['map_features'][2].cpu(), origin='lower', extent=(xmin, xmax, ymin, ymax), cmap='gray')
        axs[1].plot(dpt['traj'][:, 0].cpu(), dpt['traj'][:, 1].cpu(), c='y', label='gt_traj')
        #deconstruct range/bearing
        goal_x = dpt['kbm_traj'][0, 0] + dpt['goal'][0] * self.goal_range_norm * torch.cos(dpt['kbm_traj'][0, 2] + dpt['goal'][1])
        goal_y = dpt['kbm_traj'][0, 1] + dpt['goal'][0] * self.goal_range_norm * torch.sin(dpt['kbm_traj'][0, 2] + dpt['goal'][1])
        axs[1].plot([dpt['traj'][0, 0].cpu(), goal_x.cpu()], [dpt['traj'][0, 1].cpu(), goal_y.cpu()])

        axs[2].plot(dpt['cmd'][:, 0].cpu(), label='throttle')
        axs[2].set_ylim(-0.1, 1.1)

        axs[3].plot(dpt['cmd'][:, 1].cpu(), label='steer')
        axs[3].set_ylim(-1.1, 1.1)

        axs[1].legend()
        axs[2].legend()
        axs[3].legend()

        return fig, axs

if __name__ == '__main__':
    torch.set_printoptions(sci_mode=False)
    device = 'cuda'

    train_bag_fp = '/home/atv/Desktop/datasets/yamaha_maxent_irl/big_gridmaps/rosbags_test'
    train_preprocess_fp = '/home/atv/Desktop/datasets/yamaha_maxent_irl/big_gridmaps/torch_test_h75'
    train_dataset = BCDataset(MaxEntIRLDataset(bag_fp=train_bag_fp, preprocess_fp=train_preprocess_fp))

    for i in range(10):
        train_dataset.visualize()
        plt.show()
