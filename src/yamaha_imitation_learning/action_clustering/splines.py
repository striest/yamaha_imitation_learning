import torch
import numpy as np
import matplotlib.pyplot as plt

from maxent_irl_costmaps.dataset.maxent_irl_dataset import MaxEntIRLDataset

def get_vel_steer(batch):
    """
    grab velocity and steering from a dataset entry
    """
    vels = torch.linalg.norm(batch['traj'][:, 7:10], dim=-1)
    steers = batch['steer'] * (30./415.) * (np.pi / 180.)

    return torch.stack([vels, steers], dim=-1)

class ActionSpline:
    """
    Initial attempt to cluster actions by fitting n-th order splines
    to velocity and steer setpoint
    """
    def __init__(self, order=3):
        self.order = order

    def fit_spline(self, controls):
        """
        fit a spline of order self.order to a [T x M] set of controls
        """
        T = controls.shape[0]
        M = controls.shape[-1]
        N = self.order + 1

        t = torch.arange(T).to(controls.device) / T
        X = torch.stack([t ** n for n in range(N)], dim=-1).view(1, T, N) #[1 x T x N]
        Y = controls.T.view(M, T, 1) #[M x T x 1]

        A = torch.linalg.lstsq(X, Y)[0].view(M, 1, N) #[M x 1 x N]

        return A, (X * A).sum(dim=-1).T #[T x M]

if __name__ == '__main__':
    torch.set_printoptions(sci_mode=False)

    bag_fp = '/home/striest/Desktop/datasets/yamaha_maxent_irl/big_gridmaps/rosbags_train'
    preprocess_fp = '/home/striest/Desktop/datasets/yamaha_maxent_irl/big_gridmaps/torch_train_h75'

    dataset = MaxEntIRLDataset(bag_fp=bag_fp, preprocess_fp=preprocess_fp)

    k = 9

    spline_fits = [ActionSpline(order=n) for n in range(1, 1+k)]
    vel_error = [[] for n in range(1, 1+k)]
    steer_error = [[] for n in range(1, 1+k)]

    for i in range(len(dataset)-1):
        print('{}/{}'.format(i, len(dataset)-1), end='\r')
        controls = get_vel_steer(dataset[i])
        smoothed_controls = [s.fit_spline(controls) for s in spline_fits]
        for j in range(k):
            err = (controls - smoothed_controls[j]).pow(2).mean(dim=0).sqrt() #RMSE
            vel_error[j].append(err[0])
            steer_error[j].append(err[1])

    vel_error = [torch.tensor(v).mean().item() for v in vel_error]
    steer_error = [torch.tensor(v).mean().item() for v in steer_error]
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].plot(vel_error, marker='.')
    axs[1].plot(steer_error, marker='.')
    axs[0].set_title('vel error')
    axs[0].set_xlabel('Order')
    axs[0].set_ylabel('Error')
    axs[1].set_title('steer error')
    axs[1].set_xlabel('Order')
    axs[1].set_ylabel('Error')
    plt.show()

    for i in range(100):
        idx = np.random.choice(len(dataset) -1)
        controls = get_vel_steer(dataset[idx])
        smoothed_controls = [s.fit_spline(controls) for s in spline_fits]

        fig, axs = plt.subplots(2, k + 1,  figsize=(3 * (k + 1), 6))
        idx = np.random.choice(np.arange(len(dataset)))

        axs[0, 0].plot(controls[:, 0], c='b', label='gt vel')
        axs[1, 0].plot(controls[:, 1], c='r', label='gt steer')

        for ki in range(k):
            axs[0, 1+ki].plot(smoothed_controls[ki][:, 0], c='b', label='{} spline'.format(ki+1))
            axs[1, 1+ki].plot(smoothed_controls[ki][:, 1], c='r', label='{} spline'.format(ki+1))
    
        for ax in axs.flatten():
            ax.legend()

        plt.show()
