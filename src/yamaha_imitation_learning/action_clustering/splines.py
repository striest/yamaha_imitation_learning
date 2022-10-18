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
    def __init__(self, order=3, lam=1e-3):
        self.order = order
        self.lam = lam

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

    def fit_regularized_spline(self, controls):
        """
        fit a spline of order self.order to a [T x M] set of controls.
        Solve w/ ridge regression to get more reasonable coefficients
        """
        T = controls.shape[0]
        M = controls.shape[-1]
        N = self.order + 1

        t = torch.arange(T).to(controls.device) / T

        X = torch.stack([t ** n for n in range(N)], dim=-1).view(1, T, N) #[1 x T x N]
        Y = controls.T.view(M, T, 1) #[M x T x 1]
        Xt = torch.swapaxes(X, -2, -1)
        Yt = torch.swapaxes(Y, -2, -1)

        reg = self.lam * torch.eye(N, device=controls.device).view(1, N, N)

        A = (torch.linalg.inv(Xt @ X + reg) @ Xt @ Y).swapaxes(-2, -1)

        return A.view(M, N), (X * A).sum(dim=-1).T #[T x M]

    def get_controls(self, A, T):
        """
        generate a sequence of controls from a coeff matrix A
        """
        M = A.shape[0]
        N = A.shape[1]

        t = torch.arange(T).to(A.device) / T
        X = torch.stack([t ** n for n in range(N)], dim=-1).view(1, T, N) #[1 x T x N]
        return (X * A.view(M, 1, N)).sum(dim=-1).T #[T x M]

    def to(self, device):
        return self

if __name__ == '__main__':
    torch.set_printoptions(sci_mode=False)

    bag_fp = '/home/atv/Desktop/datasets/yamaha_maxent_irl/big_gridmaps/rosbags_train'
    preprocess_fp = '/home/atv/Desktop/datasets/yamaha_maxent_irl/big_gridmaps/torch_train_h75'

    dataset = MaxEntIRLDataset(bag_fp=bag_fp, preprocess_fp=preprocess_fp)

    k = 9
    lam = 1e-3

    spline_fits = [ActionSpline(order=n) for n in range(3, 1+k)]
    vel_error = [[] for n in range(1, 1+k)]
    steer_error = [[] for n in range(1, 1+k)]
    coeff_norm = [[] for n in range(1, 1+k)]

    for i in range(len(dataset)-1):
        print('{}/{}'.format(i, len(dataset)-1), end='\r')
        controls = get_vel_steer(dataset[i])
        res = [s.fit_regularized_spline(controls, lam=lam) for s in spline_fits]

        coeffs = [x[0] for x in res]
        smoothed_controls = [x[1] for x in res]

        for j in range(k):
            err = (controls - smoothed_controls[j]).pow(2).mean(dim=0).sqrt() #RMSE
            reconstruct_controls = spline_fits[j].get_controls(coeffs[j], T=controls.shape[0])
            vel_error[j].append(err[0])
            steer_error[j].append(err[1])
            coeff_norm[j].append(coeffs[j].abs().max())
        
    vel_error = [torch.tensor(v).mean().item() for v in vel_error]
    steer_error = [torch.tensor(v).mean().item() for v in steer_error]
    coeff_norm = [torch.tensor(v).mean().item() for v in coeff_norm]
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    axs[0].plot(vel_error, marker='.')
    axs[1].plot(steer_error, marker='.')
    axs[2].plot(coeff_norm, marker='.')
    axs[0].set_title('vel error')
    axs[0].set_xlabel('Order')
    axs[0].set_ylabel('Error')
    axs[1].set_title('steer error')
    axs[1].set_xlabel('Order')
    axs[1].set_ylabel('Error')
    axs[2].set_title('coeff max')
    axs[2].set_xlabel('Order')
    axs[2].set_ylabel('Norm')
    plt.show()

    for i in range(100):
        idx = np.random.choice(len(dataset) -1)
        controls = get_vel_steer(dataset[idx])
        res = [s.fit_regularized_spline(controls, lam=lam) for s in spline_fits]
        coeffs = [x[0] for x in res]
        smoothed_controls = [x[1] for x in res]

        fig, axs = plt.subplots(2, k + 1,  figsize=(3 * (k + 1), 6))
        idx = np.random.choice(np.arange(len(dataset)))

        axs[0, 0].plot(controls[:, 0], c='b', label='gt vel')
        axs[1, 0].plot(controls[:, 1], c='r', label='gt steer')

        for ki in range(k):
            axs[0, 1+ki].plot(smoothed_controls[ki][:, 0], c='b', label='{} spline'.format(ki+1))
            axs[0, 1+ki].plot(controls[:, 0], c='b', alpha=0.1)
            axs[1, 1+ki].plot(smoothed_controls[ki][:, 1], c='r', label='{} spline'.format(ki+1))
            axs[1, 1+ki].plot(controls[:, 1], c='r', alpha=0.1)
    
        for ax in axs.flatten():
            ax.legend()

        plt.show()
