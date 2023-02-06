import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from maxent_irl_costmaps.dataset.maxent_irl_dataset import MaxEntIRLDataset
from torch_mpc.models.steer_setpoint_throttle_kbm import SteerSetpointThrottleKBM
from yamaha_imitation_learning.dataset.bc_dataset import BCDataset

class BCTrainer:
    """
    Train behavioral cloning from a pretrained resnet. Probably just need a linear set of weights at the end
    """
    def __init__(self, dataset, test_dataset, network, opt, batch_size=32, device='cpu'):
        self.dataset = dataset
        self.test_dataset = test_dataset
        self.network = network
        self.opt = opt
        self.batch_size = batch_size
        self.epochs = 0
        self.device = device

        self.model = SteerSetpointThrottleKBM(L=3.0, throttle_lim=[0.0, 1.0], steer_lim=[-0.52, 0.52], steer_rate_lim=0.3, k_fric=0.00, k_drag=0.1, dt=0.1, w_Kp=50.).to(device)

    def update(self, n=-1):
        """
        Perform one epoch of training
        """
        self.epochs += 1
        dl = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
        nitrs = int(len(self.dataset) / self.batch_size)
        res = []

        for i, batch in enumerate(dl):
            if n > -1 and i >= n:
                break

            loss_dict = self.get_losses(batch)
            loss = sum(loss_dict.values())

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            res.append({k:v.detach() for k,v in loss_dict.items()})

        res = {k:torch.tensor([x[k] for x in res]).mean() for k in res[0].keys()}
        print('_____EPOCH {}_____'.format(self.epochs))
        return res

    def eval(self):
        dl = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=True)
        nitrs = int(len(self.test_dataset) / self.batch_size)
        res = []

        for i, batch in enumerate(dl):
            with torch.no_grad():
                loss_dict = self.get_losses(batch)
                res.append({k:v.detach() for k,v in loss_dict.items()})

        res = {k:torch.tensor([x[k] for x in res]).mean() for k in res[0].keys()}
        return res
        
    def get_losses(self, batch):
        acts = self.network.forward(batch)
        gt_acts = batch['cmd']

        gt_loss = (acts - gt_acts).pow(2).mean()
        tv_loss = torch.zeros(1, device=gt_loss.device).squeeze()

        return {
            'pred_loss': gt_loss,
            'tv_loss': tv_loss,
        }

    def visualize(self, idx=-1, fig=None, axs=None):
        if idx == -1:
            idx = torch.randint(len(self.test_dataset), size=(1, )).item()

        fig, axs = self.test_dataset.visualize(idx, fig, axs)

        dpt = self.test_dataset[idx]
        dpt['image'] = dpt['image'].unsqueeze(0)
        dpt['dynamics'] = dpt['dynamics'].unsqueeze(0)
        dpt['goal'] = dpt['goal'].unsqueeze(0)
        dpt['cmd'] = dpt['cmd'].unsqueeze(0)
        with torch.no_grad():
            pred = self.network.forward(dpt).squeeze(0).cpu()

        #plot model outputs
        gt_traj = dpt['kbm_traj']
        gt_acts = dpt['cmd'][0]
        gt_acts[:, 1] *= -0.52
        pred_acts = pred.clone()
        pred_acts[:, 1] *= -0.52

        gt_pred_traj = self.model.rollout(gt_traj[0], gt_acts).cpu()
        pred_pred_traj = self.model.rollout(gt_traj[0], pred_acts).cpu()
        axs[1].plot(gt_pred_traj[:, 0], gt_pred_traj[:, 1], c='b', label='gt_acts_w_model')
        axs[1].plot(pred_pred_traj[:, 0], pred_pred_traj[:, 1], c='r', label='pred_acts_w_model')

        axs[2].plot(pred[:, 0], label='throttle pred')
        axs[3].plot(pred[:, 1], label='steer pred')

        axs[1].legend()
        axs[2].legend()
        axs[3].legend()
        
    def to(self, device):
        self.device = device
        self.dataset = self.dataset.to(device)
        self.network = self.network.to(device)
        return self

if __name__ == '__main__':
    from yamaha_imitation_learning.networks.direct_prediction.resnet import ResnetFPV
    torch.set_printoptions(sci_mode=False)
    device = 'cuda'

    train_bag_fp = '/home/atv/Desktop/datasets/yamaha_maxent_irl/big_gridmaps/rosbags_train'
    train_preprocess_fp = '/home/atv/Desktop/datasets/yamaha_maxent_irl/big_gridmaps/torch_train_h75'
    train_dataset = BCDataset(MaxEntIRLDataset(bag_fp=train_bag_fp, preprocess_fp=train_preprocess_fp, horizon=75)).to(device)

    test_bag_fp = '/home/atv/Desktop/datasets/yamaha_maxent_irl/big_gridmaps/rosbags_test'
    test_preprocess_fp = '/home/atv/Desktop/datasets/yamaha_maxent_irl/big_gridmaps/torch_test_h75'
    test_dataset = BCDataset(MaxEntIRLDataset(bag_fp=test_bag_fp, preprocess_fp=test_preprocess_fp, horizon=75)).to(device)

    ## pretrain + fix ##
    base_net = torch.load('../../../models/resnet_byol_25000.pt')
    net = ResnetFPV(net=base_net, nf=100, freeze_backbone=True)

    ## pretrain no fix ##
#    base_net = torch.load('../../../models/resnet_byol_25000.pt')
#    net = ResnetFPV(net=base_net, nf=100, freeze_backbone=False)

    ## no pretrain or fix ##
#    net = ResnetFPV(nf=100, freeze_backbone=False)

    opt = torch.optim.Adam(net.parameters())

    trainer = BCTrainer(train_dataset, test_dataset, net, opt).to(device)

    for i in range(10):
        trainer.train_epoch()
        res = trainer.test()
        print(res)
#        for i in range(5):
#            trainer.visualize()
#            plt.show()

    for i in range(100):
        trainer.visualize()
        plt.show()
