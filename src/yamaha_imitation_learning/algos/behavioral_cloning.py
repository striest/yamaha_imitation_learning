import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from maxent_irl_costmaps.dataset.maxent_irl_dataset import MaxEntIRLDataset

from yamaha_imitation_learning.action_clustering.splines import ActionSpline, get_vel_steer
from yamaha_imitation_learning.networks.resnet_spline import ResnetSplinePredictor

class BCTrainer:
    """
    Train behavioral cloning from a pretrained resnet. Probably just need a linear set of weights at the end
    """
    def __init__(self, dataset, network, opt, action_feature_extractor, batch_size=32, device='cpu'):
        self.dataset = dataset
        self.action_extractor = action_feature_extractor
        self.network = network
        self.opt = opt
        self.batch_size = batch_size
        self.epochs = 0
        self.device = device

    def train_epoch(self):
        """
        Perform one epoch of training
        """
        dl = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
        nitrs = int(len(self.dataset) / self.batch_size)
        for i, batch in enumerate(dl):
            res = self.update(batch)
            print('{}/{} pred={:.4f}, tv={:.4f}'.format(i+1, nitrs, res['pred_loss'], res['tv_loss']))

        self.epochs += 1
        print('_____EPOCH {}_____'.format(self.epochs))
        
    def update(self, batch):
        imgs = batch['image']
        acts = get_vel_steer(batch)
#        acts = acts / torch.tensor([15.0, 0.5], device=acts.device).view(1, 1, 2)
        T = acts.shape[-2]

        act_embeddings = self.network.forward(imgs)
        #TODO: vectorize this v
#        act_preds = torch.stack([self.action_extractor.get_controls(ae, T) for ae in act_embeddings], dim=0)
        act_preds = act_embeddings
        #unnormalize
        act_preds = act_preds * torch.tensor([15.0, 0.5], device=act_preds.device).view(1, 1, 2) + acts[:, [0], :]

        res = [self.action_extractor.fit_regularized_spline(a) for a in acts]
        act_gt_embeddings = torch.stack([x[0] for x in res], dim=0)
        smooth_acts = torch.stack([x[1] for x in res], dim=0)

        gt_loss = (smooth_acts - act_preds).pow(2).mean()
        tv_loss = (act_preds[:, 1:] - act_preds[:, :-1]).abs().mean()

#        ae_loss = (act_gt_embeddings - act_embeddings).pow(2).mean()

        loss = gt_loss + tv_loss

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return {
            'pred_loss': gt_loss.item(),
            'tv_loss': tv_loss.item(),
        }

    def visualize(self, idx=-1):
        fig, axs = plt.subplots(1, 4, figsize=(24, 6))
        if idx == -1:
            idx = np.random.randint(len(self.dataset))
        img = self.dataset[idx]['image']
        gt_acts = get_vel_steer(self.dataset[idx])
        gt_embeddings, gt_smooth_acts = self.action_extractor.fit_regularized_spline(gt_acts)

        with torch.no_grad():
            pred_embeddings = self.network.forward(img.unsqueeze(0)).squeeze(0)
#            pred_acts = self.action_extractor.get_controls(pred_embeddings, gt_acts.shape[-2])
            pred_acts = pred_embeddings
            #unnormalize
            pred_acts = pred_acts * torch.tensor([15.0, 0.5], device=pred_acts.device).view(1, 2) + gt_acts[[0], :]

        gt_vel_error = (gt_acts[:, 0] - pred_acts[:, 0]).pow(2).mean()
        smooth_vel_error = (gt_smooth_acts[:, 0] - pred_acts[:, 0]).pow(2).mean()
        gt_steer_error = (gt_acts[:, 1] - pred_acts[:, 1]).pow(2).mean()
        smooth_steer_error = (gt_smooth_acts[:, 1] - pred_acts[:, 1]).pow(2).mean()

        axs[0].imshow(img.permute(1, 2, 0)[:, :, [2, 1, 0]].cpu())

        axs[1].plot(gt_acts[:, 0].cpu(), c='b', alpha=0.5, label='gt_vel')
        axs[1].plot(gt_smooth_acts[:, 0].cpu(), c='b', label='gt_smooth_vel')
        axs[1].plot(pred_acts[:, 0].cpu(), c='r', label='pred_vel')
        axs[1].set_title('Vel (error = {:.4f}, smooth_error = {:.4f})'.format(gt_vel_error.item(), smooth_vel_error.item()))

        axs[2].plot(gt_acts[:, 1].cpu(), c='b', alpha=0.5, label='gt_steer')
        axs[2].plot(gt_smooth_acts[:, 1].cpu(), c='b', label='gt_smooth_steer')
        axs[2].plot(pred_acts[:, 1].cpu(), c='r', label='pred_steer')
        axs[2].set_title('Vel (error = {:.4f}, smooth_error = {:.4f})'.format(gt_steer_error.item(), smooth_steer_error.item()))

#        axs[3].plot(gt_embeddings.flatten().cpu(), c='b', label='gt embedding')
#        axs[3].plot(pred_embeddings.flatten().cpu(), c='r', label='pred embedding')
#        axs[3].set_title('Embeddings (error = {:.4f})'.format(torch.linalg.norm(gt_embeddings - pred_embeddings).cpu().item()))

        for ax in axs.flatten():
            ax.legend()

        return fig, axs
        
    def to(self, device):
        self.device = device
        self.action_extractor = self.action_extractor.to(device)
        self.dataset = self.dataset.to(device)
        self.network = self.network.to(device)
        return self

if __name__ == '__main__':
    torch.set_printoptions(sci_mode=False)
    device = 'cuda'

    train_bag_fp = '/home/atv/Desktop/datasets/yamaha_maxent_irl/big_gridmaps/rosbags_train'
    train_preprocess_fp = '/home/atv/Desktop/datasets/yamaha_maxent_irl/big_gridmaps/torch_train_h75'
    train_dataset = MaxEntIRLDataset(bag_fp=train_bag_fp, preprocess_fp=train_preprocess_fp).to(device)

    k = 9
    net = torch.load('../pretraining/resnet_base.pt')
    action_feature_extractor = ActionSpline(order=k, lam=1e-3)
    net = ResnetSplinePredictor(net=net, outsize=(75, 2)).to(device)

    test_bag_fp = '/home/atv/Desktop/datasets/yamaha_maxent_irl/big_gridmaps/rosbags_test'
    test_preprocess_fp = '/home/atv/Desktop/datasets/yamaha_maxent_irl/big_gridmaps/torch_test_h75'
    test_dataset = MaxEntIRLDataset(bag_fp=test_bag_fp, preprocess_fp=test_preprocess_fp).to(device)

    opt = torch.optim.Adam(net.parameters())
    bc_trainer = BCTrainer(train_dataset, net, opt, action_feature_extractor).to(device)

    for i in range(10):
        bc_trainer.train_epoch()

    bc_trainer.dataset = test_dataset

    for i in range(100):
        bc_trainer.visualize()
        plt.show()
