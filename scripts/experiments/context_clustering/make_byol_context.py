import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

from torch_mpc.models.steer_setpoint_kbm import SteerSetpointKBM

from maxent_irl_costmaps.dataset.maxent_irl_dataset import MaxEntIRLDataset

from yamaha_imitation_learning.action_clustering.splines import get_vel_steer
from yamaha_imitation_learning.feature_extraction.pretrained_resnet import PretrainedResnetFeatureExtractor

if __name__ == '__main__':
    parser = argparse.ArgumentParser() 
#    parser.add_argument('--train_bag_fp', type=str, required=True, help='path to dir of bags to train on')
    parser.add_argument('--test_bag_fp', type=str, required=True, help='path to dir of bags to test on')
#    parser.add_argument('--train_preprocess_fp', type=str, required=True, help='path of preprocessed bag data for train')
    parser.add_argument('--test_preprocess_fp', type=str, required=True, help='path of preprocessed bag data for test')
    parser.add_argument('--net_fp', type=str, required=True, help='path to pretrained resnet')
    parser.add_argument('--device', type=str, required=False, default='cuda', help='device to run on')
    args = parser.parse_args()

#    train_dataset = MaxEntIRLDataset(bag_fp=args.train_bag_fp, preprocess_fp=args.train_preprocess_fp).to(args.device)
    test_dataset = MaxEntIRLDataset(bag_fp=args.test_bag_fp, preprocess_fp=args.test_preprocess_fp).to(args.device)
    net = torch.load(args.net_fp)
    net.eval()
    img_feature_extractor = PretrainedResnetFeatureExtractor(D=64, C=3, project=False, net=net).to(args.device)

    last_pose = torch.zeros(2)
    feat_seq = torch.zeros(0, 2048)
    changepoints = []
    t = 0

    for dpt in test_dataset:
        if t > len(test_dataset)-1:
            break

        #hacky changepoint detection
        curr_pose = dpt['traj'][0, :2].cpu()

        if torch.linalg.norm(curr_pose - last_pose) > 10.:
            print('changepoint!')
            changepoints.append(t)

        img = dpt['image']
        with torch.no_grad():
            feats = img_feature_extractor.get_features(img)

        feat_seq = torch.cat([feat_seq, feats.unsqueeze(0).cpu()], dim=0)

        last_pose = curr_pose
        t += 1

    #do some feature post-processing (namely SVD)
    feat_seq -= feat_seq.mean(dim=0)
    U, S, V = torch.linalg.svd(feat_seq, full_matrices=False)
    feat_seq = U @ torch.diag(S)
    feat_seq = feat_seq[:, :50]

    #also try the dumb way to check
#    stds = feat_seq.std(dim=0)
#    idxs = torch.argsort(stds, descending=True)
#    feat_seq = feat_seq[:, idxs[:50]]

    fig, axs = plt.subplots(1, 2, figsize=(18, 9))
    axs[0].imshow(img.permute(1, 2, 0)[..., [2, 1, 0]].cpu())
    axs[1].imshow(feat_seq.T)
    axs[1].set_aspect('auto')

    for cp in changepoints:
        axs[1].axvline(cp, color='r')

    curr_bar = axs[1].axvline(0, color='b')

    def on_plot_hover(event):
        global curr_bar
        if event.inaxes == axs[1]:
            axs[0].cla()
            axs[1].autoscale(False)

            idx = int(event.xdata)
            img = test_dataset[idx]['image']

            axs[0].imshow(img.permute(1, 2, 0)[..., [2, 1, 0]].cpu())
#            axs[1].imshow(feat_seq.T)

            curr_bar.remove()
            curr_bar = axs[1].axvline(idx, color='b')

#            for cp in changepoints:
#                axs[1].axvline(cp, color='r')
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect('motion_notify_event', on_plot_hover)
    plt.show()
