import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

from torch_mpc.models.steer_setpoint_kbm import SteerSetpointKBM

from maxent_irl_costmaps.dataset.maxent_irl_dataset import MaxEntIRLDataset

from yamaha_imitation_learning.action_clustering.splines import get_vel_steer
from yamaha_imitation_learning.feature_extraction.pretrained_resnet import PretrainedResnetFeatureExtractor
from yamaha_imitation_learning.algos.visual_knn import VisualKNN
from yamaha_imitation_learning.algos.sanity_check import SanityCheck
from yamaha_imitation_learning.metrics.metrics import *

def make_learners(train_dataset, img_feature_extractor, device):
    return {
        'k5_s10_expweight': VisualKNN(train_dataset, img_feature_extractor, k=5, dataset_skip=10, aggregate='expweight').to(device),
        'k10_s10_expweight': VisualKNN(train_dataset, img_feature_extractor, k=10, dataset_skip=10, aggregate='expweight').to(device),
        'k20_s10_expweight': VisualKNN(train_dataset, img_feature_extractor, k=20, dataset_skip=10, aggregate='expweight').to(device),
        'k50_s10_expweight': VisualKNN(train_dataset, img_feature_extractor, k=50, dataset_skip=10, aggregate='expweight').to(device),
        'k100_s10_expweight': VisualKNN(train_dataset, img_feature_extractor, k=100, dataset_skip=10, aggregate='expweight').to(device),
        'k200_s10_expweight': VisualKNN(train_dataset, img_feature_extractor, k=200, dataset_skip=10, aggregate='expweight').to(device),
        'k5_sanity_check_random': SanityCheck(train_dataset, k=5, dataset_skip=1, aggregate='random').to(device),
        'k10_sanity_check_random': SanityCheck(train_dataset, k=10, dataset_skip=1, aggregate='random').to(device),
        'k20_sanity_check_random': SanityCheck(train_dataset, k=20, dataset_skip=1, aggregate='random').to(device),
        'k50_sanity_check_random': SanityCheck(train_dataset, k=50, dataset_skip=1, aggregate='random').to(device),
        'k100_sanity_check_random': SanityCheck(train_dataset, k=100, dataset_skip=1, aggregate='random').to(device),
        'k200_sanity_check_random': SanityCheck(train_dataset, k=200, dataset_skip=1, aggregate='random').to(device),
        'sanity_check_mean': SanityCheck(train_dataset, k=10, dataset_skip=1, aggregate='mean').to(device),
    }

def visualize(dpt, res):
        mosaic = """
        ABCDE
        """

        fig, ax_dict = plt.subplot_mosaic(mosaic, figsize=(20, 4))
        axs = list(ax_dict.values())
        
        img = dpt['image'].permute(1, 2, 0)[..., [2, 1, 0]].cpu().numpy()
        ax_dict['A'].imshow(img)
        ax_dict['A'].set_title('FPV')

        learner_states = res['learner_states']
        learner_acts = res['learner_actions']
        learner_topk_states = res['learner_topk_states']
        learner_topk_acts = res['learner_topk_actions']
        gt_states = res['gt_states']
        gt_acts = res['gt_actions']
        costmap = res['costmap']
        costmap_metadata = res['costmap_metadata']

        ax_dict['B'].plot(learner_acts[..., 0].cpu(), c='r', label='learner speed')
        ax_dict['B'].plot(gt_acts[..., 0].cpu(), c='b', label='gt speed')

        for i in range(learner_topk_acts.shape[0]):
            ax_dict['B'].plot(learner_topk_acts[i, ..., 0].cpu(), c='r', alpha=0.2)

        ax_dict['B'].set_xlabel('T')
        ax_dict['B'].set_xlabel('Vel')
        ax_dict['B'].legend()
        ax_dict['B'].set_ylim(0., 15.)
        ax_dict['B'].set_title('Vel')

        ax_dict['C'].plot(learner_acts[..., 1].cpu(), c='r', label='learner steer')
        ax_dict['C'].plot(gt_acts[..., 1].cpu(), c='b', label='gt steer')

        for i in range(learner_topk_acts.shape[0]):
            ax_dict['C'].plot(learner_topk_acts[i, ..., 1].cpu(), c='r', alpha=0.2)

        ax_dict['C'].set_xlabel('T')
        ax_dict['C'].set_xlabel('Steer')
        ax_dict['C'].legend()
        ax_dict['C'].set_ylim(-0.52, 0.52)
        ax_dict['C'].set_title('Steer')

        ax_dict['D'].plot(learner_states[:, 0].cpu(), learner_states[:, 1].cpu(), c='r', label='learner traj')
        ax_dict['D'].plot(gt_states[:, 0].cpu(), gt_states[:, 1].cpu(), c='b', label='gt traj')

        for i in range(learner_topk_acts.shape[0]):
            ax_dict['D'].plot(learner_topk_states[i, ..., 0].cpu(), learner_topk_states[i, ..., 1].cpu(), c='r', alpha=0.2)


        ax_dict['E'].plot(learner_states[:, 0].cpu(), learner_states[:, 1].cpu(), c='r', label='learner traj')
        ax_dict['E'].plot(gt_states[:, 0].cpu(), gt_states[:, 1].cpu(), c='b', label='gt traj')

        for i in range(learner_topk_acts.shape[0]):
            ax_dict['E'].plot(learner_topk_states[i, ..., 0].cpu(), learner_topk_states[i, ..., 1].cpu(), c='r', alpha=0.4)

        xmin = costmap_metadata['origin'][0].item()
        ymin = costmap_metadata['origin'][1].item()
        xmax = xmin + costmap_metadata['height']
        ymax = ymin + costmap_metadata['width']
        ax_dict['E'].imshow(costmap.cpu(), origin='lower', extent=(xmin, xmax, ymin, ymax), cmap='plasma')

        ax_dict['D'].set_xlabel('X(m)')
        ax_dict['D'].set_ylabel('Y(m)')
        ax_dict['D'].set_aspect(1.)
        ax_dict['D'].legend()
        ax_dict['D'].set_title('Traj')

if __name__ == '__main__':
    parser = argparse.ArgumentParser() 
    parser.add_argument('--train_bag_fp', type=str, required=True, help='path to dir of bags to train on')
    parser.add_argument('--test_bag_fp', type=str, required=True, help='path to dir of bags to test on')
    parser.add_argument('--train_preprocess_fp', type=str, required=True, help='path of preprocessed bag data for train')
    parser.add_argument('--test_preprocess_fp', type=str, required=True, help='path of preprocessed bag data for test')
    parser.add_argument('--net_fp', type=str, required=True, help='path to pretrained resnet')
    parser.add_argument('--cost_net_fp', type=str, required=True, help='path to costmap network')
    parser.add_argument('--save_fp', type=str, required=True, help='location to log results')
    parser.add_argument('--device', type=str, required=False, default='cuda', help='device to run on')
    args = parser.parse_args()

    train_dataset = MaxEntIRLDataset(bag_fp=args.train_bag_fp, preprocess_fp=args.train_preprocess_fp).to(args.device)
    test_dataset = MaxEntIRLDataset(bag_fp=args.test_bag_fp, preprocess_fp=args.test_preprocess_fp).to(args.device)
    net = torch.load(args.net_fp)
    img_feature_extractor = PretrainedResnetFeatureExtractor(D=64, C=3, project=False, net=net).to(args.device)

    cost_net = torch.load(args.cost_net_fp).to(args.device)
    cost_net.network.eval()

    learners = make_learners(train_dataset, img_feature_extractor, args.device)
    for learner in learners.values():
        learner.process_dataset()

    #set up logging
    os.mkdir(args.save_fp)
    for lk in learners.keys():
        os.mkdir(os.path.join(args.save_fp, lk))
        os.mkdir(os.path.join(args.save_fp, lk, 'learner'))
        os.mkdir(os.path.join(args.save_fp, lk, 'results'))

    #make metrics
    metrics = {
        'vel_error': velocity_error,
        'steer_error': steer_error,
        'scaled_action_error': scaled_action_error,
        'model_position_error': model_position_error,
        'topk_vel_error': topk_velocity_error,
        'topk_steer_error': topk_steer_error,
        'topk_scaled_action_error': topk_scaled_action_error,
        'topk_model_position_error': topk_model_position_error,
        'expert_cost': expert_cost,
        'integrated_cost': integrated_cost,
        'top_integrated_cost': topk_integrated_cost,
    }

    model = SteerSetpointKBM(L=3.0, v_target_lim=[0.0, 15.0], steer_lim=[-0.52, 0.52], steer_rate_lim=0.2, dt=0.1).to(args.device)

    #choose these constants because we roughly have 10m/s of speed and 1.04rad of steer
    action_scaling = torch.tensor([1./10., 1./1.04]).to(args.device)

    metadata = {
        'action_scaling': action_scaling,
    }

    final_res = {vk:{k:[] for k in metrics.keys()} for vk in learners.keys()}

    for i in range(len(test_dataset)):
        print('{}/{}'.format(i, len(test_dataset)))
        dpt = test_dataset[i]
        for lk, lv in learners.items():
            with torch.no_grad():
                res, intermediate = evaluate_metrics(dpt, lv, metrics, model, cost_net, metadata)

            for rk, rv in res.items():
                final_res[lk][rk].append(rv.item())

            lv.visualize(query=dpt)
            plt.savefig(os.path.join(args.save_fp, lk, 'learner', '{:05d}_learner.png'.format(i)))
            plt.close()

            visualize(dpt, intermediate)
            plt.savefig(os.path.join(args.save_fp, lk, 'results', '{:05d}_results.png'.format(i)))
            plt.close()

    final_res = {k:{kk:torch.tensor(vv) for kk,vv in v.items()} for k,v in final_res.items()}
    torch.save(final_res, os.path.join(args.save_fp, 'metrics.pt'))
