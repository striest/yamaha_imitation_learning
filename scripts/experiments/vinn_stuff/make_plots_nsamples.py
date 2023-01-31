import os
import torch
import matplotlib.pyplot as plt

if __name__ == '__main__':
    base_resnet_fp = 'base_resnet_costmap_more/metrics.pt'
    byol_resnet_fp = 'byol_25000_resnet_costmap_more/metrics.pt'

    base_resnet_raw = torch.load(base_resnet_fp)
    byol_resnet_raw = torch.load(byol_resnet_fp)

    #experiment 1: compare random baselines to visual clustering
    resnet_keys = ['k5_s10_expweight', 'k10_s10_expweight', 'k20_s10_expweight', 'k50_s10_expweight', 'k100_s10_expweight', 'k200_s10_expweight']
    baseline_keys = ['k5_sanity_check_random', 'k10_sanity_check_random', 'k20_sanity_check_random', 'k50_sanity_check_random', 'k100_sanity_check_random', 'k200_sanity_check_random']
    exp1_plot_keys = ['scaled_action_error', 'model_position_error', 'integrated_cost']

    baseline_data = {}
    resnet_data = {}
    byol_resnet_data = {}

    #populate baseline data
    for k in baseline_keys:
        for kk,vv in base_resnet_raw[k].items():
            if kk not in baseline_data.keys():
                baseline_data[kk] = []

            baseline_data[kk].append(vv.mean())

    #populate resnet data
    for k in resnet_keys:
        for kk,vv in base_resnet_raw[k].items():
            if kk not in resnet_data.keys():
                resnet_data[kk] = []

            resnet_data[kk].append(vv.mean())

    #populate byol-resnet data
    for k in resnet_keys:
        for kk,vv in byol_resnet_raw[k].items():
            if kk not in byol_resnet_data.keys():
                byol_resnet_data[kk] = []

            byol_resnet_data[kk].append(vv.mean())

    #compare base data to resnet/resnot byol
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    for pi, pk in enumerate(exp1_plot_keys):
        axs[pi].set_title(pk)
        axs[pi].plot(baseline_data[pk], label='random')
        axs[pi].plot(resnet_data[pk], label='resnet')
        axs[pi].plot(byol_resnet_data[pk], label='byol-resnet')
        axs[pi].legend()

    plt.show()

    exp2_plot_keys = ['topk_scaled_action_error', 'topk_model_position_error', 'top_integrated_cost']

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    for pi, pk in enumerate(exp2_plot_keys):
        axs[pi].set_title(pk)
        axs[pi].plot(baseline_data[pk], label='random')
        axs[pi].plot(resnet_data[pk], label='resnet')
        axs[pi].plot(byol_resnet_data[pk], label='byol-resnet')
        axs[pi].legend()

    plt.show()

    #plot both together
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    for pi, pk in enumerate(exp1_plot_keys):
        axs[pi].set_title(pk)
        axs[pi].plot(baseline_data[pk], label='random-top1')
        axs[pi].plot(resnet_data[pk], label='resnet-top1')
        axs[pi].plot(byol_resnet_data[pk], label='byol-resnet-top1')
        axs[pi].legend()

    for pi, pk in enumerate(exp2_plot_keys):
        axs[pi].set_title(pk)
        axs[pi].plot(baseline_data[pk], label='random-topk')
        axs[pi].plot(resnet_data[pk], label='resnet-topk')
        axs[pi].plot(byol_resnet_data[pk], label='byol-resnet-topk')
        axs[pi].legend()

    plt.show()
