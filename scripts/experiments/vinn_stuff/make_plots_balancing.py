import os
import torch
import matplotlib.pyplot as plt

if __name__ == '__main__':
    baseline_fp = 'byol_25000_resnet_costmap/metrics.pt'
    balanced_fp = 'byol_25000_resnet_balanced_p02/all_data_eval/metrics.pt'

    baseline_raw = torch.load(baseline_fp)
    balanced_raw = torch.load(balanced_fp)

    #experiment 1: compare random baselines to visual clustering
    random_keys = ['k5_sanity_check_random', 'k10_sanity_check_random', 'k20_sanity_check_random', 'k50_sanity_check_random']
    baseline_keys = ['k5_s10_expweight', 'k10_s10_expweight', 'k20_s10_expweight', 'k50_s10_expweight']
    baseline2_keys = ['k5_s1_expweight', 'k10_s1_expweight', 'k20_s1_expweight', 'k50_s1_expweight']
    balanced_keys = ['k5_s1_expweight', 'k10_s1_expweight', 'k20_s1_expweight', 'k50_s1_expweight']

    random_data = {}
    baseline_data = {}
    baseline2_data = {}
    balanced_data = {}

    #populate baseline data
    for k in random_keys:
        for kk,vv in baseline_raw[k].items():
            if kk not in random_data.keys():
                random_data[kk] = []

            random_data[kk].append(vv.mean())

    #populate resnet data
    for k in baseline_keys:
        for kk,vv in baseline_raw[k].items():
            if kk not in baseline_data.keys():
                baseline_data[kk] = []

            baseline_data[kk].append(vv.mean())

    #populate resnet data
    for k in baseline2_keys:
        for kk,vv in baseline_raw[k].items():
            if kk not in baseline2_data.keys():
                baseline2_data[kk] = []

            baseline2_data[kk].append(vv.mean())

    #populate byol-resnet data
    for k in balanced_keys:
        for kk,vv in balanced_raw[k].items():
            if kk not in balanced_data.keys():
                balanced_data[kk] = []

            balanced_data[kk].append(vv.mean())

    exp2_plot_keys = ['topk_scaled_action_error', 'topk_model_position_error', 'top_integrated_cost']

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    for pi, pk in enumerate(exp2_plot_keys):
        axs[pi].set_title(pk)
        axs[pi].plot(random_data[pk], label='random')
        axs[pi].plot(baseline_data[pk], label='every tenth')
        axs[pi].plot(baseline2_data[pk], label='every')
        axs[pi].plot(balanced_data[pk], label='balanced')
        axs[pi].legend()

    plt.show()
