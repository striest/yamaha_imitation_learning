"""
Script for converting experiment yamls into the actual objects to run experiments with
"""
import argparse
import yaml
import torch
import matplotlib.pyplot as plt

## experiment ##
from yamaha_imitation_learning.experiment_management.experiment import Experiment

## dataset ##
from yamaha_imitation_learning.dataset.bc_dataset import BCDataset

## networks ##
from yamaha_imitation_learning.networks.direct_prediction.resnet import ResnetFPV, ResnetRNNFPV

## trainers ##
from yamaha_imitation_learning.algos.behavioral_cloning import BCTrainer

from maxent_irl_costmaps.dataset.maxent_irl_dataset import MaxEntIRLDataset

def setup_experiment(fp):
    """
    Expect the following top-level keys in the YAML:
        1. experiment: high-level params such as where to save to, epochs, etc.
        2. dataset
        3. network
        4. trajopt
        5. cost_function
        6. model
        7. metrics

    Design decision to use case statements instead of dicts of class types in case I want to
    handle params in specific ways for certain classes
    """
    experiment_dict = yaml.safe_load(open(fp, 'r'))
    experiment_keys = [
        'experiment',
        'train_dataset',
        'test_dataset',
        'network',
        'netopt',
    ]
    res = {}
    #check validity of experiment YAML
    for k in experiment_keys:
        assert k in experiment_dict.keys(), "Expected key {} in yaml, found these keys: {}".format(k, experiment_dict.keys())

    #move to correct device
    device = experiment_dict['experiment']['device'] if 'device' in experiment_dict['experiment'].keys() else 'cpu'

    res['params'] = experiment_dict

    #setup dataset
    dataset_params = experiment_dict['train_dataset']
    if dataset_params['type'] == 'BCDataset':
        res['train_dataset'] = BCDataset(MaxEntIRLDataset(**dataset_params['params'])).to(device)
    else:
        print('Unsupported dataset type {}'.format(dataset_params['type']))
        exit(1)

    #setup dataset
    dataset_params = experiment_dict['test_dataset']
    if dataset_params['type'] == 'BCDataset':
        res['test_dataset'] = BCDataset(MaxEntIRLDataset(**dataset_params['params'])).to(device)
    else:
        print('Unsupported dataset type {}'.format(dataset_params['type']))
        exit(1)

    #setup network
    network_params = experiment_dict['network']
    if network_params['type'] == 'ResnetFPV':
        if network_params['base_net']:
            print('loading {}...'.format(network_params['base_net']))
            base_net = torch.load(network_params['base_net'])
            res['network'] = ResnetFPV(net=base_net, **network_params['params'])
        else:
            res['network'] = ResnetFPV(**network_params['params'])
    elif network_params['type'] == 'ResnetRNNFPV':
        if network_params['base_net']:
            print('loading {}...'.format(network_params['base_net']))
            base_net = torch.load(network_params['base_net'])
            res['network'] = ResnetRNNFPV(net=base_net, **network_params['params'])
        else:
            res['network'] = ResnetRNNFPV(**network_params['params'])
    else:
        print('Unsupported network type {}'.format(network_params['type']))
        exit(1)

    #setup network opt
    netopt_params = experiment_dict['netopt']
    if netopt_params['type'] == 'Adam':
        res['netopt'] = torch.optim.Adam(res['network'].parameters(), **netopt_params['params'])
    elif netopt_params['type'] == 'AdamW':
        res['netopt'] = torch.optim.AdamW(res['network'].parameters(), **netopt_params['params'])
    else:
        print('Unsupported netopt type {}'.format(netopt_params['type']))
        exit(1)

    #setup algo
    algo_params = experiment_dict['algo']
    if algo_params['type'] == 'BCTrainer':
        res['algo'] = BCTrainer(
            dataset = res['train_dataset'],
            test_dataset = res['test_dataset'],
            network = res['network'],
            opt = res['netopt'],
            **algo_params['params']
        ).to(device)

    #setup experiment
    experiment_params = experiment_dict['experiment']
    res['experiment'] = Experiment(
        algo = res['algo'],
        params = res['params'],
        **experiment_params
    ).to(device)

    return res

#TEST
if __name__ == '__main__':
    fp = '../../../configs/training/rnn_resnet.yaml'
    res = setup_experiment(fp)

    print({k:v.shape if isinstance(v, torch.Tensor) else v for k,v in res['test_dataset'][1].items()})

#    for i in range(10):
#        res['dataset'].visualize()
#        plt.show()

    res['experiment'].run()
