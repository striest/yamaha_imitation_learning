import os
import yaml
import torch
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

from maxent_irl_costmaps.os_utils import maybe_mkdir

class Experiment:
    """
    Wrapper around RL algorithms that drives the training and handles all the IO stuff. (i.e. making directories, saving networks, recording performance, etc.)    
    """
    def __init__(self, algo, name, params, save_to='', epochs=10, steps_per_epoch=-1, save_every=10, eval_every=10, device='cpu'):
        self.algo = algo
        self.name = '{}_{}'.format(datetime.now().strftime('%Y-%m-%d-%H-%M-%S'), name)
        self.base_fp = os.path.join(os.getcwd(), save_to, self.name)
        self.log_fp = os.path.join(self.base_fp, '_log')
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.save_every = save_every
        self.eval_every = eval_every
        self.device = device
        self.params = params

    def build_experiment_dir(self):
        if os.path.exists(self.base_fp):
            i = input('Directory {} already exists. input \'q\' to stop the experiment (and anything else to keep going).'.format(self.base_fp))
            if i.lower() == 'q':
                exit(0)
        maybe_mkdir(self.base_fp, True)

    def run(self):
        for i in range(3):
            self.algo.visualize()
            plt.show()

        train_metrics = {}
        test_metrics = {}

        for e in range(self.epochs):
            #TODO: wrap the learning here

            res = self.algo.update(self.steps_per_epoch)
            train_metrics[e] = res

            if e % self.eval_every == 0:
                eval_res = self.algo.eval()
                test_metrics[e] = eval_res

            if e == 0:
                self.build_experiment_dir()
                with open(os.path.join(self.base_fp, '_params.yaml'), 'w') as fp:
                    yaml.dump(self.params, fp, default_flow_style = False)

            if e % self.save_every == 0:
                torch.save(self.algo.to('cpu'), os.path.join(self.base_fp, "itr_{}.pt".format(e + 1)))
                torch.save({'train':train_metrics, 'test':test_metrics}, os.path.join(self.base_fp, "_metrics.pt"))
            self.algo = self.algo.to('cuda')


        for i in range(10):
            self.algo.visualize()
            plt.show()

    def to(self, device):
        self.device = device
        return self
