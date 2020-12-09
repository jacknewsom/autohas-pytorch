from module.searchspace.architectures.lar_centroid_supermodel import LArCentroidSupermodel
from module.searchspace.hyperparameters.lar_centroid_hyperparameter_space import LArCentroidHyperparameterSpace
from module.controller.base_controller import BaseController
from module.utils.torch_modules import Policy
from torch.distributions import Categorical
import numpy as np
import torch.optim as optim
import torch
import os


class LArCentroidController(BaseController):
    def __init__(
        self,
        N,
        weight_directory='larsupermodel_weights',
        num_channels=[None, None],
        batch_size=1,
        epochs=25,
        device=None,
        use_baseline=True,
        reward_map_fn=None):
        super(LArCentroidController, self).__init__()

        # model names
        self.keys = ['charge', 'light']

        # track convergence
        self.converged = False

        self.device = device

        # use average reward as baseline for rollouts
        self.use_baseline = use_baseline

        # use mapping for reward functions
        self.reward_map_fn = reward_map_fn

        # track policies for archspace, hpspace
        self.policies = {'archspace': {k: {} for k in self.keys}, 
                         'hpspace'  : {k: {'optimizers': {}, 'learning_rates': {}} for k in self.keys}}

        # architecture space/supermodel
        self.archspace = LArCentroidSupermodel(N, weight_directory, num_channels, batch_size, epochs, device)
        arch_cardinality = self.archspace.cardinality
        for k in self.keys:
            for j, c in enumerate(arch_cardinality[k]):
                self.policies['archspace']['charge'][j] = Policy(c, self.device)
                self.policies['archspace']['light'][j] = Policy(c, self.device)

        # hyperparameter space
        optimizers = [optim.Adam, optim.AdamW]
        learning_rates = [1e-2, 1e-3, 1e-4]
        self.hpspace = LArCentroidHyperparameterSpace(optimizers, learning_rates)

        for k in self.keys:
            self.policies['hpspace'][k]['optimizers'] = Policy(len(optimizers), self.device)
            self.policies['hpspace'][k]['learning_rates'] = Policy(len(learning_rates), self.device)

        self.optimizers = {}

        for k in self.keys:
            parameters = [self.policies['archspace'][k][i].parameters() for i in self.policies['archspace'][k]]
            parameters += [self.policies['hpspace'][k]['optimizers'].parameters()]
            parameters += [self.policies['hpspace'][k]['learning_rates'].parameters()]
            parameters = [{'params': p} for p in parameters]
            self.optimizers[k] = optim.AdamW(parameters)

    def has_converged(self):
        return self.converged

    def sample(self):
        layerwise_actions = []
        hp_actions = []
        for k in self.keys:
            layerwise_actions.append([])
            for i in range(len(self.archspace.cardinality[k])):
                action = Categorical(self.policies['archspace'][k][i]()).sample()
                layerwise_actions[-1].append(action)

            optimizer = Categorical(self.policies['hpspace'][k]['optimizers']()).sample()
            learning_rate = Categorical(self.policies['hpspace'][k]['learning_rates']()).sample()
            hp_actions.append([optimizer, learning_rate])
        return layerwise_actions, hp_actions

    def policy_argmax(self):
        layerwise_actions = []
        hp_actions = []
        for k in self.keys:
            layerwise_actions.append([])
            for i in range(len(self.archspace.cardinality[k])):
                action = torch.argmax(self.policies['archspace'][k][i].params)
                layerwise_actions[-1].append(action)

            optimizer = torch.argmax(self.policies['hpspace'][k]['optimizers'].params)
            learning_rate = torch.argmax(self.policies['hpspace'][k]['learning_rates'].params)
            hp_actions.append([optimizer, learning_rate])
        return layerwise_actions, hp_actions

    def update(self, rollouts):
        rewards = [i[2] for i in rollouts]

        # use reward map?
        if self.reward_map_fn:
            rewards = [{k: self.reward_map_fn(r[k]) for k in r} for r in rewards]

        # use average reward as baseline
        if self.use_baseline and len(rollouts) > 1:
            avg_reward = {k: np.mean([r[k] for r in rewards]) for k in self.keys}
            rewards = [{k: r[k]-avg_reward[k] for k in r} for r in rewards]

        for k in self.keys:
            log_prob = []
            for t in rollouts:
                _log_prob = []
                layerwise_actions, hp_actions = [q[k] for q in t[:2]]
                for i in range(len(self.policies['archspace'][k])):
                    layer_action, layer_policy = layerwise_actions[i], self.policies['archspace'][k][i]
                    _log_prob.append(Categorical(layer_policy()).log_prob(layer_action))
                for action, key in zip(hp_actions, self.policies['hpspace'][k]):
                    policy = self.policies['hpspace'][k][key]
                    _log_prob.append(Categorical(policy()).log_prob(action))
                log_prob.append(torch.stack(_log_prob).sum())

            self.optimizers[k].zero_grad()
            loss = [-r[k] * lp for r, lp in zip(rewards, log_prob)]
            loss = torch.stack(loss).sum()
            loss.backward()
            self.optimizers[k].step()

    def save_policies(self, directory='larcentroidcontroller_weights/'):
        if not os.path.isdir(directory):
            os.mkdir(directory)
        if directory[-1] != '/':
            directory += '/'
        for k in self.keys:
            for key in self.policies['archspace'][k]:
                torch.save(self.policies['archspace'][k][key].state_dict(), f'{directory}{k}_archspace_{key}')
            for key in self.policies['hpspace'][k]:
                torch.save(self.policies['hpspace'][k][key].state_dict(), f'{directory}{k}_hpspace_{key}')

    def load_policies(self, directory='larcentroidcontroller_weights/'):
        if not os.path.isdir(directory):
            raise ValueError(f'Directory {directory} does not exist')

        for k in self.keys:
            for key in self.policies['archspace'][k]:
                _ = torch.load(f'{directory}{k}_archspace_{key}')
                self.policies['archspace'][k][key].load_state_dict(_)
            for key in self.policies['hpspace'][k]:
                _ = torch.load(f'{directory}{k}_hpspace_{key}')
                self.policies['hpspace'][k][key].load_state_dict(_)