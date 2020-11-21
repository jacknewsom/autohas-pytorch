from module.controller.base_controller import BaseController
from module.searchspace.architectures.mnist_supermodel import MNISTSupermodel
from module.searchspace.hyperparameters.mnist_hyperparameter_space import MNISTHyperparameterSpace
from torch.distributions import Categorical
import numpy as np
import torch
import os

class Policy(torch.nn.Module):
    def __init__(self, size, device):
        super(Policy, self).__init__()
        self.params = torch.nn.Parameter(torch.ones((size,), device=device))
    def forward(self):
        return torch.nn.Identity()(self.params)

class MNISTController(BaseController):
    def __init__(
        self, 
        N, 
        weight_directory='mnistsupermodel_weights', 
        num_channels=None, 
        batch_size=30, 
        epochs=15,
        device=None,
        use_baseline=True,
        reward_map_fn=None):
        super(MNISTController, self).__init__()

        # track 'convergence'
        self.converged = False

        # gpu/cpu
        self.device = device

        # use average reward as baseline for rollouts
        self.use_baseline = use_baseline

        # use mapping for reward functions
        self.reward_map_fn = reward_map_fn

        # track policies for archspace, hpspace
        self.policies = {'archspace': {}, 'hpspace': {'optimizers': {}, 'learning_rates': {}}}

        # architecture space
        self.archspace = MNISTSupermodel(N, weight_directory, num_channels, batch_size, epochs, device)
        n_computations = len(self.archspace.computations)
        for i in range(N):
            self.policies['archspace'][i] = Policy(n_computations, self.device)

        # hyperparameter space
        optimizers = [torch.optim.Adam]
        learning_rates = [0.001]
        self.hpspace = MNISTHyperparameterSpace(optimizers, learning_rates)
        # these should probably be OrderedDicts to make life easier
        self.policies['hpspace']['optimizers'] = Policy(len(optimizers), self.device)
        self.policies['hpspace']['learning_rates'] = Policy(len(learning_rates), self.device)

        parameters = [self.policies['archspace'][i].parameters() for i in self.policies['archspace']]
        parameters += [self.policies['hpspace']['optimizers'].parameters()]
        parameters += [self.policies['hpspace']['learning_rates'].parameters()]
        parameters = [{'params': p} for p in parameters]

        # optimizer for parameters
        self.optimizer = torch.optim.Adam(parameters)

    def has_converged(self):
        return self.converged

    def sample(self):
        '''
        Randomly sample a model and set of hyperparameters from combined space
        '''
        layerwise_actions = []
        for i in range(self.archspace.N):
            action = Categorical(self.policies['archspace'][i]()).sample()
            layerwise_actions.append(action)
        
        optimizer = Categorical(self.policies['hpspace']['optimizers']()).sample()
        learning_rate = Categorical(self.policies['hpspace']['learning_rates']()).sample()
        hp_actions = [optimizer, learning_rate]
        return layerwise_actions, hp_actions

    def policy_argmax(self):
        '''
        Return most likely candidate model and hyperparameters from combined space
        '''
        layerwise_actions = []
        for i in range(self.archspace.N):
            action = torch.argmax(self.policies['archspace'][i].params)
            layerwise_actions.append(action)

        optimizer = torch.argmax(self.policies['hpspace']['optimizers'].params)
        learning_rate = torch.argmax(self.policies['hpspace']['learning_rates'].params)
        hp_actions = [optimizer, learning_rate]
        return layerwise_actions, hp_actions

    def update(self, rollouts):
        '''
        Perform update step of REINFORCE

        args:
            rollouts: `n` long list like [(model_params, hp_params, quality), ...]
        '''
        rewards = [i[2] for i in rollouts]

        # exponentiate rewards 
        if self.reward_map_fn
            rewards = [self.reward_map_fn(r) for r in rewards]

        # calculate rewards using average reward as baseline
        if self.use_baseline and len(rollouts) > 1:
            avg_reward = np.mean(rewards)
            rewards = [r-avg_reward for r in rewards]

        # calculate log probabilities for each time step
        log_prob = []
        for t in rollouts:
            _log_prob = []
            layerwise_actions, hp_actions = t[:2]
            for i in range(len(self.policies['archspace'])):
                layer_action, layer_policy = layerwise_actions[i], self.policies['archspace'][i]
                _log_prob.append(Categorical(layer_policy()).log_prob(layer_action))
            for action, key in zip(hp_actions, self.policies['hpspace']):
                policy = self.policies['hpspace'][key]
                _log_prob.append(Categorical(policy()).log_prob(action))
            log_prob.append(torch.stack(_log_prob).sum())

        self.optimizer.zero_grad()
        loss = [-r * lp for r, lp in zip(rewards, log_prob)]
        loss = torch.stack(loss).sum()
        loss.backward()
        self.optimizer.step()

    def save_policies(self, directory='mnistcontroller_weights/'):
        if not os.path.isdir(directory):
            os.mkdir(directory)
        if directory[-1] != '/':
            directory += '/'
        for k in self.policies['archspace']:
            torch.save(self.policies['archspace'][k].state_dict(), directory + 'archspace_' + str(k))
        for k in self.policies['hpspace']:
            torch.save(self.policies['hpspace'][k].state_dict(), directory + 'hpspace_' + k)

    def load_policies(self, directory='mnistcontroller_weights/'):
        if not os.path.isdir(directory):
            raise ValueError('Directory %s does not exist' % directory)

        if directory[-1] != '/':
            directory += '/'
        for k in self.policies['archspace']:
            _ = torch.load(directory + 'archspace_' + str(k))
            self.policies['archspace'][k].load_state_dict(_)
        for k in self.policies['hpspace']:
            _ = torch.load(directory + 'hpspace_' + k)
            self.policies['hpspace'][k].load_state_dict(_)