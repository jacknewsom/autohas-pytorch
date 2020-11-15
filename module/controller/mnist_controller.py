from module.controller.base_controller import BaseController
from module.searchspace.architectures.mnist_supermodel import MNISTSupermodel
from module.searchspace.hyperparameters.mnist_hyperparameter_space import MNISTHyperparameterSpace
from torch.distributions import Categorical
import torch

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
        gamma=0.99):
        super(MNISTController, self).__init__()

        # track 'convergence'
        self.converged = False

        # gpu/cpu
        self.device = device

        # track policies for archspace, hpspace
        self.policies = {'archspace': {}, 'hpspace': {'optimizers': {}, 'learning_rates': {}}}

        # discount factor
        self.gamma = gamma

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

    def update(self, trajectory):
        '''
        Perform update step of REINFORCE

        args:
            trajectory: `t` long list like [(model_params, hp_params, val_acc), ...]
        '''
        true_rewards = [i[2] for i in trajectory]
        if true_rewards[0] > 0.99:
            self.converged = True
            return

        # calculate discounted reward-to-go
        rewards_to_go = []
        R = 0
        for r in true_rewards[::-1]:
            R = r + self.gamma * R
            rewards_to_go.insert(0, R)

        # calculate log probabilities for each time step
        log_prob = []
        for t in trajectory:
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
        loss = [-r * lp for r, lp in zip(rewards_to_go, log_prob)]
        loss = torch.stack(loss).sum()
        loss.backward()
        self.optimizer.step()