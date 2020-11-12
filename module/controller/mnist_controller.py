from module.controller.base_controller import BaseController
from module.searchspace.architectures.mnist_supermodel import MNISTSupermodel
from module.searchspace.hyperparameters.mnist_hyperparameter_space import MNISTHyperparameterSpace
import torch

class MNISTController(BaseController):
    def __init__(self, N, weight_directory='mnistsupermodel_weights', num_channels=None, batch_size=30, epochs=5):
        super(MNISTController, self).__init__()

        # track 'convergence'
        self.converged = False

        # parameter, probability dictionaries
        self.C = {}
        self.P = {}

        # architecture space
        self.archspace = MNISTSupermodel(N, weight_directory, num_channels, batch_size, epochs)
        self.C['archspace'] = torch.ones(size=(self.archspace.cardinality,), requires_grad=True)
        self.P['archspace'] = torch.distributions.Categorical(self.C['archspace'])

        # hyperparameter space
        optimizers = [torch.optim.Adam]
        learning_rates = [0.001]
        self.hpspace = MNISTHyperparameterSpace(optimizers, learning_rates)
        self.C['hpspace'] = torch.ones(size=self.hpspace.cardinality, requires_grad=True)
        self.P['hpspace'] = torch.distributions.Categorical(self.C['hpspace'])

        # optimizer for parameters
        self.optimizer = torch.optim.Adam((self.C['archspace'], self.C['hpspace']))

    def has_converged(self):
        return self.converged

    def sample(self):
        '''
        Randomly sample a model and set of hyperparameters from combined space
        '''
        return self.P['archspace'].sample(), self.P['hpspace'].sample()

    def update(self, val_acc, arch_index, hp_index):
        '''
        Perform update step of REINFORCE
        '''
        if val_acc > 0.99:
            self.converged = True
            return
            
        self.optimizer.zero_grad()
        loss = -val_acc * (self.P['archspace'].log_prob(arch_index) + self.P['hpspace'].log_prob(hp_index))
        loss.backward()
        self.optimizer.step()