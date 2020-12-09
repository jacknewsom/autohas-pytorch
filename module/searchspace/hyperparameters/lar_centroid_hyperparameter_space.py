from module.searchspace.hyperparameters.base_hyperparameter_space import BaseHyperparameterSpace
from torch import optim
import itertools

class LArCentroidHyperparameterSpace(BaseHyperparameterSpace):
    def __init__(self, optimizers, learning_rates):
        assert type(optimizers) == list
        assert len(optimizers) == 0 or issubclass(optimizers[0], optim.Optimizer)

        assert type(optimizers) == list
        assert len(learning_rates) == 0 or isinstance(learning_rates[0], float)

        self.optimizers = optimizers
        self.learning_rates = learning_rates
        self.space = {(i,j): (self.optimizers[i], self.learning_rates[j]) for i, j in itertools.product(range(len(self.optimizers)), range(len(self.learning_rates)))}
        self.cardinality = (len(self.optimizers), len(self.learning_rates))

    def get_hyperparameters(self, state):
        if (state[0] < 0 or state[0] >= self.cardinality[0]) or \
           (state[1] < 0 or state[1] >= self.cardinality[1]):
            raise IndexError('Index {} out of bounds for state space of size {}'.format(state, self.cardinality))
        return self.space[state]

    def __getitem__(self, index):
        return self.get_hyperparameters(index)