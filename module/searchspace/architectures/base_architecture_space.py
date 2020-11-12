class BaseArchitectureSpace:
    '''
    Base class for general model architecture spaces/supermodels
    '''
    def __init__(self, **kwargs):
        '''
        Create a BaseArchitectureSpace object
        '''
        super(BaseArchitectureSpace, self).__init__(**kwargs)

    def get_child(self, state):
        '''
        Get child model corresponding to `state`. In line with
        ENAS (Pham et al. 2018) and DARTS (Liu et al 2019),
        architecture spaces (aka supermodels) are large DAGs,
        where nodes represent local computations like convolution
        or max pool, while edges represent inputs to nodes

        e.g.
        >> arch_space.get_child(state)
        nn.Module subclass
        '''
        raise NotImplementedError

    def train_child(self, child, hyperparameters):
        '''
        Train `child` model returned by `self.get_architecture` using
        hyperparameters `hyperparameters`

        Should return something like a validation accuracy.
        '''
        raise NotImplementedError

    def get_reward_signal(self, child):
        '''
        Calculate reward signal associated with `child` 
        (probably after training it unless you've magically 
        figured out how to skip that part, in which case you're
        probably quite rich and not reading source code anymore
        anyway)
        '''
        raise NotImplementedError