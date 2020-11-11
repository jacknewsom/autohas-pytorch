class BaseHyperparameterSpace:
    '''
    Base class for general hyperparameter space
    '''
    def __init__(self, **kwargs):
        '''
        Create a general hyperparameter space object.
        '''

        super(BaseHyperparameterSpace, self).__init__(**kwargs)

    def get_hyperparameters(self, state):
        '''
        Get hyperparameter in space corresponding to `state`

        e.g.
        >> hp_space.get_hyperparameter(15)
        {'learning_rate': 0.001}
        '''
        raise NotImplementedError