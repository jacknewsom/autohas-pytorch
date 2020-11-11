class BaseController:
    '''
    Base class for general AutoHAS controller
    '''

    def __init__(self, **kwargs):
        '''
        Create a new BaseController object.

        Subclasses need to:
        - randomly initialize architecture space sampling
          probabilities and hyperparameter space sampling
          probabilities
        - probably something else I haven't thought of yet
        '''
        super(BaseController, self).__init__(**kwargs)

    def has_converged(self) -> bool:
        '''
        Controller-specific convergence-condition check
        '''
        raise NotImplementedError

    def sample(self):
        '''
        Sample controller's architecture space and 
        hyperparameter space according to self.P
        '''        
        raise NotImplementedError

    def update(self, reward_signal):
        '''
        Update self.P using REINFORCE algorithm and 
        `reward_signal` as, well, the reward signal :)
        '''
        raise NotImplementedError