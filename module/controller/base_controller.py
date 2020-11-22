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
        hyperparameter space according to policies
        '''        
        raise NotImplementedError

    def policy_argmax(self):
        '''
        Returns most likely candidate model and hyperparameters
        according to policies
        '''

    def update(self, reward_signal):
        '''
        Update policies using REINFORCE algorithm and 
        `reward_signal` as, well, the reward signal :)
        '''
        raise NotImplementedError