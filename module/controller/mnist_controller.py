from module.controller.base_controller import BaseController

class MNISTController(BaseController):
    def __init__(self, **kwargs):
        super(MNISTController, self).__init__(**kwargs)

    def has_converged(self):
        return True