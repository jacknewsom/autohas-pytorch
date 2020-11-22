import torch

class Policy(torch.nn.Module):
    def __init__(self, size, device):
        super(Policy, self).__init__()
        self.params = torch.nn.Parameter(torch.ones((size,), device=device))
    def forward(self):
        return torch.nn.Identity()(self.params)

class Conv2dModule(torch.nn.Module):
    '''
    Simple wrapper to allow .load_state_dict
    '''
    def __init__(self, c_in, c_out, k):
        super(Conv2dModule, self).__init__()
        self.conv = torch.nn.Conv2d(c_in, c_out, k)
    def forward(self, x):
        return self.conv(x)

class GlobalAveragePool(torch.nn.Module):
    '''
    Simple wrapper to calculate global average pooling
    as described in ENAS paper
    '''
    def forward(self, x):
        return torch.mean(x, [2, 3])