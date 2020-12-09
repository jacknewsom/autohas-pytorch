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

class SpatialPool(torch.nn.Module):
    '''
    Simple wrapper to calculate global average pooling
    (pooling across spatial dimensions)

    N.B. ~ we assume NCHW(D) 
    '''
    def forward(self, x):
        return torch.mean(x, list(range(2, x.dim())))

class ChannelPool(torch.nn.Module):
    '''
    Wrapper to calculate cross channel pooling
    '''
    def forward(self, x, channel_dim=1):
        return torch.mean(x, dim=channel_dim).reshape(x.shape[0], -1)

# Some people prefer 'global average pooling'
# but I use 'spatial pool' instead because
# I also have channel pooling and it sounds
# less ambiguous
GlobalAveragePool = SpatialPool