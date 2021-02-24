import torch
import math

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

# Some people prefer 'global average pooling'
# but I use 'spatial pool' instead because
# I also have channel pooling and it sounds
# less ambiguous
GlobalAveragePool = SpatialPool

class ChannelPool(torch.nn.Module):
    '''
    Wrapper to calculate cross channel pooling
    '''
    def forward(self, x, channel_dim=1):
        return torch.mean(x, dim=channel_dim).reshape(x.shape[0], -1)


class SinusoidalPositionalEncoding(torch.nn.Module):
    '''
    Compute positional encoding for embedding inputs to
    sequence based models
    '''
    def __init__(self, max_seq_len, d_model):
        super().__init__()

        pe = torch.zeros(max_seq_len, d_model)
        numerator = torch.arange(0, max_seq_len).unsqueeze(1)
        denominator = torch.exp(torch.arange(0, d_model, 2)*(-math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(numerator / denominator)
        pe[:, 1::2] = torch.cos(numerator / denominator)
        self.register_buffer('pe', pe)

    def forward(self, x, shape=None):
        '''
        Calculates positional encodings 
        '''
        if not shape:
            return x + self.pe[:, :x.size(1)]

        else:
            return x + self.pe.reshape(shape)