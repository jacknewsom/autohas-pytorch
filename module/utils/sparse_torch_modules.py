from module.utils.torch_modules import GlobalAveragePool
import torch
import sparseconvnet as scn

class SparseModel(torch.nn.Module):
    '''
    Wraps around nn.Module to allow
    something like an nn.Sequential with layers
    from scn and an nn.Linear layer from torch
    '''
    def __init__(self, input_layer, sparse_layers, pooling_layer, linear_layers):
        super(SparseModel, self).__init__()
        self.input_layer = input_layer
        self.sparse_layers = sparse_layers
        self.pooling_layer = pooling_layer
        self.linear_layers = linear_layers

    def forward(self, x):
        x = self.input_layer(x)
        x = self.sparse_layers(x)
        x = self.pooling_layer(x)
        x = self.linear_layers(x)
        return x

    def __getitem__(self, i):
        i %= (len(self.sparse_layers)+1)
        if i < len(self.sparse_layers):
            return self.sparse_layers[i]
        elif i == len(self.sparse_layers):
            return self.linear
        else:
            raise IndexError('Invalid index %d into SparseModel of length %d' % (i, len(self.sparse_layers)+1))

class SparseConv3dModule(torch.nn.Module):
    '''
    Simple wrapper to allow .load_state_dict

    Note: this module requires a `SparseConvNetTensor` as input
    '''
    def __init__(self, c_in, c_out, k):
        super(SparseConv3dModule, self).__init__()
        self.conv = scn.SubmanifoldConvolution(3, c_in, c_out, k, True)

    def forward(self, x):
        return self.conv(x)

    def input_spatial_size(self, out_size):
        return self.conv.input_spatial_size(out_size)

class SparseReLU(scn.ReLU):
    '''
    Wraps around scn.ReLU to provide `input_spatial_size`
    '''
    def input_spatial_size(self, out_size):
        return out_size