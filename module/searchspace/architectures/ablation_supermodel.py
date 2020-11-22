from module.searchspace.architectures.base_architecture_space import BaseArchitectureSpace
from collections import OrderedDict
from itertools import product
import numpy as np
import torchvision
import functools
import torch
import os

class AblationDataset(torch.utils.data.Dataset):
    def __init__(self, train_or_val='train', transform=None):
        if train_or_val not in ['train', 'val']:
            raise ValueError('Dataset type %s not recognized' % train_or_val)

        self.transform = transform
        self.train_or_val = train_or_val

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x, y = np.zeros((1, 28, 28), dtype='float'), None

        if self.train_or_val == 'train':
            x[:, 13:15, 13:15] = 1
            y = 0
        else:
            x[:, 13:15, :] = 1
            y = 1

        if self.transform:
            x = self.transform(x)

        return x, y 

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

class AblationSupermodel(BaseArchitectureSpace):
    '''
    Simple demonstration architecture space/supermodel for ablation tasks

    Define architecture space as set of fully convolutional models with
    self.N layers and and a small feedforward layer. To this end, the
    full-size DAG that defines this state space is the 'complete' DAG
    on N nodes (node 0 connected to nodes 1-24, node 1 connected to nodes 2-24, etc.)

    Thus, we have weight matrices for each unique set of skip connections and 
    computation type. That is, there is a weight matrix for
    - layer i performing k x k convolution with inputs from layers j, k, l, ...
      (weight matrix is W_{i, (j, k, l, ...)})

    Skip connections can cause issues if they aren't handled correctly, including
    1 layer sizes incompatible
    2 layer may not have input or output

    To resolve these issues (see original NAS paper @ https://arxiv.org/pdf/1611.01578.pdf)
    - if a layer isn't connected to any input layer, then we use source as default input
    - at the final layer, we concatenate outputs of all layers that aren't used as inputs to 
      previous layers
    - if layers to be concatenated have different sizes, we pad the smaller layers with zeros

    Other notes from ENAS paper (necessary to reduce variance in gradient updates of shared
    parameters W):
    - convolutional layers are computed ReLU-Convolution-Batchnorm
    - if a layer receives skip connections from multiple previous layers, layer outputs
      are concatenated along depth dimension, then convolved with a 1x1 filter (followed by 
      BN and ReLU)
    - after final convolutional layer, avgpool across channels and feed into softmax layer
    '''

    default_transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])

    train = AblationDataset('train')
    val = AblationDataset('val')

    def __init__(
        self, 
        N, 
        weight_directory='ablationsupermodel_weights',
        num_channels=None,
        batch_size=30,
        epochs=15,
        device=None):
        super(AblationSupermodel, self).__init__()

        # number of output classes
        self.num_classes = 2

        # size of input images
        self.input_size = (1, 28, 28)

        # training batch size
        self.batch_size = batch_size

        # training num epohcs
        self.epochs = epochs

        # gpu or cpu
        self.device = device

        # training and validation data loaders
        if self.device:
            self.train_loader = torch.utils.data.DataLoader(self.train, batch_size=self.batch_size, pin_memory=True)
            self.val_loader = torch.utils.data.DataLoader(self.val, batch_size=self.batch_size, pin_memory=True)
        else:
            self.train_loader = torch.utils.data.DataLoader(self.train, batch_size=self.batch_size)
            self.val_loader = torch.utils.data.DataLoader(self.val, batch_size=self.batch_size)

        # number of computation nodes/layers 
        self.N = N

        # define allowable computations in state space for this architecture space
        self.computations = OrderedDict({
            'conv3x3': lambda c_in, c_out: Conv2dModule(c_in, c_out, 3),
            'conv5x5': lambda c_in, c_out: Conv2dModule(c_in, c_out, 5),
            'maxpool3x3': lambda: torch.nn.MaxPool2d(3),
            'maxpool5x5': lambda: torch.nn.MaxPool2d(5),
            'avgpool3x3': lambda: torch.nn.AvgPool2d(3),
            'avgpool5x5': lambda: torch.nn.AvgPool2d(5),
        })
        if num_channels is None:
            # number of input and output channels required for each layer
            self.num_channels = []
            c_prev, c_current = 1, 4
            for i in range(self.N):
                channels = (c_prev, c_current)
                self.num_channels.append(channels)
                c_prev = c_current
                c_current *= 4

        # number of choices of computation type per layer
        self.cardinality = tuple(len(self.computations) for i in range(self.N))

        # define all possible layer combinations for this architecture space
        # note: that for this specific supermodel, we restrict possible number 
        # of inputs for a particular layer to two (so at most one skip connection)
        # also, we use -1 to denote a layer taking the input image as an input
        if not os.path.isdir(weight_directory):
            os.mkdir(weight_directory)
        if weight_directory[-1] != '/':
           weight_directory += '/'

        self.weight_directory = weight_directory
        self.layers = {}

    def get_layer_name(self, layer_index, computation_type, c_in=None, c_out=None):
        if computation_type not in self.computations:
            raise IndexError('Invalid computation type %s' % computation_type)
        elif c_in is None and c_out is not None:
            raise ValueError('If c_out is provided, c_in must be provided')
        elif c_in is not None and c_out is None:
            raise ValueError('If c_in is provided, c_out must be provided')

        if c_in is None and c_out is None:
            c_in = c_out = 0
        return '{0}_{1}_{2}_{3}'.format(layer_index, computation_type, c_in, c_out)

    def get_layer(self, layer_name):
        if layer_name not in self.layers:
            # this layer has not been created yet
            _, ctype, c_in, c_out = layer_name.split('_')
            if ctype not in self.computations and ctype != 'linear':
                raise IndexError('Invalid computation type %s' % ctype)
            if 'pool' in ctype:
                # is a pooling layer
                return self.computations[ctype]()
            elif 'linear' in ctype:
                # is a linear layer
                # c_in is number of input nodes and
                # c_out is number of output nodes
                return torch.nn.Linear(int(c_in), int(c_out))
            elif 'conv' in ctype:
                return self.computations[ctype](int(c_in), int(c_out))

        index, ctype, c_in, c_out = layer_name.split('_')
        if 'pool' in ctype:
            # is a pooling layer
            layer = self.computations[ctype]()
        elif 'linear' in ctype:
            layer = torch.nn.Linear(int(c_in), int(c_out))
            layer.load_state_dict(self._load_layer_weights(layer_name))
        elif 'conv' in ctype:
            layer = self.computations[ctype](int(c_in), int(c_out))
            layer.load_state_dict(self._load_layer_weights(layer_name))
        return layer

    def convert_layerwise_actions_to_state(self, layerwise_actions):
        '''
        Takes list of integers corresponding to computation
        choice at each layer and returns list of layer names
        '''
        state = []
        computations = list(self.computations.keys())
        for i, action in enumerate(layerwise_actions):
            if action < 0 or action >= self.cardinality[i]:
               raise IndexError('Index %d out of bounds for architecture space of size %d' % (action, self.cardinality[i])) 
            index = i
            ctype = computations[action]
            if 'conv' in ctype:
                c_in, c_out = self.num_channels[i]
            else:
                c_in, c_out = 0, 0

            layer_name = '{}_{}_{}_{}'.format(index, ctype, c_in, c_out)
            state.append(layer_name)
        return state

    def _load_layer_weights(self, layer_name):
        if layer_name not in os.listdir(self.weight_directory):
            raise IndexError('Layer %s does not exist' % layer_name)
        return torch.load(self.weight_directory + layer_name)

    def _save_layer_weights(self, layer, layer_name):
        if layer_name not in self.layers:
            # only track layers if we want to save them
            self.layers[layer_name] = layer_name
        torch.save(layer.state_dict(), self.weight_directory + layer_name)

    def get_child(self, state):
        '''
        Load PyTorch model encoded by state. For this specific class, states 
        are lists of `self.N` `layer_name` strings. 

        Note: 
            - do not need to specify final feedforward layer for this supermodel.
            - the naming scheme for layers is a bit funky, the index in `layer_name`
              refers only to which of the `self.N` layers in the search space
              the `layer_name` in question specifies (otherwise they're set to -1)

              e.g. -1_conv1x1_15_25 is just an interstitial layer for fixing mismatched
                   channel inputs and outputs
                   4_conv3x3_40_620 refers to the 4th layer in the search space

        args:
            state: list of `self.N` `layer_name` strings

        return:
            model: torch Sequential object corresponding to model
                   specified by `state`
            weightdict: dictionary with keys equal to `layer_name`
                        and values corresponding to that layer's
                        index in `model`
        '''
        def layer_breaks_sequential(layers, layer):
            if type(layer) != list:
                layer = [layer]
            try:
                get_output_shape(layers+layer)
                return False
            except:
                return True

        def get_output_shape(layers):
            return torch.nn.Sequential(*layers)(torch.rand(*self.input_size)[None, ...]).shape

        # track the index of layers in the final Sequential model
        weightdict = {}
        layers = []
        c_out_prev, h_out_prev, w_out_prev = self.input_size
        for layer_name in state:
            layer = self.get_layer(layer_name)
            _, _, c_in, c_out = layer_name.split('_')
            c_in, c_out = int(c_in), int(c_out)
            if 'conv' in layer_name:
                # Compute convolutions as ReLU - Conv - Batchnorm
                # as described in original paper
                tmp_layers, tmp_layer_names, tmp_layer_idxs = [], [], []
                if c_out_prev != c_in:
                    # previous layer (or input) has incorrect number of channels,
                    # so use 1x1 convolution to fix. (Note: We follow ENAS author's
                    # strategy of 1x1 conv, ReLU, BatchNorm)
                    tmp_layers.append(Conv2dModule(c_out_prev, c_in, 1))
                    conv1x1_name = '{}_conv1x1_{}_{}'.format(-1, c_out_prev, c_in)
                    tmp_layer_names.append(conv1x1_name)
                    tmp_layer_idxs.append(len(layers))
                    tmp_layers.append(torch.nn.ReLU(inplace=True))
                    # note that we need to keep track of BatchNorm statistics as well
                    tmp_layers.append(torch.nn.BatchNorm2d(c_in))
                    bn_name = '{}_bn_{}_{}'.format(-1, c_in, c_in)
                    tmp_layer_names.append(bn_name)
                    tmp_layer_idxs.append(len(layers)+2)

                tmp_layers.append(torch.nn.ReLU(inplace=True))
                tmp_layers.append(layer)
                tmp_layer_names.append(layer_name)
                tmp_layer_idxs.append(len(layers)+len(tmp_layers)-1)
                
                tmp_layers.append(torch.nn.BatchNorm2d(c_out))
                # keep track of BatchNorm layers for future saving
                bn_name = '{}_bn_{}_{}'.format(-1, c_out, c_out)
                tmp_layer_names.append(bn_name)
                tmp_layer_idxs.append(len(layers)+len(tmp_layers)-1)
                if layer_breaks_sequential(layers, tmp_layers):
                    break
                layers += tmp_layers
                for i in range(len(tmp_layer_names)):
                    weightdict[tmp_layer_names[i]] = tmp_layer_idxs[i]
                c_out_prev = c_out
            elif 'pool' in layer_name:
                if layer_breaks_sequential(layers, layer):
                    break
                layers.append(layer)
                weightdict[layer_name] = len(layers)-1
        layers.append(GlobalAveragePool())

        # add final linear layer
        tmp = torch.nn.Sequential(*layers)
        output_shape = functools.reduce(lambda a, b: a*b, list(get_output_shape(layers)))
        linear_name = '{}_linear_{}_{}'.format(-1, output_shape, self.num_classes)
        linear = self.get_layer(linear_name)
        layers.append(linear)
        weightdict[linear_name] = len(layers)-1
        child = torch.nn.Sequential(*layers)
        if self.device:
            child.to(self.device)

        return child, weightdict

    def train_child(self, child, hyperparameters, indentation_level=0):
        '''
        Train `child` network with `torch.nn.CrossEntropyLoss` for
        `self.num_epochs` epochs
        '''
        il = "\t"*indentation_level
        child.train()
        loss_fn = torch.nn.CrossEntropyLoss()

        optimizer_fn = hyperparameters['optimizer']
        optimizer = optimizer_fn(params=child.parameters())

        print(il + "Training for %d epochs..." % self.epochs)
        correct = 0
        for epoch in range(self.epochs):
            print("\n"+ il +"Epoch %d" % epoch)
            for i, data in enumerate(self.train_loader):
                inputs, labels = data
                if self.device:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                optimizer.zero_grad()
                predictions = child(inputs.float())
                loss = loss_fn(predictions, labels)
                loss.backward()
                optimizer.step()
                predictions = torch.nn.Softmax(dim=0)(predictions)
                predictions = torch.argmax(predictions, dim=1)
                correct += torch.sum(predictions==labels)
                complete = 100*(i+1)/(len(self.train_loader))
                if i % 100 == 0:
                    print(il + "\t{:.3f} %  Loss {:.3f}".format(complete, loss.item()), end='\r')
            print(il + "\t{:.3f} %  Loss {:.3f}".format(complete, loss.item()), end='\r')
        print()
        accuracy = int(correct) / (len(self.train_loader)*self.epochs)
        return accuracy

    def calculate_child_validation_accuracy(self, child):
        '''
        Calculate validation accuracy for `child` network
        '''
        child.eval()
        correct = 0
        for data in self.val_loader:
            inputs, labels = data
            if self.device:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
            predictions = child(inputs.float())
            predictions = torch.nn.Softmax(dim=0)(predictions)
            predictions = torch.argmax(predictions, dim=1)
            correct += torch.sum(predictions==labels)
        accuracy = int(correct) / (len(self.val_loader)*self.val_loader.batch_size)
        return accuracy

    def get_reward_signal(self, child):
        '''
        Wrapper for `calculate_child_validation_accuracy`
        '''
        return self.calculate_child_validation_accuracy(child)

    def child_repr(self, child_dict, indentation_level=0):
        '''
        Nice printable version of child model
        '''
        il = '\t'*indentation_level
        nice = []
        for k in child_dict.keys():
            _, ctype, c_in, c_out = k.split('_')
            loaded = ' (New)' if k not in self.layers else ' (Old)'
            nice.append('Layer {}: {}_{}_{}'.format(len(nice), ctype, c_in, c_out) + loaded)
        nice = il + ("\n" + il).join(nice)
        return nice

    def __len__(self):
        return self.cardinality

    def __getitem__(self, i):
        return self.get_child(self.convert_layerwise_actions_to_state(i))