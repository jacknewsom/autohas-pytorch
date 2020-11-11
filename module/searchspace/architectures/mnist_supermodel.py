from module.searchspace.architectures.base_architecture_space import BaseArchitectureSpace
from collections import defaultdict
from itertools import product
import torchvision
import torch
import os

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

class MNISTSupermodel(BaseArchitectureSpace):
    '''
    Simple demonstration architecture space/supermodel for MNIST tasks

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

    data_path = './MNIST'
    train = torchvision.datasets.MNIST(
        root=data_path,
        download=True,
        train=True,
        transform=default_transforms,)
    val = torchvision.datasets.MNIST(
        root=data_path,
        download=True,
        train=False,
        transform=default_transforms)

    def __init__(
        self, 
        N, 
        weight_directory='mnistsupermodel_weights',
        num_channels=None,
        batch_size=30,
        num_train_epochs=5):
        super(MNISTSupermodel, self).__init__()

        # number of output classes
        self.num_classes = 10

        # size of input images
        self.input_size = (1, 28, 28)

        # training batch size
        self.batch_size = batch_size

        # training num epohcs
        self.num_train_epochs = num_train_epochs

        # training and validation data loaders
        self.train_loader = torch.utils.data.DataLoader(self.train, batch_size=self.batch_size)
        self.val_loader = torch.utils.data.DataLoader(self.val, batch_size=self.batch_size)

        # number of computation nodes/layers 
        self.N = N

        # define allowable computations in state space for this architecture space
        self.computations = {
            'conv3x3': lambda c_in, c_out: Conv2dModule(c_in, c_out, 3),
            'conv5x5': lambda c_in, c_out: Conv2dModule(c_in, c_out, 5),
            'maxpool3x3': lambda: torch.nn.MaxPool2d(3),
            'maxpool5x5': lambda: torch.nn.MaxPool2d(5),
            'avgpool3x3': lambda: torch.nn.AvgPool2d(3),
            'avgpool5x5': lambda: torch.nn.AvgPool2d(5),
        }
        if num_channels is None:
            # number of input and output channels required for each layer
            self.num_channels = []
            c_prev, c_current = 1, 4
            for i in range(self.N):
                channels = (c_prev, c_current)
                self.num_channels.append(channels)
                c_prev = c_current
                c_current *= 4

        self.cardinality = int((len(self.computations) ** self.N) * (1+self.N/2*(self.N+3)))

        # define all possible layer combinations for this architecture space
        # note: that for this specific supermodel, we restrict possible number 
        # of inputs for a particular layer to two (so at most one skip connection)
        # also, we use -1 to denote a layer taking the input image as an input
        if not os.path.isdir(weight_directory):
            os.mkdir(weight_directory)
        if weight_directory[-1] != '/':
           weight_directory += '/'

        self.weight_directory = weight_directory
        self.layers = defaultdict(dict)

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
            self.layers[layer_name] = layer_name
            # this layer has not been created yet
            _, ctype, c_in, c_out = layer_name.split('_')
            if ctype not in self.computations:
                raise IndexError('Invalid computation type %s' % computation_type)
            if 'conv' not in ctype:
                # is a pooling layer
                return self.computations[ctype]()
            return self.computations[ctype](int(c_in), int(c_out))

        index, ctype, c_in, c_out = layer_name.split('_')
        if 'conv' not in ctype:
            # is a pooling layer
            return self.computations[ctype]()
        layer = self.computations[ctype](int(c_in), int(c_out))
        layer.load_state_dict(self._load_layer_weights(layer_name))
        return layer

    def _load_layer_weights(self, layer_name):
        if layer_name not in os.listdir(self.weight_directory):
            raise IndexError('Layer %s does not exist' % layer_name)
        return torch.load(self.weight_directory + layer_name)

    def _save_layer_weights(self, layer, layer_name):
        torch.save(layer.state_dict(), self.weight_directory + layer_name)

    def get_child(self, state):
        '''
        Load PyTorch model encoded by state. For this specific class, states 
        are lists of `self.N` `layer_name` strings

        args:
            state: list of `self.N` `layer_name` strings

        return:
            model: torch Sequential object corresponding to model
                   specified by `state`
            weightdict: dictionary with keys equal to `layer_name`
                        and values corresponding to that layer's
                        index in `model`
        '''
        weightdict = {}
        layers = []
        c_out_prev, prev_layer_name = self.input_size[0], None
        for layer_name in state:
            layer = self.get_layer(layer_name)
            if 'conv' in layer_name:
                # Compute convolutions as ReLU - Conv - Batchnorm
                # as described in original paper
                _, _, c_in, c_out = layer_name.split('_')
                c_in, c_out = int(c_in), int(c_out)
                if c_out_prev != c_in:
                    # previous layer (or input) has incorrect number of channels,
                    # so use 1x1 convolution to fix. (Note: We follow ENAS author's
                    # strategy of 1x1 conv, ReLU, BatchNorm)
                    layers.append(Conv2dModule(c_out_prev, c_in, 1))
                    layers.append(torch.nn.ReLU(inplace=True))
                    layers.append(torch.nn.BatchNorm2d(c_in))
                layers.append(torch.nn.ReLU(inplace=True))
                layers.append(layer)
                weightdict[layer_name] = len(layers)-1
                layers.append(torch.nn.BatchNorm2d(c_out))
            else:
                layers.append(layer)
                weightdict[layer_name] = len(layers)-1
            prev_layer_name = layer_name
            c_out_prev = c_out
        layers.append(GlobalAveragePool())

        linear = torch.nn.Linear(c_out, self.num_classes)
        layers.append(linear)
        return torch.nn.Sequential(*layers), weightdict

    def train_child(self, child, hyperparameters):
        '''
        Train `child` network with `torch.nn.CrossEntropyLoss` for
        `self.num_epochs` epochs (although maybe this should be 
        choosable as a hyperparameter?)
        '''
        loss_fn = torch.nn.CrossEntropyLoss()

        optimizer_fn = hyperparameters['optimizer']
        optimizer = optimizer_fn(params=child.parameters())

        print("Training for %d epochs..." % self.num_train_epochs)
        for epoch in range(self.num_train_epochs):
            print("\nEpoch %d" % epoch)
            for i, data in enumerate(self.train_loader):
                inputs, labels = data
                optimizer.zero_grad()
                predictions = child(inputs)
                loss = loss_fn(predictions, labels)
                loss.backward()
                optimizer.step()
                complete = 100*(i+1)/(len(self.train_loader))
                if i % 100 == 0:
                    print("\t{:.3f} %  Loss {:.3f}".format(complete, loss.item()), end='\r')
            print("\t{:.3f} %  Loss {:.3f}".format(complete, loss.item()), end='\r')
        print()

    def calculate_child_validation_accuracy(self, child):
        '''
        Calculate validation accuracy for `child` network
        '''
        correct = 0
        for data in self.val_loader:
            inputs, labels = data
            predictions = child(inputs)
            predictions = torch.nn.Softmax(dim=0)(predictions)
            predictions = torch.argmax(predictions, dim=1)
            correct += torch.sum(predictions==labels)
        accuracy = int(correct) / (len(self.val_loader)*self.val_loader.batch_size)
        return accuracy