from module.searchspace.architectures.base_architecture_space import BaseArchitectureSpace
from module.utils.torch_modules import SpatialPool, ChannelPool
from module.utils.sparse_torch_modules import SparseConv3dModule, SparseReLU, SparseModel
from module.utils.muon_track_dataset import MuonTracks
from collections import OrderedDict
from functools import reduce
from operator import mul
import sparseconvnet as scn
import numpy as np
import torchvision
import torch
import os

class LArCentroidSupermodel(BaseArchitectureSpace):
    '''
    Architecture space for identifying centroids of muon tracks in 
    liquid argon

    Here, we define the architecture space as the set of pairs of
    convolutional models: one for processing light signal and one
    for processing charge signal. Each will take 3D images as input
    and produce a point in R^3.

    Some key points:
    - light and charge signals are both three-dimensional, but are
      very sparse. Using standard convolutions instead of sparse
      ones will be very wasteful
    - Our objective is for these networks to learn to predict the
      same centroid despite their different data sources, so loss
      ought to be something that increases as the distance between
      these predictions increases

    Search space is as follows:
    - `self.N` choices of spatial computations, `len(self.computations)` each
    - two choices of pooling strategies: 'channel pooling' and 'global pooling'
      and 12 choices of two linear layers' hidden 

    '''

    default_transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])

    train_data_path = './LArCentroids/train/'
    val_data_path = './LArCentroids/val/'
    train = MuonTracks(train_data_path)
    val = MuonTracks(val_data_path)

    def __init__(
        self,
        N,
        weight_directory='larsupermodel_weights',
        num_channels=[None, None],
        batch_size=1,
        epochs=15,
        device=None,
        ):
        super(LArCentroidSupermodel, self).__init__()

        # maximum number of conv layers in each network
        # should be tuple like (N_charge, N_light)
        if type(N) == int:
            N = (N, N)
        self.N = N

        # number of output logits
        self.output_dim = 1

        # training batch size
        self.batch_size = batch_size

        # network names
        self.networks = ['charge', 'light']

        # training epochs
        self.epochs = epochs

        # gpu or cpu
        self.device = device

        # training and validation data loaders
        pin_memory = (self.device!=None)
        self.train_loader = torch.utils.data.DataLoader(
            self.train, 
            batch_size=batch_size,
            pin_memory=pin_memory)
        self.val_loader = torch.utils.data.DataLoader(
            self.val,
            batch_size=self.batch_size,
            pin_memory=pin_memory)

        # define allowable computations in state space for this architecture
        # note that light and charge networks are allowed same computation set
        self.computations = OrderedDict({
            'conv3': lambda c_in, c_out: SparseConv3dModule(c_in, c_out, 3),
            'conv5': lambda c_in, c_out: SparseConv3dModule(c_in, c_out, 5),
            'maxpool3': lambda: scn.MaxPooling(3, 3, 3),
            'maxpool5': lambda: scn.MaxPooling(3, 5, 5),
            'avgpool3': lambda: scn.AveragePooling(3, 3, 3),
            'avgpool5': lambda: scn.AveragePooling(3, 5, 5),
            'identity': lambda: scn.Identity(),
        })

        # automatically filter out sipm active sites with feature value 
        # less than or equal to this (0 means no pruning)
        self.sipm_min_photons = 1

        # maximum number of active sites for sipm model 
        # only included for numerical stability
        self.max_sipm_active_sites = 250

        # num_channels should be list like 
        # [num_channels_charge, num_channels_light]
        self.num_channels = {}
        for k in self.networks:
            for i, channels in enumerate(num_channels):
                if channels is None:
                    channels = []
                    c_prev, c_current = 1, 4
                    for j in range(self.N[i]):
                        pair = (c_prev, c_current)
                        channels.append(pair)
                        c_prev = c_current
                        c_current *= 4
            self.num_channels[k] = channels

        # desired spatial size of output of sparse conv layers
        # should change this to a model parameter
        self.output_spatial_size = [10, 10, 10]

        self.pooling_layers = OrderedDict({
            'spatialpool': SpatialPool(), 
            'channelpool': ChannelPool(),
        })
        self.hidden_layer_sizes = [10, 10**2, 10**3]

        charge_cardinality = (
            *tuple(len(self.computations) for i in range(self.N[0])), # number of computations per layer in sparse network
            len(self.pooling_layers),                                 # number of pooling choices for post-sparse network
            len(self.hidden_layer_sizes),                             # number of choices of hidden layer size for linear layers
        )
        light_cardinality = (
            *tuple(len(self.computations) for i in range(self.N[1])),
            len(self.pooling_layers),
            len(self.hidden_layer_sizes),
        )
        self.cardinality = {'charge': charge_cardinality, 'light': light_cardinality}

        if not os.path.isdir(weight_directory):
            os.mkdir(weight_directory)
        if weight_directory[-1] != '/':
            weight_directory += '/'

        self.weight_directory = weight_directory
        self.layers = {}

    def get_layer_name(self, network, layer_index, computation_type, c_in=None, c_out=None):
        if network not in self.networks:
            raise IndexError('Network type {} not one of recognized types {}'.format(network, self.networks))
        elif computation_type not in self.computations:
            raise IndexError('Invalid computation type %s' % computation_type)
        elif c_in is None and c_out is not None:
            raise ValueError('If c_out is provided, c_in must be provided')
        elif c_in is not None and c_out is None:
            raise ValueError('If c_in is provided, c_out must be provided')

        if c_in is None and c_out is None:
            c_in = c_out = 0
        return '{}_{}_{}_{}_{}'.format(network, layer_index, computation_type, c_in, c_out)

    def get_layer(self, layer_name):
        network, index, ctype, c_in, c_out = layer_name.split('_')
        if ctype == 'spatialpool':
            return self.pooling_layers[ctype]
        elif ctype == 'channelpool':
            return self.pooling_layers[ctype]
        elif ctype == 'identity':
            return self.computations[ctype]()

        if layer_name not in self.layers:
            # this layer has not been created yet
            if network not in self.networks:
                raise IndexError('Network type {} not one of recognized types {}'.format(network, self.networks))
            elif ctype not in self.computations and ctype != 'linear':
                raise IndexError('Invalid computation type %s' % ctype)

            if 'pool' in ctype:
                # is a pooling layer
                return self.computations[ctype]()
            elif 'linear' in ctype:
                # is a linear layer
                # c_in is number of input features
                # c_out is number of output features
                return torch.nn.Linear(int(c_in), int(c_out))
            elif 'conv' in ctype:
                return self.computations[ctype](int(c_in), int(c_out))

        if 'pool' in ctype:
            layer = self.computations[ctype]()
        elif 'linear' in ctype:
            layer = torch.nn.Linear(int(c_in), int(c_out))
            layer.load_state_dict(self._load_layer_weights(layer_name))
        elif 'conv' in ctype:
            layer = self.computations[ctype](int(c_in), int(c_out))
            layer.load_state_dict(self._load_layer_weights(layer_name))
        return layer

    def _load_layer_weights(self, layer_name):
        if layer_name not in os.listdir(self.weight_directory):
            raise IndexError('Layer %s does not exist' % layer_name)
        return torch.load(self.weight_directory + layer_name)

    def _save_layer_weights(self, layer, layer_name):
        if layer_name not in self.layers:
            # only track layers if we want to save them
            self.layers[layer_name] = layer_name
        torch.save(layer.state_dict(), self.weight_directory + layer_name)

    def convert_layerwise_actions_to_state(self, layerwise_actions):
        if len(layerwise_actions) != 2:
            raise ValueError(f'Two lists required, but {len(layerwise_actions)} provided')
        if len(layerwise_actions[0]) != len(self.cardinality['charge']):
            raise ValueError(f'List of length {0} required, but {1} provided')
        if len(layerwise_actions[1]) != len(self.cardinality['light']):
            raise ValueError(f'List of length {0} required, but {1} provided')

        states = []
        for k, modelwise_actions in zip(self.networks, layerwise_actions):
            state = []
            c_out_prev = 1
            for i, action in enumerate(modelwise_actions):
                if action < 0 or action >= len(self.cardinality[k]):
                    raise IndexError(f'Index {action} out of bounds for architecture space dimension of size {len(self.cardinality[k])}')
                elif i == len(self.cardinality[k])-2:
                    # pooling layer choice
                    computations = list(self.pooling_layers.keys())
                    ctype = computations[action]
                    c_in = c_out_prev
                    if ctype == 'channelpool':
                        # pool across channel dimension
                        c_out = reduce(mul, self.output_spatial_size)
                    elif ctype == 'spatialpool':
                        # pool across spatial dimensions
                        c_out = c_out_prev
                    idx = -1
                elif i == len(self.cardinality[k])-1:
                    # hidden layer size
                    ctype = 'linear'
                    c_in = c_out_prev
                    c_out = self.hidden_layer_sizes[action]
                    idx = -1
                else:
                    # sparse computations
                    computations = list(self.computations.keys())
                    ctype = computations[action]
                    if 'conv' in ctype:
                        c_in, c_out = self.num_channels[k][i]
                    else:
                        c_in, c_out = c_out_prev, c_out_prev
                    idx = i
                layer_name = '{}_{}_{}_{}_{}'.format(k, idx, ctype, c_in, c_out)
                state.append(layer_name)

                c_out_prev = c_out
            states.append(state)
        return states

    def get_child(self, state):
        '''
        Load Torch model encoded by state. We encode models in a list of lists:
        one with `self.cardinality['charge'] `layer_name` strings for charge child, and 
        one with `self.cardinality['light']` `layer_name` strings for light child, where
        second-to-last item is choice of pooling layer, and last item is 
        hidden layer size.

        Note:
            - do not need to specify final linear layer in this supermodel.
            - naming scheme is very similar as in `mnist_supermodel`, except
              there is an extra `network` field at beginning of `layer_name`
              strings to specify which of the charge, light networks a layer
              is meant to be used in. We keep these separate because we do
              not expect weight sharing to be useful across these networks
            - sparse convolutions have no notion of height, width, depth, etc.,
              so we don't need to worry about whether or not adding an additional
              strided convolution or pooling layer will reduce the size of the
              output to none

        args:
            state: list of lists of `layer_name` strings

        return:
            charge_model: torch Sequential object corresponding to model
                          specified by `state[0]`
            charge_weightdict: dictionary with keys for each `layer_name`
            light_model: see above
            light_weightdict: see above
        '''
        if len(state) != 2:
            raise ValueError(f'Two state lists required, but {len(state)} were provided')
        elif len(state[0]) != len(self.cardinality['charge']):
            raise ValueError(f'First state list should be {self.N[0]+2} long, but was {len(state[0])} long')
        elif len(state[1]) != len(self.cardinality['light']):
            raise ValueError(f'Second state list should be {self.N[1]+2} long, but was {len(state[1])} long')

        def _get_child(name, state):
            weightdict = {}
            layers = []
            c_out_prev = 1
            state, pooling_layer, hidden_layer = state[:-2], state[-2], state[-1]
            for i, layer_name in enumerate(state):
                layer = self.get_layer(layer_name)
                _, _, _, c_in, c_out = layer_name.split('_')
                c_in, c_out = int(c_in), int(c_out)
                if 'conv' in layer_name:
                    # compute convolutions as ReLU - Conv - Batchnorm
                    tmp_layers, tmp_layer_names, tmp_layer_idxs = [], [], []
                    if c_out_prev != c_in:
                        # previous layer/input has incorrect number of channels,
                        # so use 1x1x1 convolution to fix
                        conv1x1x1_name = '{}_{}_conv1x1x1_{}_{}'.format(name, -1, c_out_prev, c_in)
                        conv1x1x1 = SparseConv3dModule(c_out_prev, c_in, 1)
                        layers.append(conv1x1x1)
                        weightdict[conv1x1x1_name] = len(layers)-1

                        tmp_layers.append(SparseReLU())

                        if self.batch_size != 1:
                            # note that we need to track batchnorm stats as well
                            bn_name = '{}_{}_bn_{}_{}'.format(name, -1, c_in, c_in)
                            layers.append(scn.BatchNormalization(c_in))
                            weightdict[bn_name] = len(layers)-1
                        
                    if i > 0:
                        layers.append(SparseReLU())

                    layers.append(layer)
                    weightdict[layer_name] = len(layers)-1

                    if self.batch_size != 1:
                        bn_name = '{}_{}_bn_{}_{}'.format(name, -1, c_out, c_out)
                        layers.append(scn.BatchNormalization(c_out))
                        weightdict[bn_name] = len(layers)-1
    
                    c_out_prev = c_out
                elif 'pool' in layer_name:
                    layers.append(layer)
                    weightdict[layer_name] = len(layers)-1

            # convert back to dense
            layers.append(scn.SparseToDense(3, c_out))

            # here we force output size of sparse network to be 10x10x10, but this could in principle
            # be another dimension in the search space 
            spatial_size = scn.Sequential(*layers).input_spatial_size(torch.LongTensor(self.output_spatial_size))
            input_layer = scn.InputLayer(3, spatial_size)
            layers.insert(0, input_layer)

            # gotta shift all layers up by one because we put `input_layer` at the front
            for k in weightdict:
                weightdict[k] += 1

            pooling_choice = pooling_layer.split('_')[2]
            pooling_layer = self.pooling_layers[pooling_choice]
            layers.append(pooling_layer)

            if pooling_choice == 'channelpool':
                c_out = reduce(mul, self.output_spatial_size)

            hidden_layer_size = int(hidden_layer.split('_')[-1])
            linear1_name = '{}_{}_linear_{}_{}'.format(name, -1, c_out, hidden_layer_size)
            linear1 = self.get_layer(linear1_name)
            layers.append(linear1)
            weightdict[linear1_name] = len(layers)-1

            layers.append(torch.nn.ReLU())

            linear2_name = '{}_{}_linear_{}_{}'.format(name, -1, hidden_layer_size, self.output_dim)
            linear2 = self.get_layer(linear2_name)
            layers.append(linear2)
            weightdict[linear2_name] = len(layers)-1

            child = scn.Sequential(*layers)
            return child, weightdict

        charge_child, charge_weightdict = _get_child('charge', state[0])
        light_child, light_weightdict = _get_child('light', state[1])
        return (charge_child, charge_weightdict), (light_child, light_weightdict)

    def _load_data_from_dataset(self, dataset, index, src):
        if src not in ['sipm', 'energy']:
            raise IndexError(f'Data source {src} not recognized. Must be one of ["sipm", "energy"]')
        data = dataset[index]
        locations = torch.LongTensor(data[f'{src}_coordinates']).to(self.device)
        features = torch.FloatTensor(data[f'{src}_values']).reshape(-1, 1).to(self.device)

        if src == 'sipm':
            min_sipm = self.sipm_min_photons
            locations = locations[(features>min_sipm).reshape(-1)] 
            features = features[features>min_sipm].reshape(-1, 1)
        if len(locations) > self.max_sipm_active_sites:
            active_sites = np.random.choice(len(locations), self.max_sipm_active_sites, replace=False)
            locations, features = locations[active_sites], features[active_sites]

        target = torch.FloatTensor(data['target'].reshape(1, 1)).to(self.device)
        return locations, features, target

    def train_child_parallel(self, child, hyperparameters, indentation_level=0):
        '''
        Train `child['charge']` and `child['light']` networks with
        `torch.nn.MSELoss` for `self.num_epochs`

        ~ Batching not currently supported because PyTorch dataloaders
          don't like it when batch elements have different shapes
        '''
        il = '\t'*indentation_level
        charge, light = child['charge'], child['light']

        charge.train()
        light.train()

        if self.device:
            charge.to(self.device)
            light.to(self.device)

        loss_fn = torch.nn.L1oss()

        charge_optimizer_fn = hyperparameters['charge']['optimizer']
        charge_lr = hyperparameters['charge']['learning_rate']
        charge_optimizer = charge_optimizer_fn(charge.parameters(), lr=charge_lr)

        light_optimizer_fn = hyperparameters['light']['optimizer']
        light_lr = hyperparameters['light']['learning_rate']
        light_optimizer = light_optimizer_fn(light.parameters(), lr=light_lr)

        for epoch in range(self.epochs):
            print("\n"+il+"Epoch %d"%epoch)
            for i in range(len(self.train)):
                losses = []
                for src, (model, optim) in zip(['energy', 'sipm'], [(charge, charge_optimizer), (light, light_optimizer)]):
                    locations, features, target = self._load_data_from_dataset(self.train, i, src)

                    prediction = model([locations, features])
                    loss = loss_fn(prediction, target)
                    optim.zero_grad()
                    loss.backward()
                    optim.step()
                    losses.append(loss.item())
                complete = 100*(i+1)/len(self.train)
                if i%100 == 0:
                    print(il + "\t{:.3f} % CLoss {:.3f} LLoss {:.3f}".format(complete, losses[0], losses[1]), end='\r')
            print(il + "\t{:.3f} % CLoss {:.3f} LLoss {:.3f}".format(complete, losses[0], losses[1]), end='\r')
        print()

    def train_child_sequential(self, child, hyperparameters, indentation_level=0):
        '''
        Train `child['charge']` and `child['light']` networks with
        `torch.nn.MSELoss` for `self.num_epochs`

        ~ Batching not currently supported because PyTorch dataloaders
          don't like it when batch elements have different shapes
        '''
        il = '\t'*indentation_level
        loss_fn = torch.nn.L1Loss()

        for model_name in child:
            print("\n"+il+"Training {} child".format(model_name))
            model = child[model_name]
            model.train()

            optimizer_fn = hyperparameters[model_name]['optimizer']
            lr = hyperparameters[model_name]['learning_rate']
            optim = optimizer_fn(model.parameters(), lr=lr)
            model.to(self.device)

            src = 'sipm' if model_name == 'light' else 'energy'

            total_loss = 0
            for epoch in range(self.epochs):
                print("\n"+il+"Epoch %d"%epoch)
                for i in range(len(self.train)):
                    locations, features, target = self._load_data_from_dataset(self.train, i, src)

                    prediction = model([locations, features])
                    loss = loss_fn(prediction, target)
                    optim.zero_grad()
                    loss.backward()
                    optim.step()

                    total_loss += loss.item()

                    complete = 100*(i+1)/len(self.train)
                    if i%100 == 0:
                        print(il + "\t{:.3f}% Loss {:.3f}".format(complete, loss.item()), end='\r')
                total_loss /= len(self.train)
                print(il + "\t{:.3f}% Average loss {:.3f}".format(complete, total_loss), end='\r')
            print()

    def calculate_child_validation_loss(self, child):
        '''
        Calculate validation loss for `child['charge']` and `child['light']`
        networks
        '''
        loss_fn = torch.nn.L1Loss()
        losses = {}

        for model_name in child:
            model = child[model_name]
            model.eval()
            model.to(self.device)

            src = 'sipm' if model_name == 'light' else 'energy'
            for i in range(len(self.val)):
                locations, features, target = self._load_data_from_dataset(self.val, i, src)

                prediction = model([locations, features])
                loss = loss_fn(prediction, target)

                if model_name not in losses:
                    losses[model_name] = loss.item()
                else:
                    losses[model_name] += loss.item()

            losses[model_name] /= len(self.val)
        return losses

    def get_reward_signal(self, child):
        losses = self.calculate_child_validation_loss(child)
        return losses

    def child_repr(self, child_dict, indentation_level=0):
        il = '\t'*indentation_level
        nice = []
        for k in child_dict.keys():
            _, _, ctype, c_in, c_out = k.split('_')
            loaded = ' (New)' if k not in self.layers else ' (Old)'
            nice.append('Layer {}: {}_{}_{}'.format(len(nice), ctype, c_in, c_out) + loaded)
        nice = il + ("\n" + il).join(nice)
        return nice

    def __len__(self):
        return self.cardinality

    def __getitem__(self, i):
        return self.get_child(self.convert_layerwise_actions_to_state(i))