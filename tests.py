import module
print("sucessfully imported module")

from module.controller.mnist_controller import MNISTController

controller = MNISTController()
print("successfully imported and instantiated MNISTController")

from module.searchspace.hyperparameters.mnist_hyperparameter_space import MNISTHyperparameterSpace
print("successfully imported MNISTHyperparameterSpace")

import torch
optimizers = [torch.optim.Adam]
learning_rates = [0.001]
hp_space = MNISTHyperparameterSpace(optimizers, learning_rates)
print("successfully instantiated `MNISTHyperparameterSpace`")

assert torch.optim.Adam == hp_space.get_hyperparameters((0, 0))[0]
print("MNISTHyperparameterSpace.get_hyperparameters works :)")

from module.searchspace.architectures.mnist_supermodel import MNISTSupermodel
print("successfully imported MNISTSupermodel")

giselle = MNISTSupermodel(5)
print("sucessfully instatiated MNISTSupermodel")

layer_name = giselle.get_layer_name(0, 'conv3x3', 5, 10)
print("Successfully retrieved layer name")

state = ['0_conv3x3_1_10', '1_maxpool3x3_0_0', '2_conv3x3_10_20', '3_conv3x3_20_30', '4_conv3x3_30_40']
child, childdict = giselle.get_child(state)
print("successfully got child archictecture")

testinput = torch.rand((1, 1, 28, 28))
child(testinput)
print("passed `testinput` through `child`")

hps = hp_space.get_hyperparameters((0, 0))
hyperparameters = {'optimizer': lambda params: hps[0](params, hps[1])}

giselle.train_child(child, hyperparameters)
print("successfully trained `child`")

val_acc = giselle.calculate_child_validation_accuracy(child)
print("successfully calculated validation accuracy (%f)" % val_acc)

# save each layer in childdict
for layer_name in childdict:
    layer = child[childdict[layer_name]]
    giselle._save_layer_weights(layer, layer_name)
# don't forget you need to save the feedforward bit as well
feedforward_name = '{}_linear_0_0'.format(len(child)-1)
giselle._save_layer_weights(child[-1], feedforward_name)
print("successfully saved trained weights")

newchild, newchilddict = giselle.get_child(state)
for layer_name in newchilddict:
    state_dict = giselle._load_layer_weights(layer_name)
    newchild[newchilddict[layer_name]].load_state_dict(state_dict)
# don't forget to load feedforward bit
state_dict = giselle._load_layer_weights(feedforward_name)
newchild[-1].load_state_dict(state_dict)

new_val_acc = giselle.calculate_child_validation_accuracy(newchild)
assert new_val_acc > 0.9
print("successfully loaded trained weights")
