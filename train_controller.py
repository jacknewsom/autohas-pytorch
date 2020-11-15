'''
Strategy for training AutoHAS controller

from module.controller import Controller

# randomly initialize supermodel weights and sampling probabilities
controller = Controller()

# split data into training and validation sets
D_train, D_val = training_data.split()

while not controller.has_converged():
    # sample `controller` for a model and hyperparameters
    candidate_model, hyperparams = controller.sample()    

    # optimize `candidate_model`
    using hyperparams as hyperparameters:
        candidate_model.train(hyperparams)

    # then, we update the supermodel's shared weights (or at least the 
    # ones contained in `candidate_model`)
    controller.supermodel.update(candidate_model_copy.weights)

    # use copy of `candidate_model` to calculate `controller` gradients
    candidate_model_copy = candidate_model.copy()
    using hyperparams as hyperparameters:
        quality = candidate_model_copy.calculate_validation_accuracy()
    controller.update(reward_signal=quality)

final_model, final_hyperparams = controller.sample()
save(final_model, final_hyperparams)
'''
from module.controller.mnist_controller import MNISTController
from torch.utils.tensorboard import SummaryWriter
import torch
import copy

torch.manual_seed(42)
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
controller = MNISTController(5, device=device)
logger = SummaryWriter()

i = 0
print("Training controller...")
while not controller.has_converged():
    print("Iteration %d\n\n\n" % i, end='\r')

    # sample `controller` for a model and hyperparameters
    print("\tLoading child...")
    model_params, hp_params = controller.sample()
    hp_state = controller.hpspace[tuple([int(h) for h in hp_params])]
    hp_state = {'optimizer': hp_state[0], 'learning_rate': hp_state[1]}
    model_state, model_dict = controller.archspace[[int(i) for i in model_params]]
    print("\tChild is...")
    print(controller.archspace.child_repr(model_dict, indentation_level=2))

    # then, we update the supermodel's shared weights (or at least
    # the ones contained in `model_state`)
    print("\tTraining child...")
    controller.archspace.train_child(model_state, hp_state, indentation_level=2)

    for layer_name in model_dict:
        # save weights layer-wise
        layer = model_state[model_dict[layer_name]]
        controller.archspace._save_layer_weights(layer, layer_name)

    # then, we update `controller` using copied model(i.e. sampling probabilities)
    print("\tEvaluating child quality...", end='\r')
    model_state_copy = copy.deepcopy(model_state)
    model_params_copy, hp_params_copy = copy.deepcopy(model_params), copy.deepcopy(hp_params)
    quality = controller.archspace.get_reward_signal(model_state_copy)
    controller.update(quality, model_params_copy, hp_params_copy)
    logger.add_scalar('Accuracy/val', quality, i)
    print("\tChild quality is %f" % quality)

    i += 1