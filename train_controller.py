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
    candidate_model_copy = candidate_model.copy() # exact same model with copied weights

    # note that we don't have to train the candidate model first,
    # because we're using the supermodel's shared weights, which 
    # are trained later, to parameterize the candidate

    # first, we update `controller` (i.e. sampling probabilities)
    using hyperparams as hyperparameters:
        quality = candidate_model_copy.calculate_validation_accuracy()

    controller.update(reward_signal=quality)

    # then, we update the supermodel's shared weights (or at least the 
    # ones contained in `candidate_model`)
    using hyperparams as hyperparameters:
        candidate_model_copy.train(hyperparams)
    
    controller.supermodel.update(candidate_model_copy.weights)

final_model, final_hyperparams = controller.sample()
save(final_model, final_hyperparams)
'''
import torch
torch.manual_seed(42)
torch.autograd.set_detect_anomaly(True)

from module.controller.mnist_controller import MNISTController


controller = MNISTController(5, epochs=1)

i = 0 
print("Training controller...")
while not controller.has_converged():
    print("Iteration %d\n\n\n" % i, end='\r')

    # sample `controller` for a model and hyperparameters
    print("\tLoading child...")
    model_idx, hp_idx = controller.sample()
    model_state, _ = controller.archspace[int(model_idx)]
    hp_state = controller.hpspace[int(hp_idx)]
    hp_state = {'optimizer': hp_state[0], 'learning_rate': hp_state[1]}

    # note that we don't have to train the candidate model first,
    # because we're using the supermodel's shared weights,
    # which are trained later, to parameterize the candidate

    # first, we update `controller` (i.e. sampling probabilities)
    print("\tEvaluating child quality...")
    quality = controller.archspace.get_reward_signal(model_state)
    controller.update(quality, model_idx, hp_idx)

    # then, we update the supermodel's shared weights (or at least
    # the ones contained in `model_state`)
    print("\tTraining child...")
    model_state, model_dict = controller.archspace[model_idx]
    controller.archspace.train_child(model_state, hp_state)

    for layer_name in model_dict:
        # save weights layer-wise
        layer = model_state[model_dict[layer_name]]
        controller.archspace._save_layer_weights(layer, layer_name)