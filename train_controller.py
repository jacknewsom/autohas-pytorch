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