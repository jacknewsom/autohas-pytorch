from module.controller.lar_centroid_controller import LArCentroidController
from torch.utils.tensorboard import SummaryWriter
import datetime
import numpy as np
import torch
import copy

'''
controller = LArCentroidController(5)
models, hyperparams = controller.sample()

chp, lhp = [controller.hpspace.get_hyperparameters(tuple([int(h[0]), int(h[1])])) for h in hyperparams]
hyperparams = {'charge': {'optimizer': chp[0], 'learning_rate': chp[1]}, 
               'light': {'optimizer': lhp[0], 'learning_rate': lhp[1]}}
state = controller.archspace.convert_layerwise_actions_to_state(models)
charge, light = controller.archspace.get_child(state)

child = {'charge': charge[0], 'light': light[0]}
print('Charge')
print(charge[0])

print('Light')
print(light[0])

controller.archspace.train_child_sequential(child, hyperparams)
'''

random_seed = 42                                # torch seed
warmup_iterations = 0                           # number of iterations before training controller
num_rollouts_per_iteration = 1                  # number of child models evaluated before each controller update
save_policy_frequency = 5                       # number of PG updates before saving controller policies
reward_map_fn_str = 'lambda x: np.exp(-x/100)'  # save as a string so logger can log (yeah it's hacky I know)
reward_map_fn = eval(reward_map_fn_str)         # compute model quality as `reward_map_fn`(validation accuracy)

torch.manual_seed(random_seed)
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
controller = LArCentroidController(N=5, device=device, epochs=5, reward_map_fn=reward_map_fn)
logger = SummaryWriter(log_dir='larruns/' + datetime.datetime.now().strftime("%b%m_%H-%M-%S_lobster"))

logger.add_scalar('Random seed', random_seed)
logger.add_scalar('Warmup iterations', warmup_iterations)
logger.add_scalar('N rollout per iteration', num_rollouts_per_iteration)
logger.add_scalar('N training epochs', controller.archspace.epochs)
logger.add_text('Reward mapping function', reward_map_fn_str)

iteration = 0
print("Training controller...")
while not controller.has_converged():
    print(f"Iteration {iteration}\n\n", end='\r')

    rollouts = []
    for t in range(num_rollouts_per_iteration):
        print(f"\n\tTimestep {t}")
        print("\tLoading child...")
        models, hps = controller.sample()

        chp, lhp = [controller.hpspace.get_hyperparameters(tuple([int(h[0]), int(h[1])])) for h in hps]
        hyperparams = {'charge': {'optimizer': chp[0], 'learning_rate': chp[1]}, 
                       'light': {'optimizer': lhp[0], 'learning_rate': lhp[1]}}
        state = controller.archspace.convert_layerwise_actions_to_state(models)
        charge, light = controller.archspace.get_child(state)
        child = {'charge': charge[0], 'light': light[0]}
        print("\tCharge child is...")
        print(controller.archspace.child_repr(charge[1], indentation_level=2))

        print("\tLight child is...")
        print(controller.archspace.child_repr(light[1], indentation_level=2))

        print("\tTraining children...")
        controller.archspace.train_child_sequential(child, hyperparams, indentation_level=2)
        for k in controller.keys:
            model_state, model_dict = charge if k == 'charge' else light
            for layer_name in model_dict:
                layer = model_state[model_dict[layer_name]]
                controller.archspace._save_layer_weights(layer, layer_name)

        print("\tEvaluating child quality...", end='\r')
        charge_copy = copy.deepcopy(charge[0])
        charge_model_params_copy, charge_hp_params_copy = [copy.deepcopy(k) for k in [models[0], hps[0]]]
        light_copy = copy.deepcopy(light[0])
        light_model_params_copy, light_hp_params_copy = [copy.deepcopy(k) for k in [models[1], hps[1]]]
        child_copy = {'charge': charge_copy, 'light': light_copy}

        model_params_copy = {'charge': charge_model_params_copy, 'light': light_model_params_copy}
        hp_params_copy = {'charge': charge_hp_params_copy, 'light': light_hp_params_copy}

        quality = controller.archspace.get_reward_signal(child_copy)
        rollouts.append([model_params_copy, hp_params_copy, quality])
        print(f"\tChild quality is {quality}")

    if iteration >= warmup_iterations:
        print("\tUpdating controller...")
        controller.update(rollouts)

        # determine validation loss of most likely candidates
        print("\n\n\tLoading argmax children...")
        models, hps = controller.policy_argmax()

        chp, lhp = [controller.hpspace.get_hyperparameters(tuple([int(h[0]), int(h[1])])) for h in hps]
        hyperparams = {'charge': {'optimizer': chp[0], 'learning_rate': chp[1]}, 
                       'light': {'optimizer': lhp[0], 'learning_rate': lhp[1]}}
        state = controller.archspace.convert_layerwise_actions_to_state(models)
        charge, light = controller.archspace.get_child(state)
        child = {'charge': charge[0], 'light': light[0]}
        print("\tArgmax charge child is...")
        print(controller.archspace.child_repr(charge[1], indentation_level=2))

        print("\tArgmax light child is...")
        print(controller.archspace.child_repr(light[1], indentation_level=2))

        print("\tTraining argmax children...")
        controller.archspace.train_child_sequential(child, hyperparams, indentation_level=2)

        print("\tEvaluating argmax child quality", end='\r')
        quality = controller.archspace.get_reward_signal(child)

        for k in quality:
            logger.add_scalar(f'Loss/argmax_{k}', quality[k], iteration)

        if quality['charge'] < 5.0 and quality['light'] < 5.0:
            controller.converged = True

    else:
        print("\tNot updating controller")

    average_quality = {k: np.mean([r[2][k] for r in rollouts]) for k in controller.keys}

    for k in average_quality:
        print(f"\tAverage {k} child quality over rollout is {average_quality[k]}")
        logger.add_scalar(f'Loss/val_{k}', average_quality[k], iteration)

    for k in controller.keys:
        for p in controller.policies['archspace'][k]:
            params = controller.policies['archspace'][k][p].state_dict()['params']
            params /= torch.sum(params)
            logger.add_scalars(
                f'Parameters/{k} Policy {p} Normalized Parameters',
                {f'param {i}': params[i] for i in range(len(params))},
                iteration
            )
            logger.add_histogram(f'{k} Policy {p} Normalized Parameters', params, iteration)

    if iteration % save_policy_frequency == 0:
        print("\tSaving controller policies...")
        controller.save_policies()

    iteration += 1

controller.save_policies('larcentroidcontroller_weights_converged')