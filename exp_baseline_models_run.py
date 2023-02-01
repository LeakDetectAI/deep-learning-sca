import argparse
import copy
import json
import os
import time
from datetime import timedelta

import keras_tuner as kt
import numpy as np
from keras.saving.save import load_model
from sklearn.model_selection import train_test_split

from deepscapy.constants import *
from deepscapy.experimental_utils import *
from deepscapy.hp_tuner import HPOModelLoss
from deepscapy.hp_tuner.hpo_model_loss import HPOModelTuner
from deepscapy.models import *
from deepscapy.utils import *


def _load_attack_model(dataset, model_file, model_name, loss_function):
    custom_objects = {}
    try:
        if '_tuned' in model_name:
            best_hp_dict = load_best_hp_to_dict(model_name=model_name)

            if loss_function == RANKING_LOSS:
                params = dict(alpha_value=best_hp_dict['alpha_value'])
                custom_objects = {'loss': loss_dictionary_attack_models[loss_function](**params)}
            elif loss_function in [FOCAL_LOSS_BE, FOCAL_LOSS_CE]:
                params = dict(alpha=best_hp_dict['alpha'], gamma=best_hp_dict['gamma'],
                              from_logits=best_hp_dict['from_logits'])
                custom_objects = {'loss': loss_dictionary_attack_models[loss_function](**params)}
            attack_model = load_model(model_file, custom_objects=custom_objects)

        else:
            if loss_function == CATEGORICAL_CROSSENTROPY_LOSS:
                attack_model = load_model(model_file)
            elif loss_function == RANKING_LOSS:
                custom_objects = {'loss': loss_dictionary_rkl_models[dataset]}
                attack_model = load_model(model_file, custom_objects=custom_objects)
            else:
                custom_objects = {'loss': loss_dictionary_train_models[loss_function]}
                attack_model = load_model(model_file, custom_objects=custom_objects)
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        print(str(e))
        attack_model = None
    return attack_model


def check_if_model_trained(model_name, loss_function, dataset):
    if '_tuned' not in model_name:
        model_file = os.path.join(get_trained_models_path(), '{}.tf'.format(model_name))
    else:
        model_file = os.path.join(get_trained_models_path(TRAINED_MODELS_TUNED), '{}.tf'.format(model_name))
    if_exist = os.path.isdir(model_file)
    attack_model = _load_attack_model(dataset, model_file, model_name, loss_function)
    if_load_model = attack_model is not None

    if not if_exist:
        print("Model {} was not saved so retraining the model".format(model_name))
    else:
        print("Model {} was stored".format(model_name))

    if not if_load_model:
        print("Model {} was not stored and could not be loaded properly so retraining the model".format(model_name))
    else:
        print("Model {} was stored and was loaded properly".format(model_name))
    return if_load_model


if __name__ == "__main__":
    # Make results deterministic
    seed = 1234
    os.environ['PYTHONHASHSEED'] = str(seed)
    # Argument parser
    parser = argparse.ArgumentParser(description='Model HP Tuner & Model Training')
    parser.add_argument('--dataset', type=str, required=True,
                        help='An ASCAD dataset for the nas_attack. Possible values are ASCAD_desync0_variable, ASCAD_desync50_variable, ASCAD_desync100, AES_HD, AES_RD, DP4CONTEST.')
    parser.add_argument('--model_name', type=str, required=True,
                        help='Name of the model to use. Possible values are ascad_mlp_baseline, ascad_cnn_baseline, ascad_cnn_rkl_baseline, lenet5, nas_basic, nas_lenet5.')
    parser.add_argument('--loss_function', type=str, required=True,
                        help='Loss function for the models. Possible values are categorical_crossentropy, ranking_loss, cross_entropy_ratio, dice_bce_loss, sigmoid_focal_binary_crossentropy, sigmoid_focal_categorical_crossentropy')
    parser.add_argument('--epochs', default=200, type=int,
                        help='Epochs for the tuner hypermodel or the model. Default value is 150.')
    parser.add_argument('--batch_size', default=200, type=int,
                        help='Batch Size for the tuner hypermodel or the model. Default value is 200.')
    parser.add_argument('--weight_averaging', default=False, action=argparse.BooleanOptionalAction,
                        help='Whether to use early stopping or not for the tuner hypermodel or the model. Default value is True.')
    parser.add_argument('--use_tuner', default=False, action=argparse.BooleanOptionalAction,
                        help='Whether to use keras tuner or not before training the model. Default value is True.')
    parser.add_argument('--leakage_model', default='ID', type=str,
                        help=' The type of leakage one wants to exploit to attack the system. ID, HW. Default value is ID.')

    arguments = parser.parse_args()
    print(arguments)
    dataset_name = arguments.dataset
    args_model_name = arguments.model_name
    model_name = args_model_name
    epochs = arguments.epochs
    batch_size = arguments.batch_size
    weight_averaging = arguments.weight_averaging
    use_tuner = arguments.use_tuner
    loss_function = arguments.loss_function
    leakage_model = arguments.leakage_model

    # Load dataset
    dataset_reader_class = datasets[dataset_name]
    dataset_reader_obj = dataset_reader_class(dataset_type=dataset_name, leakage_model=leakage_model)

    (X_profiling, Y_profiling), (X_attack, Y_attack) = dataset_reader_obj.get_train_test_dataset()
    X_profiling_train, X_profiling_val, Y_profiling_train, Y_profiling_val = train_test_split(X_profiling, Y_profiling,
                                                                                              test_size=0.1)

    # Define the model parameters
    if model_name == CNN_ZAID_BASELINE:
        model_class = cnn_rkl_baseline_dictionary[dataset_name]
    else:
        model_class = model_dictionary[model_name]

    if model_name == CNN_ZAID_BASELINE and (dataset_name == DP4_CONTEST or dataset_name == AES_HD or \
                                            dataset_name == ASCAD_DESYNC0 or dataset_name == ASCAD_DESYNC0_VARIABLE):
        X_profiling, X_attack = standardize_features(X_profiling, X_attack, standardize='standard')
        X_profiling, X_attack = standardize_features(X_profiling, X_attack, standardize='minmax')

    input_dim = X_profiling.shape[1]
    num_classes = len(np.unique(Y_profiling))
    # loss_function = loss_dictionary_hpo_models[args.loss_function]

    metrics = ['accuracy']

    learner_params = {'num_classes': num_classes, 'metrics': metrics, 'input_dim': input_dim}

    if model_class == ASCADMLPBaseline:
        learner_params['n_units'] = 200
        learner_params['n_hidden'] = 6

    if model_class == CustomLeNet5:
        learner_params['dataset_type'] = dataset_name

    model_name = '{}_{}_{}_{}'.format(dataset_name.lower(), args_model_name, input_dim, LF_EXTENSION[loss_function])
    model_condition = (model_class in [CNNRankingLossASCADDesync0Baseline, CNNRankingLossASCADDesync50Baseline,
                                       CNNRankingLossASCADDesync100Baseline, CNNRankingLossAESHDBaseline,
                                       CNNCCEAESRDBaseline, CNNCCEDPACV4Baseline])
    if weight_averaging:
        model_name = '{}_{}'.format(model_name, 'weight_averaging')
    if use_tuner:
        if loss_function in [FOCAL_LOSS_BE, FOCAL_LOSS_CE, RANKING_LOSS]:
            model_name = '{}_{}'.format(model_name, 'tuned')

    if leakage_model == HW:
        model_name = f"{model_name}_{leakage_model.lower()}"

    check_if_exist = check_if_model_trained(model_name, loss_function, dataset_name)

    log_path = os.path.join(os.getcwd(), 'logs', f'{model_name}.log')
    create_dir_recursively(log_path, is_file_path=True)
    logger = setup_logging(log_path=log_path)
    setup_random_seed(seed=seed)
    logger.info('Model name {}'.format(model_name))
    config = vars(arguments)
    logger.info("Arguments {}".format(print_dictionary(config)))
    print(log_path)

    if loss_function == RANKING_LOSS:
        learner_params['loss_function'] = loss_dictionary_rkl_models[dataset_name]
    else:
        learner_params['loss_function'] = loss_dictionary_train_models[loss_function]

    learner_params['model_name'] = model_name
    learner_params['weight_averaging'] = weight_averaging

    start_time = time.time()
    if use_tuner:
        if loss_function in [FOCAL_LOSS_BE, FOCAL_LOSS_CE, RANKING_LOSS]:
            epochs_hpo = 20
            objective = kt.Objective("val_accuracy", direction="min")
            file_name = 'hypermodel_{}'.format(loss_function)
            directory = os.path.join(get_trained_models_path(TRAINED_HYPERMODEL), file_name)
            learner_params_copy = copy.deepcopy(learner_params)
            model_class_copy = copy.deepcopy(model_class)
            mod = model_class_copy(**learner_params_copy)
            x_train, y_train = mod.reshape_inputs(X_profiling_train, Y_profiling_train)
            x_val, y_val = mod.reshape_inputs(X_profiling_val, Y_profiling_val)
            max_trials = 30
        loss_params = None
        if loss_function in [FOCAL_LOSS_BE, FOCAL_LOSS_CE]:
            hpo_hypermodel_loss = HPOModelLoss(learner=model_class_copy, learner_params=learner_params_copy,
                                               hp_dict=focal_loss_dict, lf_name=loss_function)
            tuner = HPOModelTuner(hypermodel=hpo_hypermodel_loss, objective="val_accuracy", max_trials=max_trials,
                                  overwrite=False, directory=directory, project_name=model_name)
            tuner.search(x_train, y_train, epochs=epochs_hpo, batch_size=batch_size, validation_data=(x_val, y_val))
            tuner.results_summary(num_trials=max_trials)
            best_hp_loss = tuner.get_best_hyperparameters()[0]
            loss_params = dict(alpha=best_hp_loss.get('alpha'), gamma=best_hp_loss.get('gamma'), from_logits=False)

        elif loss_function == RANKING_LOSS:
            hpo_hypermodel_loss = HPOModelLoss(learner=model_class_copy, learner_params=learner_params_copy,
                                               hp_dict=rkl_loss_dict, lf_name=loss_function)
            tuner = HPOModelTuner(hypermodel=hpo_hypermodel_loss, objective="val_accuracy", max_trials=max_trials,
                                  overwrite=False, directory=directory, project_name=model_name)
            tuner.search(x_train, y_train, epochs=epochs_hpo, batch_size=batch_size, validation_data=(x_val, y_val))
            tuner.results_summary(num_trials=max_trials)
            best_hp_loss = tuner.get_best_hyperparameters()[0]
            loss_params = dict(alpha_value=best_hp_loss.get('alpha_value'))
        if loss_params is not None:
            with open('{}/{}.json'.format(get_trained_models_path(folder=HP_TRAINED_MODELS_TUNED), model_name),
                      'w') as f:
                json.dump(loss_params, f, sort_keys=True, indent=4)
            logger.info("Loss {}, best parameters {}".format(loss_function, print_dictionary(loss_params)))
            learner_params['loss_function'] = loss_dictionary_attack_models[loss_function](**loss_params)
    model = model_class(**learner_params)
    model.fit(X=X_profiling, y=Y_profiling, batch_size=batch_size, epochs=epochs, verbose=1)
    end_time = time.time()
    time_taken = timedelta(seconds=(end_time - start_time))
    logger.info('The total time elapsed for model {} is {}'.format(model_name, time_taken))
    if check_if_exist:
        logger.info(f"Model {model_name} already Trained")
