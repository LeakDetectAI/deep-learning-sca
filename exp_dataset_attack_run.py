import argparse
import os
import time
from datetime import timedelta

import numpy as np

from deepscapy.attacks import *
from deepscapy.constants import *
from deepscapy.core import NASModel
from deepscapy.experimental_utils import *
from deepscapy.utils import *

attack_classes = {ASCAD_DESYNC0: ASCADAttack, ASCAD_DESYNC50: ASCADAttack, ASCAD_DESYNC100: ASCADAttack,
                  ASCAD_DESYNC0_VARIABLE: ASCADAttack, ASCAD_DESYNC50_VARIABLE: ASCADAttack,
                  ASCAD_DESYNC100_VARIABLE: ASCADAttack, AES_HD: AESHDAttack,
                  AES_RD: AESRDAttack, DP4_CONTEST: DP4ContestAttack, CHES_CTF: CHESCTFAttack}

if __name__ == "__main__":
    # Make results deterministic
    seed = 1234
    os.environ['PYTHONHASHSEED'] = str(seed)
    # os.environ["WANDB_API_KEY"] = "e1cd10bac622be84198c705e89f0209dd0fc0ac2"
    # os.environ["WANDB_MODE"] = "online"
    setup_random_seed(seed=seed)

    # Argument parser
    parser = argparse.ArgumentParser(description='ASCAD, AES_HD, AES_RD, DP4CONTEST Attack')
    parser.add_argument('--dataset', type=str, required=True,
                        help='An ASCAD dataset for the nas_attack. Possible values are ASCAD_desync0_variable, ASCAD_desync50_variable, ASCAD_desync100, AES_HD, AES_RD, DP4CONTEST.')
    parser.add_argument('--model_name', type=str, required=True,
                        help='Name of the model for nas_attack. Possible values are ascad_mlp_baseline, ascad_cnn_baseline, ascad_cnn_rkl_baseline, lenet5, nas_basic, nas_lenet5.')
    parser.add_argument('--loss_function', type=str, required=True,
                        help='Loss function for the models. Possible values are categorical_crossentropy, ranking_loss, cross_entropy_ratio, dice_bce_loss, sigmoid_focal_crossentropy, sigmoid_focal_categorical_crossentropy, smooth_topk_loss, hard_topk_loss, smooth_hard_topk_loss.')
    parser.add_argument('--use_tuner', default=False, action=argparse.BooleanOptionalAction,
                        help='Whether to use keras tuner or not before training the model. Default value is True.')
    parser.add_argument('--tuner_type', type=str, default='random_search',
                        help='Type of tuner to use for hyperparameter search. Possible values are hyperband, random_search, bayesian_optimization. Default value is random_search.')
    parser.add_argument('--reshape_type', default='2dCNNSqr', type=str,
                        help='Type of reshaping to use. Possible values are 2dCNNSqr, 2dCNNRect. Default value is 2dCNNSqr.')
    # parser.add_argument('--number_of_attack_traces', type=int, default=500,
    #                     help='Number of nas_attack traces to use for ASCAD nas_attack. Default value is 500.')
    parser.add_argument('--weight_averaging', default=False, action=argparse.BooleanOptionalAction,
                        help='Whether to use weight averaging or not for the tuner hypermodel or the model. Default value is True.')
    parser.add_argument('--number_of_attacks', type=int, default=100,
                        help='Number of nas_attack to perform. Default value is 100.')
    parser.add_argument('--byte', type=int, default=2,
                        help='Position of the byte to nas_attack. Default value is 2.')
    parser.add_argument('--n_folds', type=int, default=10,
                        help='Number of nas_attack dataset folds to evaluate the model. Default value is 10.')
    parser.add_argument('--leakage_model', default='ID', type=str,
                        help=' The type of leakage one wants to exploit to attack the system. ID, HW. Default value is ID.')

    arguments = parser.parse_args()
    print("Arguments {}".format(arguments))

    dataset_name = arguments.dataset
    model_name = arguments.model_name
    loss_function = arguments.loss_function
    n_folds = arguments.n_folds
    num_attacks = arguments.number_of_attacks
    byte = arguments.byte
    reshape_type = arguments.reshape_type
    tuner_type = arguments.tuner_type
    use_tuner = arguments.use_tuner
    weight_averaging = arguments.weight_averaging
    leakage_model = arguments.leakage_model

    # Load dataset
    dataset_reader_class = datasets[dataset_name]
    dataset_reader_obj = dataset_reader_class(dataset_type=dataset_name, load_key=True, load_metadata=True,
                                              load_ciphertext=True, leakage_model=leakage_model)
    (plaintext_ciphertext_profiling, plaintext_ciphertext_attack) = dataset_reader_obj.get_plaintext_ciphertext()

    (X_profiling, Y_profiling), (X_attack, Y_attack) = dataset_reader_obj.get_train_test_dataset()
    # real_key_path = dataset_reader_obj.key_file
    # Define the model parameters
    if model_name == CNN_ZAID_BASELINE:
        model_class = cnn_rkl_baseline_dictionary[dataset_name]
        # X_profiling, X_attack = standardize_features(X_profiling, X_attack)
    else:
        model_class = model_dictionary[model_name]
        # X_profiling, X_attack = standardize_features(X_profiling, X_attack, standardize='standard')

    if model_name == CNN_ZAID_BASELINE and (dataset_name == ASCAD_DESYNC0 or dataset_name == AES_HD):
        X_profiling, X_attack = standardize_features(X_profiling, X_attack)

    if model_name == CNN_ZAID_BASELINE and dataset_name == DP4_CONTEST:
        X_profiling, X_attack = standardize_features(X_profiling, X_attack, standardize='standard')
        X_profiling, X_attack = standardize_features(X_profiling, X_attack, standardize='minmax')

    n_dimensions = X_profiling.shape[1]
    if issubclass(model_class, NASModel):
        if reshape_type == TWOD_CNN_SQR:
            model_name = '{}_{}_{}_{}_{}_{}'.format(dataset_name.lower(), model_name, n_dimensions,
                                                    'sqr', LF_EXTENSION[loss_function], tuner_type)
        elif reshape_type == TWOD_CNN_RECT:
            model_name = '{}_{}_{}_{}_{}_{}'.format(dataset_name.lower(), model_name, n_dimensions,
                                                    'rect', LF_EXTENSION[loss_function], tuner_type)
        elif reshape_type == ONED_CNN:
            model_name = '{}_{}_{}_{}_{}_{}'.format(dataset_name.lower(), model_name, n_dimensions,
                                                    'one_d', LF_EXTENSION[loss_function], tuner_type)
    else:
        model_name = '{}_{}_{}_{}'.format(dataset_name.lower(), model_name, n_dimensions,
                                          LF_EXTENSION[loss_function])
        if weight_averaging:
            model_name = '{}_{}'.format(model_name, 'weight_averaging')
        if use_tuner:
            if loss_function in [FOCAL_LOSS_BE, FOCAL_LOSS_CE, RANKING_LOSS]:
                model_name = '{}_{}'.format(model_name, 'tuned')

    if leakage_model == HW:
        model_name = f"{model_name}_{leakage_model.lower()}"

    log_path = os.path.join(os.getcwd(), 'logs', f'{model_name}_attack.log')
    print(model_name, log_path)
    config = vars(arguments)
    # wandb.init(project='DLSCA', name=model_name, config=config)
    create_dir_recursively(log_path, is_file_path=True)
    logger = setup_logging(log_path=log_path)
    setup_random_seed(seed=seed)
    logger.info('Attack using model {}'.format(model_name))
    logger.info("Arguments {}".format(print_dictionary(config)))

    num_classes = len(np.unique(Y_profiling))
    # num_traces = args.number_of_attack_traces
    total_attack_traces = X_attack.shape[0]

    model_evaluate_args = {'verbose': 0}
    model_predict_args = {'verbose': 0}

    # Attack using the chosen model
    if dataset_name == DP4_CONTEST:
        real_key = dataset_reader_obj.get_key()
        mask, offset = dataset_reader_obj.get_meta_data()
    else:
        real_key = dataset_reader_obj.get_key()
        mask, offset = None, None

    start_time = time.time()
    attack_class_params = dict(model_name=model_name, model_class=model_class, loss_name=loss_function,
                               num_attacks=num_attacks, total_attack_traces=total_attack_traces,
                               dataset_type=dataset_name, plaintext_ciphertext=plaintext_ciphertext_attack, mask=mask,
                               offset=offset, real_key=real_key, byte=byte, use_tuner=use_tuner, n_folds=n_folds,
                               seed=seed, shuffle=True, extension='tf', reshape_type=reshape_type,
                               leakage_model=leakage_model)
    attack_params = dict(X_attack=X_attack, Y_attack=Y_attack, model_evaluate_args=model_evaluate_args,
                         model_predict_args=model_predict_args)
    attack_obj_class = attack_classes[dataset_name]
    attack_obj = attack_obj_class(**attack_class_params)
    try:
        attack_obj.attack(**attack_params)
    except Exception as e:
        import traceback

        logger.error(traceback.format_exc())
        logger.error(str(e))

    end_time = time.time()
    t_delta = timedelta(seconds=(end_time - start_time))
    logger.info(f'The total time elapsed for nas_attack using model {model_name} is {t_delta}')
