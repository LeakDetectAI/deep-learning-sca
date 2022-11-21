import argparse
import os
import time
from datetime import timedelta

import numpy as np

from deepscapy.constants import *
from deepscapy.experimental_utils import *
from deepscapy.utils import *

if __name__ == "__main__":
    # Make results deterministic
    seed = 1234
    os.environ['PYTHONHASHSEED'] = str(seed)
    # Argument parser
    parser = argparse.ArgumentParser(description='Model HP Tuner & Model Training')
    parser.add_argument('--dataset', type=str, required=True,
                        help='An ASCAD dataset for the attack. Possible values are ASCAD_desync0, ASCAD_desync50, ASCAD_desync100, ASCAD_desync0_variable, ASCAD_desync50_variable, ASCAD_desync100_variable, AES_HD, AES_RD, DP4CONTEST.')
    parser.add_argument('--model_name', type=str, required=True,
                        help='Name of the model to use. Possible values are ascad_mlp_baseline, ascad_cnn_baseline, ascad_cnn_rkl_baseline, lenet5, nas_basic, nas_lenet5.')
    parser.add_argument('--loss_function', type=str, required=True,
                        help='Loss function for the models. Possible values are categorical_crossentropy, ranking_loss, cross_entropy_ratio, dice_bce_loss, sigmoid_focal_binary_crossentropy, sigmoid_focal_categorical_crossentropy')
    parser.add_argument('--epochs', default=200, type=int,
                        help='Epochs for the tuner hypermodel or the model. Default value is 150.')
    parser.add_argument('--batch_size', default=200, type=int,
                        help='Batch Size for the tuner hypermodel or the model. Default value is 200.')
    parser.add_argument('--tuner_type', default='hyperband', type=str,
                        help='Type of tuner to use for hyperparameter search. Possible values are hyperband, random_search, bayesian. Default value is random_search.')
    parser.add_argument('--max_trials', default=50, type=int,
                        help='Number of trials to use for random or bayesian tuner. Default value is 50.')
    parser.add_argument('--reshape_type', default='2dCNNSqr', type=str,
                        help='Type of reshaping to use. Possible values are 2dCNNSqr, 2dCNNRect, 1dCNN. Default value is 2dCNNSqr.')
    parser.add_argument('--leakage_model', default='ID', type=str,
                        help=' The type of leakage one wants to exploit to attack the system. ID, HW. Default value is ID.')

    arguments = parser.parse_args()
    print("Arguments {}".format(arguments))
    dataset_name = arguments.dataset
    args_model_name = arguments.model_name
    epochs = arguments.epochs
    batch_size = arguments.batch_size
    loss_function = arguments.loss_function
    tuner_type = arguments.tuner_type
    max_trials = arguments.max_trials
    reshape_type = arguments.reshape_type
    leakage_model = arguments.leakage_model

    # Load dataset
    dataset_reader_class = datasets[dataset_name]
    dataset_reader_obj = dataset_reader_class(dataset_type=dataset_name, load_key=True, load_metadata=True,
                                              load_ciphertext=True, leakage_model=leakage_model)
    (plaintext_ciphertext_profiling, plaintext_ciphertext_attack) = dataset_reader_obj.get_plaintext_ciphertext()

    (X_profiling, Y_profiling), (X_attack, Y_attack) = dataset_reader_obj.get_train_test_dataset()


    input_dim = X_profiling.shape[1]
    num_classes = len(np.unique(Y_profiling))
    # metrics = ['accuracy', MeanRank()]
    metrics = ['accuracy']
    loss = loss_dictionary_train_models[loss_function]
    #loss = loss_dictionary_train_models[FOCAL_LOSS_CER]
    if reshape_type == TWOD_CNN_SQR:
        model_name = '{}_{}_{}_{}_{}_{}'.format(dataset_name.lower(), args_model_name, input_dim,
                                                'sqr', LF_EXTENSION[loss_function], tuner_type)
    elif reshape_type == TWOD_CNN_RECT:
        model_name = '{}_{}_{}_{}_{}_{}'.format(dataset_name.lower(), args_model_name, input_dim,
                                                'rect', LF_EXTENSION[loss_function], tuner_type)
    elif reshape_type == ONED_CNN:
        model_name = '{}_{}_{}_{}_{}_{}'.format(dataset_name.lower(), args_model_name, input_dim,
                                                'one_d', LF_EXTENSION[loss_function], tuner_type)

    if leakage_model == HW:
        model_name = f"{model_name}_{leakage_model.lower()}"

    log_path = os.path.join(os.getcwd(), 'logs', f'{model_name}.log')
    create_dir_recursively(log_path, is_file_path=True)
    logger = setup_logging(log_path=log_path)

    # objective = Objective("val_mean_rank", direction="min")
    objective = 'val_accuracy'

    learner_params = {'num_classes': num_classes, 'metrics': metrics, 'loss_function': loss,
                      'loss_function_name': loss_function, 'input_dim': input_dim,
                      'seed': seed, 'max_trials': max_trials, 'tuner': tuner_type,
                      'dataset': dataset_name, 'model_name': model_name, 'objective': objective,
                      'reshape_type': reshape_type, 'leakage_model': leakage_model}

    model_class = model_dictionary[args_model_name]
    model = model_class(**learner_params)
    setup_random_seed(seed=seed)
    logger.info(f'Model name {model_name}')
    config = vars(arguments)
    logger.info(f"Arguments {print_dictionary(config)}")
    logger.info(f"Log File {log_path}")

    start_time = time.time()
    verbose = 1
    if tuner_type in [RANDOM_TUNER, BAYESIAN_TUNER]:
        n_e = 20
        model.fit(X=X_profiling, y=Y_profiling, batch_size=batch_size, epochs=n_e, final_model_epochs=epochs - n_e,
                  verbose=verbose)
    elif tuner_type == GREEDY_TUNER:
        n_e = 50
        model.fit(X=X_profiling, y=Y_profiling, batch_size=batch_size, epochs=n_e, final_model_epochs=epochs - n_e,
                  verbose=verbose)
    else:
        model.fit(X=X_profiling, y=Y_profiling, batch_size=batch_size, final_model_epochs=epochs, epochs=epochs,
                  verbose=verbose)


    logger.info('Best model summary:')
    model.summary(print_fn=logger.info)
    logger.info('Search Space summary:')
    model.search_space_summary()

    # model.evaluate(X_profiling, Y_profiling)
    # model.predict(X_profiling)
    # model.summary(print_fn=logger.info)
    end_time = time.time()
    time_taken = timedelta(seconds=(end_time - start_time))
    logger.info(f'The total time elapsed for model {model_name} is {time_taken}')
