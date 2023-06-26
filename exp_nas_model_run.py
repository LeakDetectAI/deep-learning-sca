import argparse
import os
import time
from datetime import timedelta

import numpy as np
from keras.saving.save import load_model

from deepscapy.constants import *
from deepscapy.experimental_utils import *
from deepscapy.utils import *


def _load_attack_model(dataset, model_file, model_name, loss_function, logger):
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
        logger.info(traceback.format_exc())
        logger.info(str(e))
        attack_model = None
    return attack_model


if __name__ == "__main__":
    # Make results deterministic
    seed = 1234
    os.environ['PYTHONHASHSEED'] = str(seed)
    # os.environ["WANDB_API_KEY"] = "e1cd10bac622be84198c705e89f0209dd0fc0ac2"
    # os.environ["WANDB_MODE"] = "online"
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
    start_time = time.time()
    # objective = Objective("val_mean_rank", direction="min")
    condition = False
    model_file = os.path.join(get_trained_models_path(folder=TRAINED_MODELS_NAS_NEW), f'{model_name}.tf')
    attack_model = _load_attack_model(dataset_name, model_file, model_name, loss_function, logger)
    job_id = str(os.environ["SLURM_JOB_ID"])
    logger.info(f"Slurm job id {job_id}")
    print(f"Slurm job id {job_id}")
    if attack_model is not None:
        loss_str = str(attack_model.loss)
        logger.info("Model loss_function at {}".format(loss_str))
        cnd = "CategoricalCrossentropy" in loss_str
        logger.info(f"Loss function {loss_function} Loss str {loss_str}, condition {cnd}")
        if loss_function != CATEGORICAL_CROSSENTROPY_LOSS:
            if cnd:
                condition = True
        else:
            condition = True
    if condition:
        logger.info(f'The model is already trained with correct loss function')
        logger.info('Best model summary:')
        attack_model.summary(print_fn=logger.info)
    else:
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
