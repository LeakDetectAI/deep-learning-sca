import os

import numpy as np
from keras.models import load_model

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

    log_path = os.path.join(os.getcwd(), 'logs', 'models_trained.log')
    create_dir_recursively(log_path, is_file_path=True)
    logger = setup_logging(log_path=log_path)
    setup_random_seed(seed=seed)
    models_not_finished = []
    models_with_other_loss_function = dict()
    for leakage_model in ['HW', 'ID']:
        logger.info("#############################################################################")
        logger.info(f"***************************Leakage Model {leakage_model}***************************")
        for dataset_name in datasets.keys():
            if dataset_name in [AES_HDV2_EXT, AES_HDV2_NORM]:
                continue
            dataset_reader_class = datasets[dataset_name]
            dataset_reader_obj = dataset_reader_class(dataset_type=dataset_name, load_key=True, load_metadata=True,
                                                      load_ciphertext=True)
            (X_profiling, Y_profiling), (X_attack, Y_attack) = dataset_reader_obj.get_train_test_dataset()
            input_dim = X_profiling.shape[1]
            num_classes = len(np.unique(Y_profiling))
            for args_model_name in NAS_MODELS:
                for loss_function in LF_EXTENSION.keys():
                    for tuner_type in TUNER_TYPES:
                        for reshape_type in RESHAPE_TYPES:
                            logger.info("#############################################################################")
                            logger.info(f"Dataset {dataset_name} Model Name {args_model_name} "
                                        f"Loss function {loss_function} Tuner Type {tuner_type}")
                            shape_type = INPUT_SHAPE_DICT[reshape_type]
                            model_name = '{}_{}_{}_{}_{}_{}'.format(dataset_name.lower(), args_model_name, input_dim,
                                                                    shape_type, LF_EXTENSION[loss_function], tuner_type)
                            if leakage_model == HW:
                                model_name = f"{model_name}_{leakage_model.lower()}"
                            model_file = os.path.join(get_trained_models_path(folder=TRAINED_MODELS_NAS_NEW),
                                                      '{}.tf'.format(model_name))
                            logger.info('Model name {}'.format(model_name))
                            logger.info("Model stored at {}".format(model_file))
                            attack_model = _load_attack_model(dataset_name, model_file, model_name, loss_function, logger)
                            if attack_model is None:
                                models_not_finished.append(model_name)
                                logger.info("Model Still not finished")
                            else:
                                loss_str = str(attack_model.loss)
                                logger.info("Model loss_function at {}".format(loss_str))
                                logger.info(f"Is loss CCE, condition {loss_function == CATEGORICAL_CROSSENTROPY_LOSS:}")
                                logger.info(f"Loss function {loss_str}, condition {'CategoricalCrossentropy' not in loss_str}")
                                if loss_function != CATEGORICAL_CROSSENTROPY_LOSS:
                                    if "CategoricalCrossentropy" not in loss_str:
                                        if dataset_name not in models_with_other_loss_function.keys():
                                            models_with_other_loss_function[dataset_name] = [model_name]
                                        else:
                                             models_with_other_loss_function[dataset_name].append(model_name)

            # for args_model_name in BASELINES:
            #     for loss_function in LF_EXTENSION.keys():
            #         logger.info("#############################################################################")
            #         logger.info(f"Dataset {dataset_name} Model Name {args_model_name} "
            #                     f"Loss function {loss_function}")
            #         model_name = '{}_{}_{}_{}'.format(dataset_name.lower(), args_model_name, input_dim,
            #                                           LF_EXTENSION[loss_function])
            #         if leakage_model == HW:
            #             model_name = f"{model_name}_{leakage_model.lower()}"
            #         model_file = os.path.join(get_trained_models_path(folder=TRAINED_MODELS_NON_TUNED),
            #                                   '{}.tf'.format(model_name))
            #         logger.info('Model name {}'.format(model_name))
            #         logger.info("Model stored at {}".format(model_file))
            #         attack_model = _load_attack_model(dataset_name, model_file, model_name, loss_function, logger)
            #         if attack_model is None:
            #             models_not_finished.append(model_name)
            #             logger.info("Model Still not finished")

    logger.info(f"Models which are not finished {models_not_finished}")
    for key, value  in models_with_other_loss_function.items():
        logger.info(f"Dataset {key}: Models Left {value}")
