import argparse
import math
import os
import pickle

import h5py
import numpy as np
import pandas as pd

from deepscapy.constants import *
from deepscapy.experimental_utils import LF_EXTENSION
from deepscapy.utils import get_results_path, get_absolute_path, setup_random_seed, create_dir_recursively, \
    setup_logging, create_directory_safely, get_h5py_tree

if __name__ == "__main__":
    # Make results deterministic
    seed = 1234
    os.environ['PYTHONHASHSEED'] = str(seed)
    setup_random_seed(seed=seed)

    parser = argparse.ArgumentParser(description='Extract Attack Results')
    parser.add_argument('--dataset', type=str, required=True,
                        help='An ASCAD dataset for the nas_attack. Possible values are ASCAD_desync0_variable, ASCAD_desync50_variable, ASCAD_desync100.')
    parser.add_argument('--n_folds', type=int, default=10,
                        help='Number of nas_attack dataset folds to evaluate the model. Default value is 10.')
    parser.add_argument('--leakage_model', default='ID', type=str,
                        help=' The type of leakage one wants to exploit to attack the system. ID, HW. Default value is ID.')
    arguments = parser.parse_args()
    print(arguments)
    n_folds = arguments.n_folds
    dataset = arguments.dataset
    leakage_model = arguments.leakage_model
    log_path = os.path.join(os.getcwd(), 'logs', f'extract_results_{dataset.lower()}_{leakage_model.lower()}.log')
    create_dir_recursively(log_path, is_file_path=True)
    logger = setup_logging(log_path=log_path)
    logger.info(f"Arguments {arguments}")

    results_path = get_results_path(folder=f"{RESULTS}_{leakage_model.lower()}")
    absolute_path = get_absolute_path()
    attack_results_path = os.path.join(absolute_path, f'excel_results_{leakage_model.lower()}')
    create_directory_safely(attack_results_path)

    if dataset in ASCAD_DATASETS:
        dataset_dir_path = os.path.join(results_path, ASCAD, dataset)
    elif dataset == AES_HD:
        dataset_dir_path = os.path.join(results_path, AES_HD)
    elif dataset == AES_RD:
        dataset_dir_path = os.path.join(results_path, AES_RD)
    elif dataset == DP4_CONTEST:
        dataset_dir_path = os.path.join(results_path, DP4_CONTEST)
    elif dataset == CHES_CTF:
        dataset_dir_path = os.path.join(results_path, CHES_CTF)

    mean_rank_final = np.zeros(n_folds)
    min_complete_mean_rank = np.zeros(n_folds)
    last_index_mean_rank = np.zeros(n_folds)
    min_last_100_mean_rank = np.zeros(n_folds)
    accuracy = np.zeros(n_folds)
    number_of_traces = np.zeros(n_folds)

    trainable_params = 0
    non_trainable_params = 0
    total_params = 0
    n_conv_layers = 5
    n_dense_layers = 3

    RECTANGLE = 'rect'
    SQUARE = 'sqr'
    ONEDCNN = 'one_d'

    REHSAPE_TYPES = [RECTANGLE, SQUARE, ONEDCNN]
    model_name_index = []
    BASEMODEL_NAMES = NAS_MODELS + BASELINES
    for base_name in BASELINES:
        model_name_index.append(base_name)
    for model_type in NAS_MODELS:
        for reshape_type in REHSAPE_TYPES:
            for tuner_type in TUNER_TYPES:
                model_name_index.append('{}_{}_{}'.format(model_type, reshape_type, tuner_type))
    SE = 'se'
    loss_name_list = [value for key, value in LF_EXTENSION.items()]
    loss_name_list_ext = loss_name_list + [value + SE for value in loss_name_list]

    mean_rank_final_df = pd.DataFrame(columns=loss_name_list_ext, index=model_name_index)
    min_complete_df = pd.DataFrame(columns=loss_name_list_ext, index=model_name_index)
    last_index_mean_rank_df = pd.DataFrame(columns=loss_name_list_ext, index=model_name_index)
    min_last_100_df = pd.DataFrame(columns=loss_name_list_ext, index=model_name_index)
    number_of_traces_df = pd.DataFrame(columns=loss_name_list_ext, index=model_name_index)
    accuracy_df = pd.DataFrame(columns=loss_name_list_ext, index=model_name_index)

    trainable_params_df = pd.DataFrame(columns=loss_name_list, index=model_name_index)
    non_trainable_params_df = pd.DataFrame(columns=loss_name_list, index=model_name_index)
    total_params_df = pd.DataFrame(columns=loss_name_list, index=model_name_index)
    n_conv_layers_df = pd.DataFrame(columns=loss_name_list, index=model_name_index)
    n_dense_layers_df = pd.DataFrame(columns=loss_name_list, index=model_name_index)
    number_of_successful_attacks = pd.DataFrame(columns=loss_name_list, index=model_name_index)

    logger.info("Dataset Dir path".format(dataset_dir_path))
    model_results_dict = {}
    model_qte_results = {}
    for subdir, dirs, files in os.walk(dataset_dir_path):
        for file in files:
            if file.endswith('.h5'):
                file_path = os.path.join(subdir, file)
                logger.info('#########################################################################################')
                logger.info('subdir: {}'.format(subdir))
                model_folder_name = os.path.basename(subdir)
                logger.info('Actual Model: {}'.format(model_folder_name))
                logger.info('File Name: {}'.format(file))
                file_name = file.replace('.h5', '')
                with h5py.File(file_path, 'r') as hdf:
                    dd = get_h5py_tree(hdf)
                    logger.info(f"Keys {list(dd.keys())}")
                    for fold_id in range(n_folds):
                        mean_rank_final[fold_id] = np.array(hdf['fold_id_{}/{}'.format(fold_id, MEAN_RANK_FINAL)])
                        accuracy[fold_id] = np.array(hdf['fold_id_{}/{}'.format(fold_id, ACCURACY)])
                        min_complete_mean_rank[fold_id] = np.array(
                            hdf['fold_id_{}/{}'.format(fold_id, MIN_COMPLETE_MEAN_RANK)])
                        last_index_mean_rank[fold_id] = np.array(
                            hdf['fold_id_{}/{}'.format(fold_id, LAST_INDEX_MEAN_RANK)])
                        min_last_100_mean_rank[fold_id] = np.array(
                            hdf['fold_id_{}/{}'.format(fold_id, MIN_LAST_100_MEAN_RANK)])
                        try:
                            number_of_traces[fold_id] = np.array(
                                hdf['fold_id_{}/{}'.format(fold_id, QTE_NUM_TRACES)])
                        except KeyError as e:
                            number_of_traces[fold_id] = np.array(
                                hdf['fold_id_{}/{}'.format(fold_id, 'guess_entropy')])
                        logger.info('Fold ID: {}, Mean Rank Final: {}'.format(fold_id, mean_rank_final[fold_id]))
                        logger.info('Fold ID: {}, Accuracy: {}'.format(fold_id, accuracy[fold_id]))
                        logger.info(
                            'Fold ID: {}, Min Complete Mean Rank: {}'.format(fold_id, min_complete_mean_rank[fold_id]))
                        logger.info(
                            'Fold ID: {}, Last Index Mean Rank: {}'.format(fold_id, last_index_mean_rank[fold_id]))
                        logger.info(
                            'Fold ID: {}, Min Last 100 Mean Rank: {}'.format(fold_id, min_last_100_mean_rank[fold_id]))
                    model_name = ''
                    for base_name in BASEMODEL_NAMES:
                        if base_name in model_folder_name:
                            model_name = model_name + base_name
                            if 'weight_averaging' in model_folder_name:
                                model_name = '{}_{}'.format(model_name, 'weight_averaging')
                                logger.info(model_name)
                            if 'nas' in model_folder_name:
                                for reshape_type in REHSAPE_TYPES:
                                    if reshape_type in model_folder_name:
                                        model_name = '{}_{}'.format(model_name, reshape_type)
                                for tuner_type in TUNER_TYPES:
                                    if tuner_type in model_folder_name:
                                        model_name = '{}_{}'.format(model_name, tuner_type)

                    for loss_name in loss_name_list:
                        if loss_name in model_folder_name:
                            loss = loss_name
                    logger.info("Model name {}, loss {}".format(model_name, loss))
                    if not ('_tuned' in model_folder_name or 'weight_averaging' in model_folder_name or 'lenet' in model_folder_name):
                        lossse = loss + SE
                        try:
                            trainable_params = np.array(hdf[f'model_parameters/{TRAINABLE_PARAMS}'])
                            non_trainable_params = np.array(hdf[f'model_parameters/{NON_TRAINABLE_PARAMS}'])
                            total_params = np.array(hdf[f'model_parameters/{TOTAL_PARAMS}'])
                            n_conv_layers = np.array(hdf[f'model_parameters/{N_CONV_LAYERS}'])
                            n_dense_layers = np.array(hdf[f'model_parameters/{N_DENSE_LAYERS}'])
                        except Exception as e:
                            n_conv_layers = "NA"
                            n_dense_layers = "NA"
                            logger.error(f"Fields missing in a Model {model_name} loss {loss}")
                            logger.error(f"Error is {str(e)}")

                        logger.info(f'Trainable params = {trainable_params}')
                        logger.info(f'Non-trainable params = {non_trainable_params}')
                        logger.info(f'Total params = {total_params}')
                        logger.info(f'Convolutional layers = {n_conv_layers}')
                        logger.info(f'Dense layers = {n_dense_layers}')

                        mean_rank_final_df.at[model_name, loss] = np.around(np.mean(mean_rank_final), 2)
                        mean_rank_final_df.at[model_name, lossse] = np.around(np.std(mean_rank_final) / math.sqrt(10),
                                                                              2)

                        min_complete_df.at[model_name, loss] = np.around(np.mean(min_complete_mean_rank), 2)
                        min_complete_df.at[model_name, lossse] = np.around(
                            np.std(min_complete_mean_rank) / math.sqrt(10), 2)

                        last_index_mean_rank_df.at[model_name, loss] = np.around(np.mean(last_index_mean_rank), 2)
                        last_index_mean_rank_df.at[model_name, lossse] = np.around(
                            np.std(last_index_mean_rank) / math.sqrt(10), 2)

                        min_last_100_df.at[model_name, loss] = np.around(np.mean(min_last_100_mean_rank), 2)
                        min_last_100_df.at[model_name, lossse] = np.around(
                            np.std(min_last_100_mean_rank) / math.sqrt(10), 2)

                        accuracy_df.at[model_name, loss] = np.around(np.mean(accuracy), 6)
                        accuracy_df.at[model_name, lossse] = np.around(np.std(accuracy) / math.sqrt(10), 6)

                        number_of_traces_df.at[model_name, loss] = np.around(np.nanmean(number_of_traces), 6)
                        number_of_traces_df.at[model_name, lossse] = np.around(
                            np.nanstd(number_of_traces) / math.sqrt(10), 6)

                        number_of_successful_attacks.at[model_name, loss] = np.sum(min_complete_mean_rank < 1)

                        trainable_params_df.at[model_name, loss] = trainable_params
                        non_trainable_params_df.at[model_name, loss] = non_trainable_params
                        total_params_df.at[model_name, loss] = total_params
                        n_conv_layers_df.at[model_name, loss] = n_conv_layers
                        n_dense_layers_df.at[model_name, loss] = n_dense_layers
                        #model_results_dict[f"{model_name}_{loss}"] = min_complete_mean_rank
                        #model_qte_results[f"{model_name}_{loss}"] = number_of_traces

    number_of_successful_attacks.to_excel(os.path.join(attack_results_path, f'n_successful_attacks_{dataset}_attack.xlsx'),
                                          index=True, header=True, index_label='Model Name')

    mean_rank_final_df.to_excel(os.path.join(attack_results_path, f'mean_rank_final_{dataset}_attack.xlsx'),
                                index=True, header=True, index_label='Model Name')
    min_complete_df.to_excel(os.path.join(attack_results_path, f'min_complete_mean_rank_{dataset}_attack.xlsx'),
                             index=True, header=True, index_label='Model Name')

    last_index_mean_rank_df.to_excel(os.path.join(attack_results_path, f'last_index_mean_rank_{dataset}_attack.xlsx'),
                                     index=True, header=True, index_label='Model Name')
    min_last_100_df.to_excel(os.path.join(attack_results_path, f'min_last_100_mean_rank_{dataset}_attack.xlsx'),
                             index=True, header=True, index_label='Model Name')
    accuracy_df.to_excel(os.path.join(attack_results_path, f'accuracy_{dataset}_attack.xlsx'), index=True,
                         header=True, index_label='Model Name')

    number_of_traces_df.to_excel(os.path.join(attack_results_path, 'number_of_traces_{}_attack.xlsx'.format(dataset)),
                                 index=True, header=True, index_label='Model Name')

    n_conv_layers_df.to_excel(os.path.join(attack_results_path, f'number_of_conv_layers_{dataset}_attack.xlsx'),
                              index=True, header=True, index_label='Model Name')
    n_dense_layers_df.to_excel(os.path.join(attack_results_path, f'number_of_dense_layers_{dataset}_attack.xlsx'),
                               index=True, header=True, index_label='Model Name')
    trainable_params_df.to_excel(os.path.join(attack_results_path, f'trainable_params_{dataset}_attack.xlsx'),
                                 index=True, header=True, index_label='Model Name')
    non_trainable_params_df.to_excel(os.path.join(attack_results_path, f'non_trainable_params_{dataset}_attack.xlsx'),
                                     index=True, header=True, index_label='Model Name')
    total_params_df.to_excel(os.path.join(attack_results_path, f'total_params_{dataset}_attack.xlsx'), index=True,
                             header=True, index_label='Model Name')
    logger.info("Finish")