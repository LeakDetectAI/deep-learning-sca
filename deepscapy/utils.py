import inspect
import json
import logging
import multiprocessing
import os
import os.path
import random
import re
import sys

import h5py
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as K
from keras.optimizers import Adam, SGD, RMSprop, Nadam, Adagrad
from keras.regularizers import l2
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from tensorflow.core.protobuf.config_pb2 import ConfigProto
from tensorflow.python.client.session import Session

from deepscapy.constants import *

__all__ = ['check_file_exists', 'load_ascad', 'get_duration_seconds', 'print_dictionary',
           'setup_random_seed', 'create_dir_recursively', 'get_ascad_dataset', 'setup_logging', 'mean_rank_metric',
           'standardize_features', 'add_best_hp_parameters', 'save_best_hp_to_json', 'load_best_hp_to_dict',
           'get_absolute_path', 'create_directory_safely', 'get_trained_models_path', 'get_results_path',
           'get_datasets_path', 'get_model_parameters_count', 'softmax']



# Function from ASCAD Paper Github Repository (https://github.com/ANSSI-FR/ASCAD)
def check_file_exists(file_path):
    file_path = os.path.normpath(file_path)
    if not os.path.exists(file_path):
        print("Error: provided file path '%s' does not exist!" % file_path)
        sys.exit(-1)
    return


# Function from ASCAD Paper Github Repository (https://github.com/ANSSI-FR/ASCAD)
def load_ascad(ascad_database_file, load_metadata=False):
    check_file_exists(ascad_database_file)
    # Open the ASCAD database HDF5 for reading
    try:
        in_file = h5py.File(ascad_database_file, "r")
    except:
        print("Error: can't open HDF5 file '%s' for reading (it might be malformed) ..." % ascad_database_file)
        sys.exit(-1)
    # Load profiling traces
    X_profiling = np.array(in_file['Profiling_traces/traces'], dtype=np.float32)
    # Load profiling labels
    Y_profiling = np.array(in_file['Profiling_traces/labels'])
    # Load attacking traces
    X_attack = np.array(in_file['Attack_traces/traces'], dtype=np.float32)
    # Load attacking labels
    Y_attack = np.array(in_file['Attack_traces/labels'])
    if load_metadata == False:
        return (X_profiling, Y_profiling), (X_attack, Y_attack)
    else:
        return (X_profiling, Y_profiling), (X_attack, Y_attack), (
            in_file['Profiling_traces/metadata'], in_file['Attack_traces/metadata'])


def get_duration_seconds(duration):
    time = int(re.findall(r"\d+", duration)[0])
    d = duration.split(str(time))[1].upper()
    options = {"D": 24 * 60 * 60, "H": 60 * 60, "M": 60}
    return options[d] * time


def print_dictionary(dictionary):
    output = "\n"
    for key, value in dictionary.items():
        output = output + str(key) + " => " + str(value).strip() + "\n"
    return output



def setup_random_seed(seed=1234):
    # logger.info('Seed value: {}'.format(seed))
    logger = logging.getLogger("Setup Logging")
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["KERAS_BACKEND"] = "tensorflow"
    devices = tf.config.experimental.list_physical_devices('GPU')
    logger.info("Devices {}".format(devices))
    n_gpus = len(devices)
    logger.info("Number of GPUS {}".format(n_gpus))
    if n_gpus == 0:
        config = ConfigProto(
            intra_op_parallelism_threads=1,
            inter_op_parallelism_threads=1,
            allow_soft_placement=True,
            log_device_placement=False,
            device_count={"CPU": multiprocessing.cpu_count() - 2},
        )
    else:
        config = ConfigProto(
            allow_soft_placement=True,
            log_device_placement=True,
            intra_op_parallelism_threads=2,
            inter_op_parallelism_threads=2,
        )
        config.gpu_options.allow_growth = True
    sess = Session(config=config)
    K.set_session(sess)

def create_dir_recursively(path, is_file_path=False):
    if is_file_path:
        path = os.path.dirname(path)
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def create_directory_safely(path, is_file_path=False):
    try:
        if is_file_path:
            path = os.path.dirname(path)
        if not os.path.exists(path):
            os.mkdir(path)
    except Exception as e:
        print(str(e))


def get_ascad_dataset(dataset_path_prefix, dataset):
    if dataset == 'ASCAD_desync0_variable':
        ascad_database = dataset_path_prefix + '/' + 'ASCAD/' + dataset + '/ASCAD_desync0_variable.h5'
    elif dataset == 'ASCAD_desync50_variable' or dataset == 'ASCAD_desync100':
        ascad_database = dataset_path_prefix + '/' + 'ASCAD/' + dataset + '/' + dataset + '.h5'
    else:
        ascad_database = ''

    return ascad_database


def setup_logging(log_path=None, level=logging.INFO):
    """Function setup as many logging for the experiments"""
    if log_path is None:
        dirname = os.path.dirname(
            os.path.abspath(inspect.getfile(inspect.currentframe()))
        )
        dirname = os.path.dirname(dirname)
        log_path = os.path.join(dirname, "experiments", "logs", "logs.log")
        create_dir_recursively(log_path, True)
    logging.basicConfig(
        filename=log_path,
        filemode='w',
        level=level,
        format="%(asctime)s %(name)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("SetupLogger")
    logger.info("log file path: {}".format(log_path))
    return logger


def standardize_features(x_train, x_test, standardize='minmax'):
    if standardize == 'minmax':
        standardize = MinMaxScaler()
    elif standardize == 'standard':
        standardize = StandardScaler()
    else:
        standardize = RobustScaler()
    x_train = standardize.fit_transform(x_train)
    x_test = standardize.transform(x_test)
    return x_train, x_test


def mean_rank_metric(y_true, y_pred):
    y_true = np.argmax(y_true, axis=1).astype(np.int32)
    scores_df = pd.DataFrame(data=y_pred)
    final_ranks = scores_df.rank(ascending=False, axis=1)
    final_ranks = final_ranks.to_numpy(dtype='int32')
    predicted_ranks = np.zeros(shape=(y_true.shape[0]))
    for itr in range(y_true.shape[0]):
        true_label = y_true[itr]
        predicted_ranks[itr] = final_ranks[itr, true_label]
    return np.mean(predicted_ranks)


def add_best_hp_parameters(learner_params, best_hp):
    reg_strength = best_hp.get('reg_strength')
    learning_rate = best_hp.get('learning_rate')
    kernel_initializer = best_hp.get('kernal_initializer')
    optimizer = best_hp.get('optimizer')

    # Initialize the optimizer with the learning_rate
    if optimizer.lower() == 'adam':
        optimizer = Adam(learning_rate=learning_rate)
    elif optimizer.lower() == 'sgd':
        optimizer = SGD(learning_rate=learning_rate, nesterov=True, momentum=0.9)
    elif optimizer.lower() == 'rmsprop':
        optimizer = RMSprop(learning_rate=learning_rate)
    elif optimizer.lower() == 'nadam':
        optimizer = Nadam(learning_rate=learning_rate)
    elif optimizer.lower() == 'adagrad':
        optimizer = Adagrad(learning_rate=learning_rate)
    else:
        optimizer = None

    learner_params['optimizer'] = optimizer
    learner_params['kernel_initializer'] = kernel_initializer
    learner_params['kernel_regularizer'] = l2(l=reg_strength)
    return learner_params


def save_best_hp_to_json(best_hp, model_name, loss_function, best_hp_loss):
    best_hp_dict = {}

    best_hp_dict['optimizer'] = best_hp.get('optimizer')
    best_hp_dict['learning_rate'] = best_hp.get('learning_rate')
    best_hp_dict['reg_strength'] = best_hp.get('reg_strength')
    best_hp_dict['kernal_initializer'] = best_hp.get('kernal_initializer')

    if loss_function == FOCAL_LOSS_BE:
        best_hp_dict['alpha'] = best_hp_loss.get('alpha')
        best_hp_dict['gamma'] = best_hp_loss.get('gamma')
        best_hp_dict['from_logits'] = best_hp_loss.get('from_logits')
    elif loss_function == RANKING_LOSS:
        best_hp_dict['alpha_value'] = best_hp_loss.get('alpha_value')


    with open('{}/{}.json'.format(get_trained_models_path(folder=HP_TRAINED_MODELS_TUNED), model_name), 'w') as f:
        json.dump(best_hp_dict, f, sort_keys=True, indent=4)


def load_best_hp_to_dict(model_name):
    with open('{}/{}.json'.format(get_trained_models_path(folder=HP_TRAINED_MODELS_TUNED), model_name), 'r') as f:
        best_hp_dict = json.load(f)

    return best_hp_dict


def get_absolute_path():
    if "pc2" in os.environ["HOME"]:
        absolute_path = os.path.join(os.environ["PFS_FOLDERA"], "deep-learning-sca")
    else:
        dirname = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        absolute_path = os.path.dirname(dirname)

    return absolute_path


# ASCAD, AES-HD, AES-SD, DP4-contest
def get_trained_models_path(folder=TRAINED_MODELS_NON_TUNED, subfolder='dataset'):
    absolute_path = get_absolute_path()
    models_path = os.path.join(absolute_path, TRAINED_MODELS)
    # trained_models_path = os.path.join(absolute_path, "models", subfolder, folder)
    trained_models_path = os.path.join(models_path, folder)
    create_directory_safely(models_path)
    create_directory_safely(trained_models_path)
    return trained_models_path


def get_results_path(folder=RESULTS):
    absolute_path = get_absolute_path()
    results_path = os.path.join(absolute_path, folder)
    create_directory_safely(results_path)
    return results_path


def get_datasets_path():
    absolute_path = get_absolute_path()
    datasets_path = os.path.join(absolute_path, "deepscapy", "datasets")
    return datasets_path


def get_model_parameters_count(model):
    stringlist = []
    trainable_params = 0
    non_trainable_params = 0
    total_params = 0

    model.summary(print_fn=lambda x: stringlist.append(x))
    conv_layers = []
    dense_layers = []
    for line in stringlist:
        if 'Total params:' in line:
            total_params = int(line.split('Total params: ')[1].replace(',', ''))
        if 'Trainable params: ' in line:
            trainable_params = int(line.split('Trainable params: ')[1].replace(',', ''))
        if 'Non-trainable params: ' in line:
            non_trainable_params = int(line.split('Non-trainable params: ')[1].replace(',', ''))
        if 'Conv' in line:
            conv_layers.append(line)
        if 'Dense' in line:
            dense_layers.append(line)
    n_conv_layers = len(conv_layers)
    n_dense_layers = len(dense_layers)
    return trainable_params, non_trainable_params, total_params, n_conv_layers, n_dense_layers


def get_h5py_tree(hdf):
    dd = {}

    def extract(name, node):
        if isinstance(node, h5py.Dataset):
            dd[name] = node[...]
        return None

    hdf.visititems(extract)
    return dd


def logsumexp(x, axis=1):
    max_x = x.max(axis=axis, keepdims=True)
    return max_x + np.log(np.sum(np.exp(x - max_x), axis=axis, keepdims=True))


def softmax(x, axis=1):
    """
    Take softmax for the given numpy array.
    :param axis: The axis around which the softmax is applied
    :param x: array-like, shape (n_samples, ...)
    :return: softmax taken around the given axis
    """
    lse = logsumexp(x, axis=axis)
    return np.exp(x - lse)
