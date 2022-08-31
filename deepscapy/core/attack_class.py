import logging
import os
from abc import ABCMeta, abstractmethod

import h5py
import matplotlib.pyplot as plot
import numpy as np
from keras.saving.save import load_model
from sklearn.utils import check_random_state

from deepscapy.constants import *
from deepscapy.core import NASModel
from deepscapy.experimental_utils import loss_dictionary_train_models, loss_dictionary_attack_models, \
    loss_dictionary_rkl_models
from deepscapy.utils import get_results_path, create_directory_safely, load_best_hp_to_dict, get_trained_models_path, \
    get_model_parameters_count



class AttackModel(metaclass=ABCMeta):
    def __init__(self, AES_SBOX, model_name, model_class, loss_name, num_attacks, total_attack_traces,
                 dataset_type, real_key, byte, plaintext_ciphertext=None, use_tuner=False, reshape_type=TWOD_CNN_SQR,
                 leakage_model='ID', extension='tf', num_classes=256, n_folds=10, seed=None, shuffle=True, **kwargs):
        self.AES_SBOX = AES_SBOX
        self.model_name = model_name
        self.model_class = model_class
        self.loss_name = loss_name
        self.num_attacks = num_attacks
        self.total_attack_traces = total_attack_traces
        self.dataset_type = dataset_type
        self.real_key = real_key
        self.byte = byte
        self.plaintext_ciphertext = plaintext_ciphertext
        self.use_tuner = use_tuner
        self.reshape_type = reshape_type
        self.num_classes = num_classes
        self.n_folds = n_folds
        self.shuffle = shuffle
        self.extension = extension
        self.leakage_model = leakage_model
        self.random_state = check_random_state(seed=seed)
        self.num_traces = int(self.total_attack_traces / self.n_folds)
        self.logger = logging.getLogger(AttackModel.__name__)

        # Initialize result variables
        self.model_mean_ranks = np.zeros((self.n_folds, self.num_traces))
        self.model_mean_rank_final = np.zeros(self.n_folds)
        self.model_accuracy = np.zeros(self.n_folds)
        self.model_guess_entropy = np.zeros(self.n_folds)
        self.model_guess_entropy_3 = np.zeros(self.n_folds)
        self.model_guess_entropy_5 = np.zeros(self.n_folds)
        self.model_guess_entropy_10 = np.zeros(self.n_folds)
        self.model_scores = np.zeros((self.n_folds, self.num_traces, self.num_classes))
        self.trainable_params = 0
        self.non_trainable_params = 0
        self.total_params = 0
        self.model = None
        self.model_min_complete_mean_rank = np.zeros(self.n_folds)
        self.model_last_index_mean_rank = np.zeros(self.n_folds)
        self.model_min_last_100_mean_rank = np.zeros(self.n_folds)
        self.hw_box = [bin(x).count("1") for x in range(256)]

    def attack(self, X_attack, Y_attack, model_evaluate_args, model_predict_args):
        # Load model for attack
        self.model = self._load_attack_model_()
        self.model.summary(print_fn=self.logger.info)

        self.trainable_params, self.non_trainable_params, self.total_params = get_model_parameters_count(self.model)
        self.logger.info('Trainable params for model {} = {}'.format(self.model_name, self.trainable_params))
        self.logger.info('Non-trainable params for model {} = {}'.format(self.model_name, self.non_trainable_params))
        self.logger.info('Total params for model {} = {}'.format(self.model_name, self.total_params))

        for fold_id in range(self.n_folds):
            X_attack_fold = X_attack[fold_id * self.num_traces:(fold_id + 1) * self.num_traces]
            Y_attack_fold = Y_attack[fold_id * self.num_traces:(fold_id + 1) * self.num_traces]
            plain_cipher_fold = self.plaintext_ciphertext[fold_id * self.num_traces:(fold_id + 1) * self.num_traces]
            offset_fold = self.offset[fold_id * self.num_traces:(fold_id + 1) * self.num_traces]

            input_dim = X_attack.shape[1]

            if issubclass(self.model_class, NASModel):
                custom_model = self.model_class(model_name=self.model_name, num_classes=self.num_classes,
                                                input_dim=input_dim, dataset=DP4_CONTEST,
                                                reshape_type=self.reshape_type)
            else:
                custom_model = self.model_class(model_name=self.model_name, num_classes=self.num_classes,
                                                input_dim=input_dim)

            X_attack_fold, Y_attack_fold = custom_model.reshape_inputs(X_attack_fold, Y_attack_fold)

            if model_evaluate_args is not None:
                model_metrics = self.model.evaluate(X_attack_fold, Y_attack_fold, **model_evaluate_args)
            else:
                model_metrics = self.model.evaluate(X_attack_fold, Y_attack_fold)

            self.model_accuracy[fold_id] = model_metrics[1]

            self.logger.info('*****************************************************************************')
            self.logger.info('Performing attack using model {} for Fold {}'.format(self.model_name, fold_id))
            self.logger.info('Fold {}: accuracy for model {} = {}'.format(fold_id, self.model_name, model_metrics[1]))

            if model_predict_args is not None:
                self.model_scores[fold_id] = self.model.predict(X_attack_fold, **model_predict_args)
            else:
                self.model_scores[fold_id] = self.model.predict(X_attack_fold)
            predictions = self.model_scores[fold_id]

            avg_rank = self._perform_attacks_(predictions=predictions, plain_cipher_fold=plain_cipher_fold,
                                              offset_fold=offset_fold)
            self.model_mean_ranks[fold_id] = avg_rank
            self.model_mean_rank_final[fold_id] = np.mean(self.model_mean_ranks[fold_id])
            self.model_min_complete_mean_rank[fold_id] = np.amin(self.model_mean_ranks[fold_id])
            self.model_last_index_mean_rank[fold_id] = self.model_mean_ranks[fold_id][-1]
            self.model_min_last_100_mean_rank[fold_id] = np.amin(self.model_mean_ranks[fold_id][self.num_traces - 100:])

            rank = self.model_mean_rank_final[fold_id]
            self.logger.info('Fold {}: final mean rank for model {} = {}'.format(fold_id, self.model_name, rank))
            rank = self.model_min_complete_mean_rank[fold_id]
            self.logger.info('Fold {}: min complete mean rank for model {} = {}'.format(fold_id, self.model_name, rank))
            rank = self.model_last_index_mean_rank[fold_id]
            self.logger.info('Fold {}: last index mean rank for model {} = {}'.format(fold_id, self.model_name, rank))
            rank = self.model_min_last_100_mean_rank[fold_id]
            self.logger.info('Fold {}: min last 100 mean rank for model {} = {}'.format(fold_id, self.model_name, rank))
            def get_value(avg_rank, n):
                if np.argmax(avg_rank <= n) == 0:
                    return self.num_traces
                else:
                    return np.argmax(avg_rank <= n)
            self.model_guess_entropy[fold_id] = get_value(avg_rank, 0)
            self.model_guess_entropy_3[fold_id] = get_value(avg_rank, 2)
            self.model_guess_entropy_5[fold_id] = get_value(avg_rank, 4)
            self.model_guess_entropy_10[fold_id] = get_value(avg_rank, 9)
            self.logger.info('Fold {}: model {} GE smaller that 1: {}'.format(fold_id, self.model_name,
                                                                              self.model_guess_entropy[fold_id]))
            self.logger.info('Fold {}: model {} GE smaller that 3: {}'.format(fold_id, self.model_name,
                                                                              self.model_guess_entropy_3[fold_id]))
            self.logger.info('Fold {}: model {} GE smaller that 5: {}'.format(fold_id, self.model_name,
                                                                              self.model_guess_entropy_5[fold_id]))
            self.logger.info('Fold {}: model {} GE smaller that 10: {}'.format(fold_id, self.model_name,
                                                                               self.model_guess_entropy_10[fold_id]))

        self._store_results()

    def attack_from_scores(self, scores, model):
        _, _, total_params = get_model_parameters_count(model)
        model_min_complete_mean_rank = np.zeros(self.n_folds)
        for fold_id in range(self.n_folds):
            predictions = scores[fold_id * self.num_traces:(fold_id + 1) * self.num_traces]
            plain_cipher_fold = self.plaintext_ciphertext[fold_id * self.num_traces:(fold_id + 1) * self.num_traces]
            offset_fold = self.offset[fold_id * self.num_traces:(fold_id + 1) * self.num_traces]
            avg_rank = self._perform_attacks_(predictions=predictions, plain_cipher_fold=plain_cipher_fold,
                                              offset_fold=offset_fold)
            model_min_complete_mean_rank[fold_id] = rank = np.amin(avg_rank)
            self.logger.info('Fold {}: min complete mean rank for model = {}'.format(fold_id, rank))
        return model_min_complete_mean_rank, total_params

    def _rk_key(self, rank_array, key):
        #from scipy.stats import rankdata
        # key_val = rank_array[key]
        # idx = np.where(np.sort(rank_array)[::-1] == key_val)
        # if len(idx) != 0:
        #    rank = idx[0][0]
        # else:
        #    rank = 255
        # ranking = np.argsort(np.argsort(rank_array)[::-1])
        #ranking = len(rank_array) - rankdata(rank_array, method='max')
        #rank = ranking[key]
        #if np.any(np.isnan(rank_array)):
        #    rank = 125
        if np.any(np.isnan(rank_array)):
            rank = 125
        else:
            ranking = np.argsort(np.argsort(rank_array)[::-1])
            rank = ranking[key]
        return rank


    @abstractmethod
    def _perform_attacks_(self, predictions, plain_cipher_fold, offset_fold):
        pass


    def _plot_model_attack_results(self, model_results_dir_path):
        for fold_id in range(self.n_folds):
            plot.rcParams["figure.figsize"] = (15, 10)
            plot.ylim(-5, 200)
            plot.grid(True)
            plot.plot(self.model_mean_ranks[fold_id], '-')
            plot.xlabel('Number of Traces', size=30)
            plot.ylabel('Guessing Entropy', size=30)
            plot.xticks(fontsize=30)
            plot.yticks(fontsize=30)

            plot.savefig(os.path.join(model_results_dir_path, '{}_{}trs_{}att_{}fid.svg'.format(self.model_name,
                                                                                                self.num_traces,
                                                                                                self.num_attacks,
                                                                                                fold_id)), format='svg',
                         dpi=1200)

            plot.close()

    def trim_outlier_ranks(self, all_rk_evol, num=100):
        b = []
        for col in all_rk_evol.T:
            col = col[np.argpartition(col, num + 1)[:num]]
            b.append(col)
        return np.array(b).T

    def _store_results(self, dataset=AES_HD):
        # Store the final evaluated results to .npy files & .svg file
        results_path = get_results_path(folder=f"{RESULTS}_{self.leakage_model.lower()}")
        if dataset == ASCAD:
            for dir in [dataset, self.dataset_type, self.model_name]:
                results_path = os.path.join(results_path, dir)
                create_directory_safely(results_path)
                self.logger.info("Creating Directory {}: ".format(results_path))
        else:
            for dir in [dataset, self.model_name]:
                results_path = os.path.join(results_path, dir)
                create_directory_safely(results_path)
                self.logger.info("Creating Directory {}: ".format(results_path))
        result_file_path = os.path.join(results_path, 'final_results.h5')
        self.logger.info("Creating results at path {}: ".format(result_file_path))

        with h5py.File(result_file_path, 'w') as hdf:
            model_params_group = hdf.create_group('model_parameters')
            model_params_group.create_dataset(TRAINABLE_PARAMS, data=self.trainable_params)
            model_params_group.create_dataset(NON_TRAINABLE_PARAMS, data=self.non_trainable_params)
            model_params_group.create_dataset(TOTAL_PARAMS, data=self.total_params)

            for fold_id in range(self.n_folds):
                fold_id_group = hdf.create_group('fold_id_{}'.format(fold_id))
                fold_id_group.create_dataset(MEAN_RANKS, data=self.model_mean_ranks[fold_id])
                fold_id_group.create_dataset(GUESS_ENTROPY, data=self.model_guess_entropy[fold_id])
                fold_id_group.create_dataset(GUESS_ENTROPY + '3', data=self.model_guess_entropy_3[fold_id])
                fold_id_group.create_dataset(GUESS_ENTROPY + '5', data=self.model_guess_entropy_5[fold_id])
                fold_id_group.create_dataset(GUESS_ENTROPY + '10', data=self.model_guess_entropy_10[fold_id])
                fold_id_group.create_dataset(SCORES, data=self.model_scores[fold_id])
                fold_id_group.create_dataset(MEAN_RANK_FINAL, data=self.model_mean_rank_final[fold_id])
                fold_id_group.create_dataset(ACCURACY, data=self.model_accuracy[fold_id])
                fold_id_group.create_dataset(MIN_COMPLETE_MEAN_RANK, data=self.model_min_complete_mean_rank[fold_id])
                fold_id_group.create_dataset(LAST_INDEX_MEAN_RANK, data=self.model_last_index_mean_rank[fold_id])
                fold_id_group.create_dataset(MIN_LAST_100_MEAN_RANK, data=self.model_min_last_100_mean_rank[fold_id])

        self._plot_model_attack_results(results_path)


    def _load_attack_model_(self, dataset=AES_HD):
        attack_model = None
        custom_objects = {}

        if self.use_tuner and "_tuned" in self.model_name:
            best_hp_dict = load_best_hp_to_dict(model_name=self.model_name)
            model_file_name = os.path.join(get_trained_models_path(folder=TRAINED_MODELS_TUNED),
                                           '{}.{}'.format(self.model_name, self.extension))
            self.logger.info("Model stored at {}".format(model_file_name))
            if self.loss_name == RANKING_LOSS:
                params = dict(alpha_value=best_hp_dict['alpha_value'])
                custom_objects = {'loss': loss_dictionary_attack_models[self.loss_name](**params)}

            elif self.loss_name in [FOCAL_LOSS_BE, FOCAL_LOSS_CE]:
                params = dict(alpha=best_hp_dict['alpha'], gamma=best_hp_dict['gamma'],
                              from_logits=best_hp_dict['from_logits'])
                custom_objects = {'loss': loss_dictionary_attack_models[self.loss_name](**params)}
            if self.loss_name in [RANKING_LOSS, FOCAL_LOSS_BE, FOCAL_LOSS_CE]:
                attack_model = load_model(model_file_name, custom_objects=custom_objects)

        else:
            if issubclass(self.model_class, NASModel):
                trained_models_path = get_trained_models_path(folder=TRAINED_MODELS_NAS_NEW)
            else:
                trained_models_path = get_trained_models_path()
            model_file_name = os.path.join(trained_models_path, '{}.{}'.format(self.model_name, self.extension))
            self.logger.info("Model stored at {}".format(model_file_name))
            if self.loss_name == CATEGORICAL_CROSSENTROPY_LOSS:
                attack_model = load_model(model_file_name)
            elif self.loss_name == RANKING_LOSS:
                if dataset == ASCAD:
                    custom_objects = {'loss': loss_dictionary_rkl_models[self.dataset_type]}
                else:
                    custom_objects = {'loss': loss_dictionary_rkl_models[dataset]}
                attack_model = load_model(model_file_name, custom_objects=custom_objects)

            else:
                custom_objects = {'loss': loss_dictionary_train_models[self.loss_name]}
                attack_model = load_model(model_file_name, custom_objects=custom_objects)
        return attack_model
