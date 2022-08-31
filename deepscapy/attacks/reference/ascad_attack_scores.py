import logging
import os

import h5py
import matplotlib.pyplot as plot
import numpy as np
from sklearn.utils import check_random_state, shuffle

from deepscapy.constants import *
from deepscapy.utils import get_results_path, create_directory_safely


# Attack Code from Ranking Loss Paper Github Repository (https://github.com/gabzai/Ranking-Loss-SCA)
class ASCADAttackScores:
    def __init__(self, model_name, model_scores, num_attacks, total_attack_traces, plt, real_key, dataset_type,
                 byte, save_fig=True, store_results=True, num_classes=256, n_folds=10, seed=None, shuffle=True):
        self.ASCAD_AES_Sbox = ASCAD_AES_Sbox
        self.model_name = model_name
        self.num_attacks = num_attacks
        self.total_attack_traces = total_attack_traces
        self.plt = plt
        self.real_key = real_key
        self.dataset_type = dataset_type
        self.byte = byte
        self.save_fig = save_fig
        self.store_results = store_results
        self.num_classes = num_classes
        self.n_folds = n_folds
        self.shuffle = shuffle
        self.random_state = check_random_state(seed=seed)
        self.num_traces = int(self.total_attack_traces / self.n_folds)

        # Logger
        self.logger = logging.getLogger(ASCADAttackScores.__name__)

        # Initialize result variables
        self.model_mean_ranks = np.zeros((self.n_folds, self.num_traces))
        self.model_mean_rank_final = np.zeros(self.n_folds)
        self.model_accuracy = np.zeros(self.n_folds)
        self.model_guess_entropy = [np.array([-1], dtype=np.int64) for _ in range(self.n_folds)]
        #self.model_scores = np.zeros((self.n_folds, self.num_traces, self.num_classes))
        self.model_scores = model_scores.reshape(self.n_folds, self.num_traces, self.num_classes)

        self.model_min_complete_mean_rank = np.zeros(self.n_folds)
        self.model_last_index_mean_rank = np.zeros(self.n_folds)
        self.model_min_last_100_mean_rank = np.zeros(self.n_folds)

        self.ascad_attack()

    def ascad_attack(self):
        for fold_id in range(self.n_folds):
            plt_fold = self.plt[fold_id * self.num_traces:(fold_id + 1) * self.num_traces]
            avg_rank = self._perform_attacks_ascad(self.num_traces, self.model_scores[fold_id], self.num_attacks,
                                                   plt_fold, self.real_key, byte=self.byte)

            self.model_mean_ranks[fold_id] = avg_rank
            self.model_mean_rank_final[fold_id] = np.mean(self.model_mean_ranks[fold_id])
            self.model_min_complete_mean_rank[fold_id] = np.amin(self.model_mean_ranks[fold_id])
            self.model_last_index_mean_rank[fold_id] = self.model_mean_ranks[fold_id][-1]
            self.model_min_last_100_mean_rank[fold_id] = np.amin(self.model_mean_ranks[fold_id][self.num_traces - 100:])

            # self.logger.info('Fold ' + str(fold_id) + ': final mean rank for model ' + self.model_name + ' = ' + str(self.model_mean_rank_final[fold_id]))
            self.logger.info('Fold {}: final mean rank for model {} = {}'.format(fold_id, self.model_name,
                                                                                 self.model_mean_rank_final[fold_id]))
            self.logger.info('Fold {}: min complete mean rank for model {} = {}'.format(fold_id, self.model_name,
                                                                                        self.model_min_complete_mean_rank[
                                                                                            fold_id]))
            self.logger.info('Fold {}: last index mean rank for model {} = {}'.format(fold_id, self.model_name,
                                                                                      self.model_last_index_mean_rank[
                                                                                          fold_id]))
            self.logger.info('Fold {}: min last 100 mean rank for model {} = {}'.format(fold_id, self.model_name,
                                                                                        self.model_min_last_100_mean_rank[
                                                                                            fold_id]))

            results_guess_entropy = np.where(self.model_mean_ranks[fold_id] == 0)[0]
            if results_guess_entropy.size != 0:
                self.model_guess_entropy[fold_id] = results_guess_entropy

        if self.store_results:
            self._store_results()

    def _rk_key(self, rank_array, key):
        key_val = rank_array[key]
        return np.where(np.sort(rank_array)[::-1] == key_val)[0][0]

    def _rank_compute(self, prediction, att_plt, key, byte):
        (nb_trs, nb_hyp) = prediction.shape

        key_log_prob = np.zeros(nb_hyp)
        rank_evol = np.full(nb_trs, 255)
        prediction = np.log(prediction + 1e-40)

        for i in range(nb_trs):
            for k in range(nb_hyp):
                # print(k, self.ASCAD_AES_Sbox[k ^ att_plt[i, byte]])
                idx = self.ASCAD_AES_Sbox[k ^ att_plt[i, byte]]
                if self.leakage_model == HW:
                    idx = self.hw_box[idx]
                key_log_prob[k] += prediction[i, idx]
            rank_evol[i] = self._rk_key(key_log_prob, key[byte])

        return rank_evol

    def _perform_attacks_ascad(self, nb_traces, predictions, nb_attacks, plt, key, byte=2):
        all_rk_evol = np.zeros((nb_attacks, nb_traces))

        for i in range(nb_attacks):
            if self.shuffle:
                predictions_shuffled, plt_shuffled = shuffle(predictions, plt, random_state=self.random_state)
            else:
                predictions_shuffled, plt_shuffled = predictions, plt

            all_rk_evol[i] = self._rank_compute(predictions_shuffled, plt_shuffled, key, byte=byte)

        rk_avg = np.mean(all_rk_evol, axis=0)

        return rk_avg

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

    def _store_results(self):
        # Store the final evaluated results to .npy files & .svg file
        results_path = get_results_path()
        for dir in [ASCAD, self.dataset_type, self.model_name]:
            results_path = os.path.join(results_path, dir)
            create_directory_safely(results_path)
            self.logger.info("Creating Directory {}".format(results_path))

        result_file_path = os.path.join(results_path, 'final_results.h5')
        self.logger.info("Creating results at path {}: ".format(result_file_path))
        with h5py.File(result_file_path, 'w') as hdf:
            for fold_id in range(self.n_folds):
                fold_id_group = hdf.create_group('fold_id_{}'.format(fold_id))
                fold_id_group.create_dataset(MEAN_RANKS, data=self.model_mean_ranks[fold_id])
                fold_id_group.create_dataset(GUESS_ENTROPY, data=self.model_guess_entropy[fold_id])
                fold_id_group.create_dataset(SCORES, data=self.model_scores[fold_id])
                fold_id_group.create_dataset(MEAN_RANK_FINAL, data=self.model_mean_rank_final[fold_id])
                fold_id_group.create_dataset(ACCURACY, data=self.model_accuracy[fold_id])
                fold_id_group.create_dataset(MIN_COMPLETE_MEAN_RANK, data=self.model_min_complete_mean_rank[fold_id])
                fold_id_group.create_dataset(LAST_INDEX_MEAN_RANK, data=self.model_last_index_mean_rank[fold_id])
                fold_id_group.create_dataset(MIN_LAST_100_MEAN_RANK, data=self.model_min_last_100_mean_rank[fold_id])

        if self.save_fig:
            self._plot_model_attack_results(results_path)
