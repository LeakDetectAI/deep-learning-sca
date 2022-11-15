import logging
import os

import h5py
import matplotlib.pyplot as plot
import numpy as np
from sklearn.utils import check_random_state, shuffle
from keras.models import load_model

from deepscapy.constants import *
from deepscapy.experimental_utils import *
from deepscapy.models import *
from deepscapy.utils import get_results_path, load_best_hp_to_dict, get_trained_models_path, create_dir_recursively, \
    setup_logging, create_directory_safely


# Attack Code from Ranking Loss Paper Github Repository (https://github.com/gabzai/Ranking-Loss-SCA)
class ASCADAttack:
    def __init__(self, model_name, model_class, loss, num_attacks, total_attack_traces, plt, real_key, dataset_type,
                 byte, use_tuner, extension='tf', save_fig=True, store_results=True, num_classes=256, n_folds=10,
                 seed=None, shuffle=True, metrics=['accuracy']):
        self.ASCAD_AES_Sbox = ASCAD_AES_Sbox
        self.model_name = model_name
        self.model_class = model_class
        self.loss = loss
        self.num_attacks = num_attacks
        self.total_attack_traces = total_attack_traces
        self.plt = plt
        self.real_key = real_key
        self.dataset_type = dataset_type
        self.byte = byte
        self.use_tuner = use_tuner
        self.save_fig = save_fig
        self.store_results = store_results
        self.num_classes = num_classes
        self.n_folds = n_folds
        self.shuffle = shuffle
        self.metrics = metrics
        self.extension = extension
        self.random_state = check_random_state(seed=seed)
        self.num_traces = int(self.total_attack_traces / self.n_folds)
        self.logger = logging.getLogger(ASCADAttack.__name__)

        # Load model for attack
        self.model = self._load_attack_model()
        self.model.summary(print_fn=self.logger.info)

        # Initialize result variables
        self.model_mean_ranks = np.zeros((self.n_folds, self.num_traces))
        self.model_mean_rank_final = np.zeros(self.n_folds)
        self.model_accuracy = np.zeros(self.n_folds)
        self.model_guess_entropy = [np.array([-1], dtype=np.int64) for _ in range(self.n_folds)]
        self.model_scores = np.zeros((self.n_folds, self.num_traces, self.num_classes))

        self.model_min_complete_mean_rank = np.zeros(self.n_folds)
        self.model_last_index_mean_rank = np.zeros(self.n_folds)
        self.model_min_last_100_mean_rank = np.zeros(self.n_folds)

    def ascad_attack(self, X_attack, Y_attack, model_evaluate_args=None, model_predict_args=None):
        for fold_id in range(self.n_folds):
            X_attack_fold = X_attack[fold_id * self.num_traces:(fold_id + 1) * self.num_traces]
            Y_attack_fold = Y_attack[fold_id * self.num_traces:(fold_id + 1) * self.num_traces]
            plt_fold = self.plt[fold_id * self.num_traces:(fold_id + 1) * self.num_traces]

            input_dim = X_attack.shape[1]

            if self.model_class == NASBasic or self.model_class == NASLeNet5:
                custom_model = self.model_class(model_name=self.model_name, num_classes=self.num_classes,
                                                input_dim=input_dim, dataset=self.dataset_type)
            else:
                custom_model = self.model_class(model_name=self.model_name, num_classes=self.num_classes,
                                                input_dim=input_dim)

            X_attack_fold, Y_attack_fold = custom_model.reshape_inputs(X_attack_fold, Y_attack_fold)

            if model_evaluate_args is not None:
                model_metrics = self.model.evaluate(X_attack_fold, Y_attack_fold, **model_evaluate_args)
            else:
                model_metrics = self.model.evaluate(X_attack_fold, Y_attack_fold)

            self.model_accuracy[fold_id] = model_metrics[1]

            # self.logger.info('Fold ' + str(fold_id) + ': accuracy for model ' + self.model_name + ' = ' + str(self.model_accuracy[fold_id]))
            self.logger.info(
                'Fold {}: accuracy for model {} = {}'.format(fold_id, self.model_name, self.model_accuracy[fold_id]))

            if model_predict_args is not None:
                self.model_scores[fold_id] = self.model.predict(X_attack_fold, **model_predict_args)
            else:
                self.model_scores[fold_id] = self.model.predict(X_attack_fold)

            avg_rank = self._perform_attacks_ascad(self.num_traces, self.model_scores[fold_id], self.num_attacks,
                                                   plt_fold,
                                                   self.real_key, byte=self.byte)

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
                key_log_prob[k] += prediction[i, self.ASCAD_AES_Sbox[k ^ att_plt[i, byte]]]

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
        dataset_root_dir_path = os.path.join(results_path, ASCAD)
        dataset_dir_path = os.path.join(results_path, ASCAD, self.dataset_type)
        model_results_dir_path = os.path.join(results_path, ASCAD, self.dataset_type, self.model_name)
        result_file_path = os.path.join(model_results_dir_path, '{}.h5'.format(self.model_name))

        create_directory_safely(dataset_root_dir_path)
        create_directory_safely(dataset_dir_path)
        create_directory_safely(model_results_dir_path)

        with h5py.File(result_file_path, 'w') as hdf:
            for fold_id in range(self.n_folds):
                fold_id_group = hdf.create_group('fold_id_{}'.format(fold_id))
                fold_id_group.create_dataset(MEAN_RANKS, data=self.model_mean_ranks[fold_id])
                fold_id_group.create_dataset(QTE_NUM_TRACES, data=self.model_guess_entropy[fold_id])
                fold_id_group.create_dataset(SCORES, data=self.model_scores[fold_id])
                fold_id_group.create_dataset(MEAN_RANK_FINAL, data=self.model_mean_rank_final[fold_id])
                fold_id_group.create_dataset(ACCURACY, data=self.model_accuracy[fold_id])
                fold_id_group.create_dataset(MIN_COMPLETE_MEAN_RANK, data=self.model_min_complete_mean_rank[fold_id])
                fold_id_group.create_dataset(LAST_INDEX_MEAN_RANK, data=self.model_last_index_mean_rank[fold_id])
                fold_id_group.create_dataset(MIN_LAST_100_MEAN_RANK, data=self.model_min_last_100_mean_rank[fold_id])

        if self.save_fig:
            self._plot_model_attack_results(model_results_dir_path)

    def _load_attack_model(self):
        attack_model = None
        custom_objects = {}

        if self.use_tuner:
            best_hp_dict = load_best_hp_to_dict(model_name=self.model_name)

            if self.loss == CATEGORICAL_CROSSENTROPY_LOSS:
                attack_model = load_model(os.path.join(get_trained_models_path(folder=TRAINED_MODELS_TUNED),
                                                       '{}.{}'.format(self.model_name, self.extension)))
            elif self.loss == CROSS_ENTROPY_RATIO:
                custom_objects = {'loss': loss_dictionary_train_models[self.loss]}

            elif self.loss == RANKING_LOSS:
                custom_objects = {
                    'loss': loss_dictionary_attack_models[self.loss](alpha_value=best_hp_dict['alpha_value'])}
            elif self.loss == DICE_BCE_LOSS:
                custom_objects = {'loss': loss_dictionary_train_models[self.loss]}
            elif self.loss == FOCAL_LOSS_BE:
                custom_objects = {'loss': loss_dictionary_attack_models[self.loss](alpha=best_hp_dict['alpha'],
                                                                                   gamma=best_hp_dict['gamma'],
                                                                                   from_logits=best_hp_dict[
                                                                                       'from_logits'])}
            if self.loss in [CROSS_ENTROPY_RATIO, RANKING_LOSS, FOCAL_LOSS_BE, DICE_BCE_LOSS]:
                attack_model = load_model(os.path.join(get_trained_models_path(folder=TRAINED_MODELS_TUNED),
                                                       '{}.{}'.format(self.model_name, self.extension)),
                                          custom_objects=custom_objects)

        else:
            if self.model_class == NASBasic or self.model_class == NASLeNet5 or \
                    self.model_class == NASBasic2 or self.model_class == NASBasic3:
                trained_models_path = get_trained_models_path(folder=TRAINED_MODELS_NAS)

                attack_model = load_model(
                    os.path.join(trained_models_path, '{}.{}'.format(self.model_name, self.extension)), compile=False)

                if self.loss == RANKING_LOSS:
                    loss_function = loss_dictionary_rkl_models[self.dataset_type]
                else:
                    loss_function = loss_dictionary_train_models[self.loss]

                attack_model.compile(loss=loss_function, metrics=self.metrics)
            else:
                trained_models_path = get_trained_models_path()

                if self.loss == CATEGORICAL_CROSSENTROPY_LOSS:
                    attack_model = load_model(
                        os.path.join(trained_models_path, '{}.{}'.format(self.model_name, self.extension)))
                elif self.loss == RANKING_LOSS:
                    custom_objects = {'loss': loss_dictionary_rkl_models[self.dataset_type]}
                    attack_model = load_model(
                        os.path.join(trained_models_path, '{}.{}'.format(self.model_name, self.extension)),
                        custom_objects=custom_objects)
                else:
                    custom_objects = {'loss': loss_dictionary_train_models[self.loss]}
                    attack_model = load_model(
                        os.path.join(trained_models_path, '{}.{}'.format(self.model_name, self.extension)),
                        custom_objects=custom_objects)
        return attack_model
