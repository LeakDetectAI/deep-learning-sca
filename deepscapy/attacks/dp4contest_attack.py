import logging

import numpy as np
from sklearn.utils import shuffle

from deepscapy.constants import *
from deepscapy.core.attack_class import AttackModel


# Attack Code from Methodology for Efficient CNN Architectures in Profiling Attacks Paper
# Github Repository (https://github.com/gabzai/Methodology-for-efficient-CNN-architectures-in-SCA)
class DP4ContestAttack(AttackModel):
    def __init__(self, model_name, model_class, loss_name, num_attacks, total_attack_traces, mask, offset, real_key, byte,
                 plaintext_ciphertext=None, use_tuner=False, reshape_type=TWOD_CNN_SQR, extension='h5',
                 num_classes=256, n_folds=10, seed=None, shuffle=True, **kwargs):
        # Logger
        self.logger = logging.getLogger(DP4ContestAttack.__name__)
        super().__init__(AES_SBOX=DP4CONTEST_BOX, model_name=model_name, model_class=model_class,
                         loss_name=loss_name, num_attacks=num_attacks, total_attack_traces=total_attack_traces,
                         real_key=real_key, byte=byte,plaintext_ciphertext=plaintext_ciphertext,
                         use_tuner=use_tuner, reshape_type=reshape_type, extension=extension,
                         num_classes=num_classes, n_folds=n_folds, seed=seed, shuffle=shuffle, **kwargs)

        self.mask = mask
        self.offset = offset
        self.dataset = DP4_CONTEST


    def attack(self, X_attack, Y_attack, model_evaluate_args=None, model_predict_args=None):
        super().attack(X_attack=X_attack, Y_attack=Y_attack, model_evaluate_args=model_evaluate_args,
                       model_predict_args=model_predict_args)

    def _rank_compute(self, prediction, att_plt, key, mask, offset, byte):
        (nb_trs, nb_hyp) = prediction.shape
        if self.leakage_model == HW:
            nb_hyp = 256
        key_log_prob = np.zeros(nb_hyp)
        rank_evol = np.full(nb_trs, 255)
        prediction = np.log(prediction + 1e-40)

        for i in range(nb_trs):
            for k in range(nb_hyp):
                idx = (self.AES_SBOX[int(att_plt[i, byte]) ^ int(k)] ^ int(mask[int(offset[i, byte] + 1) % 16]))
                if self.leakage_model == HW:
                    idx = self.hw_box[idx]
                key_log_prob[k] += prediction[i, idx]
            rank_evol[i] = self._rk_key(key_log_prob, key[byte])

        return rank_evol

    def _perform_attacks_(self, predictions, plain_cipher_fold, offset_fold):
        nb_traces = self.num_traces
        nb_attacks = self.num_attacks
        key = self.real_key
        byte = self.byte
        mask = self.mask
        all_rk_evol = np.zeros((nb_attacks, nb_traces))

        for i in range(nb_attacks):
            if self.shuffle:
                predictions_shuffled, plt_shuffled, offset_shuffled = shuffle(predictions, plain_cipher_fold,
                                                                              offset_fold,
                                                                              random_state=self.random_state)
            else:
                predictions_shuffled, plt_shuffled, offset_shuffled = predictions, plain_cipher_fold, offset_fold

            all_rk_evol[i] = self._rank_compute(predictions_shuffled, plt_shuffled, key, mask, offset_shuffled,
                                                byte=byte)

        all_rk_evol = self.trim_outlier_ranks(all_rk_evol, num=50)
        rk_avg = np.mean(all_rk_evol, axis=0)
        self.logger.info(f"All Ranks \n {all_rk_evol}")
        return rk_avg

    def _plot_model_attack_results(self, model_results_dir_path):
        super()._plot_model_attack_results(model_results_dir_path=model_results_dir_path)

    def _store_results(self, dataset=DP4_CONTEST):
        super()._store_results(dataset=dataset)

    def _load_attack_model_(self, dataset=DP4_CONTEST):
        return super()._load_attack_model_()
