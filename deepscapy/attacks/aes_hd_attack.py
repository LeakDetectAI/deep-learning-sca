import logging

import numpy as np
from sklearn.utils import shuffle

from deepscapy.constants import AES_HD, AES_Sbox_inv, TWOD_CNN_SQR, HW
from deepscapy.core.attack_class import AttackModel


# Attack Code from Ranking Loss Paper Github Repository (https://github.com/gabzai/Ranking-Loss-SCA)
class AESHDAttack(AttackModel):
    def __init__(self, model_name, model_class, loss_name, num_attacks, total_attack_traces, real_key, byte,
                 plaintext_ciphertext=None, use_tuner=False, reshape_type=TWOD_CNN_SQR, extension='h5',
                 num_classes=256, n_folds=10, seed=None, shuffle=True, **kwargs):
        # Logger
        self.logger = logging.getLogger(AESHDAttack.__name__)
        super().__init__(AES_SBOX=AES_Sbox_inv, model_name=model_name, model_class=model_class, loss_name=loss_name,
                         num_attacks=num_attacks, total_attack_traces=total_attack_traces,
                         plaintext_ciphertext=plaintext_ciphertext, real_key=real_key, byte=byte, use_tuner=use_tuner,
                         reshape_type=reshape_type, extension=extension, num_classes=num_classes, n_folds=n_folds,
                         seed=seed, shuffle=shuffle, **kwargs)
        self.offset = np.zeros_like((self.plaintext_ciphertext))

    def attack(self, X_attack, Y_attack, model_evaluate_args=None, model_predict_args=None):
        super().attack(X_attack=X_attack, Y_attack=Y_attack, model_evaluate_args=model_evaluate_args,
                       model_predict_args=model_predict_args)

    def _rank_compute(self, score, att_ciph, key, byte):
        (nb_trs, nb_hyp) = score.shape

        key_log_prob = np.zeros(nb_hyp)
        rank_evol = np.full(nb_trs, 255)
        score = np.log(score + 1e-40)

        for i in range(nb_trs):
            for k in range(nb_hyp):
                idx = self.AES_SBOX[k ^ int(att_ciph[i, 11])] ^ int(att_ciph[i, 7])
                if self.leakage_model == HW:
                    idx = self.hw_box[idx]
                key_log_prob[k] += score[i, idx]

            rank_evol[i] = self._rk_key(key_log_prob, key[byte])

        return rank_evol

    def _perform_attacks_(self, predictions, plain_cipher_fold, offset_fold):
        nb_traces = self.num_traces
        nb_attacks = self.num_attacks
        key = self.real_key
        byte = self.byte
        all_rk_evol = np.zeros((nb_attacks, nb_traces))

        for i in range(nb_attacks):
            if self.shuffle:
                score_shuffled, ciph_shuffled = shuffle(predictions, plain_cipher_fold, random_state=self.random_state)
            else:
                score_shuffled, ciph_shuffled = predictions, plain_cipher_fold

            all_rk_evol[i] = self._rank_compute(score_shuffled, ciph_shuffled, key, byte=byte)

        all_rk_evol = self.trim_outlier_ranks(all_rk_evol, num=50)
        rk_avg = np.mean(all_rk_evol, axis=0)
        self.logger.info("Rank Average {}".format(rk_avg))
        self.logger.info("All Ranks {}".format(all_rk_evol))
        return rk_avg

    def _plot_model_attack_results(self, model_results_dir_path):
        super()._plot_model_attack_results(model_results_dir_path=model_results_dir_path)

    def _store_results(self, dataset=AES_HD):
        super()._store_results(dataset=dataset)

    def _load_attack_model_(self, dataset=AES_HD):
        return super()._load_attack_model_()
