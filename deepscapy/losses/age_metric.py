import numpy as np
import random
import scipy.stats as ss
import sys
import tensorflow as tf
from keras import backend as K
from keras.metrics import Metric
from scipy import stats


class Utility:
    def __init__(self):
        self.AES_Sbox = np.array([
            0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
            0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
            0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
            0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
            0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
            0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
            0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
            0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
            0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
            0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
            0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
            0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
            0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
            0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
            0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
            0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16
        ])

    def hw(self):
        return np.array([bin(x).count("1") for x in range(256)])

    def msb(self):
        data = np.zeros(256)
        data[128:] = 1
        return data

    def labelize(self, plaintexts, keys):
        return self.AES_Sbox[plaintexts ^ keys]

    def bit_diff(self, a, b):
        return [self.hw()[int(a[i]) ^ int(b[i])] for i in range(len(a))]

    def calculate_HW(self, data):
        if isinstance(data, int):
            print('Input must be an array')
            sys.exit(-1)
        if data.ndim == 1:
            return self.hw()[data]
        else:
            return np.reshape([self.hw()[data.ravel()]], np.shape(data))

    def calculate_MSB(self, data):
        if isinstance(data, int):
            print('Input must be an array')
            sys.exit(-1)
        if data.ndim == 1:
            return self.msb()[data]
        else:
            return np.reshape([self.msb()[data.ravel()]], np.shape(data))

    def rk_key_all_traces(self, rank_array):
        container = np.empty(rank_array.shape, dtype=int)
        for k, row in enumerate(rank_array):
            container[k] = ss.rankdata(-row, method='dense') - 1
        return container


# Compute key distribution
class Key_Distribution():
    def __init__(self, leakage_model):
        self.leakage_model = leakage_model
        self.p = np.array([range(256)])
        self.k_all = np.array([range(256)])
        self.container = np.zeros((len(self.k_all), len(self.p)), int)
        # initialize the containers for the later calculation
        if self.leakage_model == 'HW':
            self.container = self.utility.calculate_HW(self.utility.labelize(self.p, self.k_all.T))
        elif self.leakage_model == 'ID':
            self.container = self.utility.labelize(self.p, self.k_all.T)
        else:
            self.container = self.utility.calculate_MSB(self.utility.labelize(self.p, self.k_all.T))

    def KD(self, correct_key):
        return np.array([np.sum(abs(np.power(self.container[correct_key] - self.container[k], 2))) for k in range(256)])

    def ranked_KD(self, correct_key):
        return ss.rankdata(self.KD(correct_key))

    def normalized_KD(self, correct_key):
        KD = -self.KD(correct_key)
        return (KD - np.min(KD)) / np.ptp(KD)

    def prob_LD(self, correct_key):
        KD = -self.KD(correct_key) + max(self.KD(correct_key))
        return KD / np.sum(KD)


# Compute the evolution of the key rank
class Attack:
    def __init__(self, KD, leakage_model, correct_key, nb_traces_attacks=2000, nb_attacks=5, attack_byte=2,
                 shuffle=True, output='rank'):
        self.leakage_model = leakage_model
        self.correct_key = correct_key
        self.nb_traces_attacks = nb_traces_attacks
        self.nb_attacks = nb_attacks
        self.attack_byte = attack_byte
        self.KD_allKey = np.zeros((256, 256))
        for k in range(256):
            self.KD_allKey[k] = -KD.KD(k)

        if self.leakage_model == 'HW':
            self.classes = 9
        else:
            self.classes = 256
        self.shuffle = shuffle
        self.output = output
        self.utility = Utility()

    def calculate_key_prob(self, y_true, y_pred):
        plt_attack = y_true[:, self.classes + 1:]
        return self.perform_attacks(y_pred, plt_attack)

    def perform_attacks(self, predictions, plt_attack):
        # for metric calculation
        if self.nb_traces_attacks == 'default':
            self.nb_traces_attacks = len(predictions)
        all_rk_evol = np.zeros((self.nb_attacks, self.nb_traces_attacks, 256))
        all_corr_evol = np.zeros((self.nb_attacks, self.nb_traces_attacks, 256))
        all_key_log_prob = np.zeros((self.nb_attacks, 256))

        for i in range(self.nb_attacks):
            if self.shuffle:
                l = list(zip(predictions, plt_attack))
                random.shuffle(l)
                sp, splt = list(zip(*l))
                sp = np.array(sp)
                splt = np.array(splt)
                att_pred = sp[:self.nb_traces_attacks]
                att_plt = splt[:self.nb_traces_attacks]
            else:
                att_pred = predictions[:self.nb_traces_attacks]
                att_plt = plt_attack[:self.nb_traces_attacks]

            results = self.rank_compute(att_pred, att_plt)

            if self.output == 'rank_corr':
                (all_rk_evol[i], all_corr_evol[i]) = results
            elif self.output == 'rank':
                all_rk_evol[i] = results
            else:
                all_key_log_prob[i] = results

        if self.output == 'rank_corr':
            return (np.mean(all_rk_evol, axis=0), np.mean(all_corr_evol, axis=0))
        elif self.output == 'rank':
            return (np.mean(all_rk_evol, axis=0))
        elif self.output == 'rank_metric':
            return (np.float32(np.mean(all_key_log_prob, axis=0)))
        elif self.output == 'prob_metric':
            return (np.float32(all_key_log_prob))

    def rank_compute(self, prediction, att_plt):
        (nb_traces, nb_hyp) = prediction.shape

        key_log_prob_accu = np.zeros(256)
        key_log_prob_evol = np.zeros((nb_traces, 256))

        prediction = np.log(np.where(prediction <= K.epsilon(), K.epsilon(), prediction))

        for i in range(nb_traces):
            if self.leakage_model == 'ID':
                predicted_output = prediction[
                    i, self.utility.AES_Sbox[np.bitwise_xor(range(256), int(att_plt[i, self.attack_byte]))]]
            else:
                predicted_output = prediction[i, self.utility.calculate_HW(
                    self.utility.AES_Sbox[np.bitwise_xor(range(256), int(att_plt[i, self.attack_byte]))])]
            key_log_prob_accu += predicted_output
            key_log_prob_evol[i] = key_log_prob_accu

        if self.output == 'rank_corr':
            rank_evol = self.utility.rk_key_all_traces(key_log_prob_evol)
            corr_prob_array = stats.spearmanr(key_log_prob_evol, self.KD_allKey, axis=1)[0][:len(key_log_prob_evol),
                              len(key_log_prob_evol):]
            corr_evol = self.utility.rk_key_all_traces(corr_prob_array)
            return (rank_evol, corr_evol)
        elif self.output == 'rank':
            rank_evol = self.utility.rk_key_all_traces(key_log_prob_evol)
            return rank_evol
        elif self.output == 'rank_metric':
            return 256 - ss.rankdata(key_log_prob_accu)
        elif self.output == 'prob_metric':
            return (key_log_prob_accu)


# Customized AGE metric
class AGE_Metric(Metric):
    def __init__(self, KD, leakage_model, correct_key, ATK, name='age', **kwargs):
        super(AGE_Metric, self).__init__(name=name, **kwargs)
        self.ATK = ATK
        self.correct_key = correct_key
        self.acc_sum = self.add_weight(name='acc_sum', shape=(self.ATK.nb_attacks, 256), initializer='zeros')
        self.KD_allKey = np.zeros((256, 256))
        for k in range(256):
            self.KD_allKey[k] = KD.KD(k)

        if leakage_model == 'HW':
            self.classes = 9
        else:
            self.classes = 256

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.acc_sum.assign_add(self.tf_calculate_key_prob(y_true, y_pred))

    def result(self):
        return tf.numpy_function(self.calculate_AGE, [self.acc_sum], tf.float32)

    def reset_states(self):
        self.acc_sum.assign(K.zeros((self.ATK.nb_attacks, 256)))

    @tf.function
    def tf_calculate_key_prob(self, y_true, y_pred):
        _ret = tf.numpy_function(self.calculate_key_prob, [y_true, y_pred], tf.float32)
        return _ret

    def calculate_key_prob(self, y_true, y_pred):
        plt_attack = y_true[:, self.classes:]
        if plt_attack[0][0] == 1:  # check if data is from validation set, then compute GE
            key_prob_multi_attacks = self.ATK.perform_attacks(y_pred, plt_attack[:, 1:])
        else:  # otherwise, return zeros
            key_prob_multi_attacks = np.float32(np.zeros((self.ATK.nb_attacks, 256)))
        return key_prob_multi_attacks

    def calculate_AGE(self, key_prob):
        # calculate AGE with the most likely key
        KD_bestKey = self.KD_allKey[np.argmax(key_prob)]

        # calculate AGE with the correct key
        KD_bestKey = self.KD_allKey[self.correct_key]

        avg_key_rank = np.sum(key_prob, axis=0)
        try:
            corr, _ = stats.spearmanr(avg_key_rank, -KD_bestKey)
        except ValueError:
            print('Value error when calculating age')
            return np.float32(0)
        if not np.isfinite(float(corr)):
            return np.float32(0)
        else:
            return np.float32(corr)


if __name__ == "__main__":
    leakage = "HW"
    correct_key = 224  # correct key of the attack traces, you can set it to the most likely key as well (see L277 - L281)
    nb_traces_attacks_metric = 5000  # number of attack traces used for metric calculation
    nb_attacks_metric = 10  # number of attacks for GE calculation
    attack_byte = 2

    # Init the key distribution matrix
    KD = Key_Distribution(leakage)
    # Init the attack setting
    Atk_ge = Attack(KD, leakage, correct_key, nb_traces_attacks=nb_traces_attacks_metric, nb_attacks=nb_attacks_metric,
                    attack_byte=attack_byte, shuffle=True, output='prob_metric')

    # Init the AGE metric, this metric can be directly used during the compiling of the model
    AGE = AGE_Metric(KD, leakage, correct_key, Atk_ge)
    # Y_profiling = np.concatenate(
    #    (to_categorical(Y_profiling, num_classes=classes), np.zeros((len(plt_profiling), 1)), plt_profiling), axis=1)
    # Y_attack = np.concatenate((to_categorical(Y_attack, num_classes=classes), np.ones((len(plt_attack), 1)), plt_attack), axis=1)
