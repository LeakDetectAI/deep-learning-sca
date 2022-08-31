import logging
import os
from abc import ABCMeta

import numpy as np

from deepscapy.dataset_reader.dataset_reader import DatasetReader


class DP4ContestDatasetReader(DatasetReader, metaclass=ABCMeta):
    def __init__(self, load_ciphertext=False, load_key=False, **kwargs):
        super(DP4ContestDatasetReader, self).__init__(dataset_folder='DP4CONTEST', **kwargs)
        self.logger = logging.getLogger(DP4ContestDatasetReader.__name__)
        self.load_ciphertext = load_ciphertext
        self.load_key = load_key

        self.dpav4_profiling_traces_file = os.path.join(self.dirname, 'profiling_traces_dpav4.npy')
        self.dpav4_profiling_labels_file = os.path.join(self.dirname, 'profiling_labels_dpav4.npy')
        self.dpav4_attack_traces_file = os.path.join(self.dirname, 'attack_traces_dpav4.npy')
        self.dpav4_attack_labels_file = os.path.join(self.dirname, 'attack_labels_dpav4.npy')

        if self.load_ciphertext:
            self.dpav4_profiling_plaintext_file = os.path.join(self.dirname, 'profiling_plaintext_dpav4.npy')
            self.dpav4_attack_plaintext_file = os.path.join(self.dirname, 'attack_plaintext_dpav4.npy')
        else:
            self.dpav4_profiling_plaintext_file = ''
            self.dpav4_attack_plaintext_file = ''

        if self.load_key:
            self.key_file = os.path.join(self.dirname, 'key.npy')
            self.mask_file = os.path.join(self.dirname, 'mask.npy')
            self.offset_file = os.path.join(self.dirname, 'attack_offset_dpav4.npy')
        else:
            self.key_file = ''
            self.mask_file = ''
            self.offset_file = ''

        self.logger.info("Dataset Folder Path {}".format(self.dirname))
        self.__load_dataset__()

    def __load_dataset__(self):
        self.X_profiling = np.load(self.dpav4_profiling_traces_file)
        self.Y_profiling = np.load(self.dpav4_profiling_labels_file)
        self.X_attack = np.load(self.dpav4_attack_traces_file)
        self.Y_attack = np.load(self.dpav4_attack_labels_file)
        self.Y_profiling_hw = self.calculate_HW(self.Y_profiling)
        self.Y_attack_hw = self.calculate_HW(self.Y_attack)

        if self.load_ciphertext:
            self.dpav4_profiling_plaintext = np.load(self.dpav4_profiling_plaintext_file)
            self.dpav4_attack_plaintext = np.load(self.dpav4_attack_plaintext_file)
        else:
            self.dpav4_profiling_plaintext = None
            self.dpav4_attack_plaintext = None

        if self.load_key:
            self.key = np.load(self.key_file)
            self.mask = np.load(self.mask_file)
            self.offset = np.load(self.offset_file)
        else:
            self.key = None
            self.mask = None
            self.offset = None

    def get_plaintext_ciphertext(self):
        return self.get_plaintext()

    def get_plaintext(self):
        return (self.dpav4_profiling_plaintext, self.dpav4_attack_plaintext)

    def get_ciphertext(self):
        return None, None

    def get_key(self):
        return self.key

    def get_meta_data(self):
        return self.mask, self.offset
