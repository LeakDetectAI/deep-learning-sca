import logging
import os
from abc import ABCMeta

import numpy as np

from deepscapy.dataset_reader.dataset_reader import DatasetReader


class AESRDDatasetReader(DatasetReader, metaclass=ABCMeta):
    def __init__(self, load_ciphertext=False, load_key=False, **kwargs):
        super(AESRDDatasetReader, self).__init__(dataset_folder='AES_RD', **kwargs)
        self.logger = logging.getLogger(AESRDDatasetReader.__name__)
        self.load_ciphertext = load_ciphertext
        self.load_key = load_key

        self.aes_rd_profiling_traces_file = os.path.join(self.dirname, 'profiling_traces_AES_RD.npy')
        self.aes_rd_profiling_labels_file = os.path.join(self.dirname, 'profiling_labels_AES_RD.npy')
        self.aes_rd_attack_traces_file = os.path.join(self.dirname, 'attack_traces_AES_RD.npy')
        self.aes_rd_attack_labels_file = os.path.join(self.dirname, 'attack_labels_AES_RD.npy')

        if self.load_ciphertext:
            self.aes_rd_profiling_plaintext_file = os.path.join(self.dirname, 'profiling_plaintext_AES_RD.npy')
            self.aes_rd_attack_plaintext_file = os.path.join(self.dirname, 'attack_plaintext_AES_RD.npy')
        else:
            self.aes_rd_profiling_plaintext_file = ''
            self.aes_rd_attack_plaintext_file = ''

        if self.load_key:
            self.key_file = os.path.join(self.dirname, 'key.npy')
        else:
            self.key_file = ''

        self.logger.info("Dataset Folder Path {}".format(self.dirname))
        self.__load_dataset__()

    def __load_dataset__(self):
        self.X_profiling = np.load(self.aes_rd_profiling_traces_file)
        self.Y_profiling = np.load(self.aes_rd_profiling_labels_file)
        self.X_attack = np.load(self.aes_rd_attack_traces_file)
        self.Y_attack = np.load(self.aes_rd_attack_labels_file)
        self.Y_profiling_hw = self.calculate_HW(self.Y_profiling)
        self.Y_attack_hw = self.calculate_HW(self.Y_attack)

        if self.load_ciphertext:
            self.aes_rd_profiling_plaintext = np.load(self.aes_rd_profiling_plaintext_file)
            self.aes_rd_attack_plaintext = np.load(self.aes_rd_attack_plaintext_file)
        else:
            self.aes_rd_profiling_plaintext = None
            self.aes_rd_attack_plaintext = None

        if self.load_key:
            self.key = np.load(self.key_file)
        else:
            self.key = None

    def get_plaintext_ciphertext(self):
        return (self.aes_rd_profiling_plaintext, self.aes_rd_attack_plaintext)

    def get_key(self):
        return self.key
