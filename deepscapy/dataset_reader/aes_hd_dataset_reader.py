import logging
import os
from abc import ABCMeta
import numpy as np

from deepscapy.dataset_reader.dataset_reader import DatasetReader

from deepscapy.constants import aes_hd_key


class AESHDDatasetReader(DatasetReader, metaclass=ABCMeta):
    def __init__(self, load_ciphertext=False, load_key=False, **kwargs):
        super(AESHDDatasetReader, self).__init__(dataset_folder='AES_HD', **kwargs)
        self.logger = logging.getLogger(AESHDDatasetReader.__name__)
        self.load_ciphertext = load_ciphertext
        self.load_key = load_key

        self.aes_hd_profiling_traces_file = os.path.join(self.dirname, 'profiling_traces_AES_HD.npy')
        self.aes_hd_profiling_labels_file = os.path.join(self.dirname, 'profiling_labels_AES_HD.npy')
        self.aes_hd_attack_traces_file = os.path.join(self.dirname, 'attack_traces_AES_HD.npy')
        self.aes_hd_attack_labels_file = os.path.join(self.dirname, 'attack_labels_AES_HD.npy')

        if self.load_ciphertext:
            self.aes_hd_profiling_ciphertext_file = os.path.join(self.dirname, 'profiling_ciphertext_AES_HD.npy')
            self.aes_hd_attack_ciphertext_file = os.path.join(self.dirname, 'attack_ciphertext_AES_HD.npy')
        else:
            self.aes_hd_profiling_ciphertext_file = ''
            self.aes_hd_attack_ciphertext_file = ''
        self.key_string = '2b7e151628aed2a6abf7158809cf4f3c'

        self.logger.info("Dataset Folder Path {}".format(self.dirname))
        self.__load_dataset__()

    def __load_dataset__(self):
        self.X_profiling = np.load(self.aes_hd_profiling_traces_file)
        self.Y_profiling = np.load(self.aes_hd_profiling_labels_file)
        self.X_attack = np.load(self.aes_hd_attack_traces_file)
        self.Y_attack = np.load(self.aes_hd_attack_labels_file)
        self.Y_profiling_hw = self.calculate_HW(self.Y_profiling)
        self.Y_attack_hw = self.calculate_HW(self.Y_attack)

        if self.load_ciphertext:
            self.aes_hd_profiling_ciphertext = np.load(self.aes_hd_profiling_ciphertext_file)
            self.aes_hd_attack_ciphertext = np.load(self.aes_hd_attack_ciphertext_file)
        else:
            self.aes_hd_profiling_ciphertext = None
            self.aes_hd_attack_ciphertext = None

        if self.load_key:
            self.key = np.array(aes_hd_key)
        else:
            self.key = None


    def get_plaintext_ciphertext(self):
        return (self.aes_hd_profiling_ciphertext, self.aes_hd_attack_ciphertext)

    def get_key(self):
        return self.key
