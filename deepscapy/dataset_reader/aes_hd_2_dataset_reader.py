import logging
import os
import sys
from abc import ABCMeta

import h5py
import numpy as np

from deepscapy.dataset_reader.dataset_reader import DatasetReader
from ..constants import AES_HDV2_DATASETS, AES_HDV2_NORM, AES_HDV2
from ..utils import check_file_exists


class AESHDv2DatasetReader(DatasetReader, metaclass=ABCMeta):
    def __init__(self, dataset_type=AES_HDV2_NORM, load_key=False, load_metadata=False, **kwargs):
        super(AESHDv2DatasetReader, self).__init__(dataset_folder=AES_HDV2, **kwargs)
        self.logger = logging.getLogger(AESHDv2DatasetReader.__name__)
        self.load_metadata = load_metadata
        self.load_key = load_key
        assert (dataset_type in AES_HDV2_DATASETS), f"Dataset Type {dataset_type} does not exist in {AES_HDV2_DATASETS}"
        self.database_file = os.path.join(self.dirname, dataset_type, f"{dataset_type}.h5")

        if self.load_key:
            self.key_file = os.path.join(self.dirname, dataset_type, "key.npy")
        else:
            self.key_file = ''
        self.key_string = "2b7e151628aed2a6abf7158809cf4f3c"
        self.logger.info("Dataset File Path {}".format(self.database_file))
        check_file_exists(self.database_file)
        self.__load_dataset__()

    def __load_dataset__(self):
        try:
            in_file = h5py.File(self.database_file, "r")
        except:
            self.logger.error(
                f"Error: can't open HDF5 file {self.database_file} for reading (it might be malformed) ...")
            sys.exit(-1)
        # Load profiling traces
        self.X_profiling = np.array(in_file['Profiling_traces/traces'], dtype=np.float32)
        self.X_attack = np.array(in_file['Attack_traces/traces'], dtype=np.float32)
        self.Y_profiling = np.array(in_file['Profiling_traces/labels'], dtype=np.int)
        self.Y_attack = np.array(in_file['Attack_traces/labels'], dtype=np.int)
        self.Y_profiling_hw = self.calculate_HW(self.Y_profiling)
        self.Y_attack_hw = self.calculate_HW(self.Y_attack)

        # Load metadata
        if self.load_metadata:
            self.profiling_metadata = in_file['Profiling_traces/metadata']
            self.attack_metadata = in_file['Attack_traces/metadata']
            self.plaintext_profiling = self.profiling_metadata['plaintext']
            self.plaintext_attack = self.attack_metadata['plaintext']
            self.profiling_keys = self.profiling_metadata['key']
            self.attack_keys = self.attack_metadata['key']
            if "ciphertext" in self.profiling_metadata.dtype.fields and "ciphertext" in self.attack_metadata.dtype.fields:
                self.ciphertext_profiling = self.profiling_metadata['ciphertext']
                self.ciphertext_attack = self.attack_metadata['ciphertext']
            else:
                self.ciphertext_profiling, self.ciphertext_attack = None, None
        else:
            self.profiling_metadata = None
            self.attack_metadata = None

        if self.load_key:
            self.key = np.load(self.key_file)
            self.logger.info("Key {}".format(self.key))

        else:
            self.key = None

        in_file.close()

    def get_plaintext_ciphertext(self):
        return self.get_plaintext()

    def get_plaintext(self):
        return self.plaintext_profiling, self.plaintext_attack

    def get_ciphertext(self):
        return self.ciphertext_profiling, self.ciphertext_attack

    def get_metadata(self):
        return self.profiling_metadata, self.attack_metadata

    def get_key(self):
        return self.key

    def get_keys(self):
        return self.profiling_keys, self.attack_keys

    def get_train_test_dataset(self):
        return super(AESHDv2DatasetReader, self).get_train_test_dataset()
