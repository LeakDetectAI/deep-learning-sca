import inspect
import logging
import os
from abc import ABCMeta, abstractmethod

import numpy as np

from deepscapy.constants import ID, HW


class DatasetReader(metaclass=ABCMeta):
    def __init__(self, dataset_folder="", leakage_model='HW', **kwargs):
        """
        The generic dataset parser for parsing datasets for solving different learning problems.

        Parameters
        ----------
        dataset_folder: string
            Folder name in the csank/datasets folder in which

        kwargs:
            Keyword arguments for the dataset parser
        """
        self.dr_logger = logging.getLogger("DatasetReader")
        if "pc2" in os.environ["HOME"]:
            dirname = os.path.join(os.environ["PFS_FOLDER"], "deep-learning-sca", "deepscapy", "datasets")
        else:
            dirname = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))).replace(
                "dataset_reader",
                "datasets")
        # if "pc2" in os.environ["HOME"]:
        #    dirname = os.path.join(os.environ["PFS_FOLDER"], "deep-learning-sca", "deepscapy", "datasets")
        # else:
        #    dirname = os.path.join(os.environ["HOME"], "deep-learning-sca", "deepscapy", "datasets")
        if dataset_folder is not None:
            self.dirname = os.path.join(dirname, dataset_folder)
            if not os.path.exists(self.dirname):
                self.dr_logger.warning("Path given for dataset does not exists {}".format(self.dirname))
        else:
            self.dirname = None
        self.leakage_model = leakage_model
        self.dr_logger.info(f"Leakage model {self.leakage_model}")
        if self.leakage_model not in [HW, ID]:
            raise ValueError(f"Wrong Leakage model {self.leakage_model}")
        self.X_profiling = None
        self.X_attack = None
        self.Y_profiling = None
        self.Y_attack = None
        self.Y_profiling_hw = None
        self.Y_attack_hw = None

    @abstractmethod
    def __load_dataset__(self):
        raise NotImplementedError

    @property
    def input_dim(self):
        return self.X_attack.shape[-1]

    @property
    def n_classes(self):
        return len(np.unique(self.Y_attack))

    def get_plaintext_ciphertext(self):
        raise NotImplementedError

    def calculate_HW(self, Y):
        Y_hw = np.array([bin(int(y)).count("1") for y in Y])
        return Y_hw

    def get_train_test_dataset(self):
        if self.leakage_model == 'ID':
            X_profiling, Y_profiling, X_attack, Y_attack = self.X_profiling, self.Y_profiling, self.X_attack, self.Y_attack
        elif self.leakage_model == "HW":
            X_profiling, Y_profiling, X_attack, Y_attack = self.X_profiling, self.Y_profiling_hw, self.X_attack, self.Y_attack_hw
        else:
            raise ValueError(f"Wrong Leakage model {self.leakage_model}")
        return (X_profiling, Y_profiling), (X_attack, Y_attack)
