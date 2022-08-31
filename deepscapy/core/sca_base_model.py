from abc import ABCMeta, abstractmethod

from sklearn.base import BaseEstimator, ClassifierMixin


class SCABaseModel(BaseEstimator, ClassifierMixin, metaclass=ABCMeta):
    @abstractmethod
    def _construct_model_(self, kwargs):
        pass

    @abstractmethod
    def fit(self, X, y, epochs, batch_size, verbose, kwargs):
        pass

    @abstractmethod
    def predict(self, X, verbose, batch_size, kwargs):
        pass

    @abstractmethod
    def predict_scores(self, X, verbose, batch_size, kwargs):
        pass

    @abstractmethod
    def evaluate(self, X, y, verbose, batch_size, kwargs):
        pass

    @abstractmethod
    def summary(self, kwargs):
        pass
