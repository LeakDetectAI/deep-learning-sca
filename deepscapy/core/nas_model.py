import logging
import math
import os

import numpy as np
from keras.saving.save import load_model
from sklearn.base import BaseEstimator, ClassifierMixin
from tensorflow.python.keras.utils.np_utils import to_categorical

from deepscapy.callbacks import PrintTrialModelCallback
from deepscapy.constants import *
from deepscapy.utils import get_trained_models_path


class NASModel(BaseEstimator, ClassifierMixin):
    def __init__(self, model_name, num_classes, input_dim, dataset, loss_function, loss_function_name,
                 model_directory, reshape_type=TWOD_CNN_SQR, leakage_model='ID', max_trials=100,
                 objective='val_accuracy', overwrite=False, metrics=['accuracy'], tuner='greedy',
                 seed=1234, max_model_size=None, **auto_model_kwargs):
        self.num_classes = num_classes
        self.dataset = dataset
        self.input_dim = input_dim
        self.loss_function = loss_function
        self.max_trials = max_trials
        self.objective = objective
        self.overwrite = overwrite
        self.metrics = metrics
        self.tuner = tuner
        self.seed = seed
        self.auto_model_kwargs = auto_model_kwargs

        self.max_model_size = max_model_size
        self.reshape_type = reshape_type
        self.leakage_model = leakage_model
        self.logger = logging.getLogger(NASModel.__name__)

        self.best_model = None

        dataset_dir = "{}_{}_{}".format(self.dataset.lower(), self.input_dim, 'accuracy')
        self.one_dim = False
        reshape_type_dir = 'default'
        if self.reshape_type == TWOD_CNN_SQR:
            reshape_type_dir = 'square'
        elif self.reshape_type == TWOD_CNN_RECT:
            reshape_type_dir = 'rectangle'
        elif self.reshape_type == ONED_CNN:
            reshape_type_dir = 'one_d'
            self.one_dim = True
        if self.leakage_model == HW:
            model_directory = f"{model_directory}_{self.leakage_model.lower()}"

        self.directory = os.path.join(NAS_TRIALS_DIRECTORY_NEW, model_directory, self.tuner, dataset_dir,
                                      reshape_type_dir)

        self.project_name = loss_function_name.lower()

        self.best_model_file_path = os.path.join(get_trained_models_path(folder=TRAINED_MODELS_NAS_NEW),
                                                 f'{model_name}.tf')
        self.logger.info(f"Nas Model Directory {self.directory} Project Name {self.project_name}")
        self.logger.info(f"Best Model stored at {self.best_model_file_path}")

        self.n_filters = [2, 8, 16, 32, 64, 128, 256]
        self.n_units = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
        self.dropout = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        self.kernel_size = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        self.auto_model = self._construct_model_()


    def reshape_inputs(self, X, y):
        features = X.shape[-1]
        image_size = int(math.sqrt(features))

        if image_size ** 2 == features:
            X = X.reshape((-1, image_size, image_size, 1))
        elif self.reshape_type == TWOD_CNN_RECT:
            def get_reshaped_input_dim(features):
                def factor_int(n):
                    val = math.ceil(math.sqrt(n))
                    val2 = int(n / val)
                    while val2 * val != float(n):
                        val -= 1
                        val2 = int(n / val)
                    return val, val2

                s1, s2 = factor_int(features)
                input_shape = (s1, s2, 1)
                return input_shape
            image_size = get_reshaped_input_dim(features)
            s1 = image_size[0]
            s2 = image_size[1]
            X = X.reshape(-1, s1, s2, 1)
        elif self.reshape_type == TWOD_CNN_SQR:
            new_features = int(np.floor(np.sqrt(features)) + 1)
            n = new_features * new_features - features
            constant_values = np.mean(X)
            X = np.lib.pad(X, ((0, 0), (0, n)), 'constant', constant_values=constant_values)
            X = X.reshape(-1, new_features, new_features, 1)
        elif self.reshape_type == ONED_CNN:
            X = X.reshape((X.shape[0], X.shape[1], 1))
        if y is not None:
            y = to_categorical(y, num_classes=self.num_classes)
        return X, y

    def fit(self, X, y, epochs=200, final_model_epochs=200, batch_size=100, validation_split=0.1, verbose=0, **kwargs):
        X, y = self.reshape_inputs(X, y)
        callbacks = [PrintTrialModelCallback()]
        self.logger.info(f"Remaining Trials for the tuner are {self.auto_model.tuner.remaining_trials}")
        self.auto_model.tuner.results_summary()
        self.auto_model.fit(X, y, verbose=verbose, epochs=epochs, batch_size=batch_size,
                            validation_split=validation_split, callbacks=callbacks, **kwargs)
        if str(self.loss_function) != 'categorical_crossentropy':
            model_path = os.path.join(self.directory, self.project_name, 'best_model')
            custom_objects = {'loss': self.loss_function}
            self.best_model = load_model(model_path, custom_objects=custom_objects)
            # self.best_model.compile(loss=self.loss_function, metrics=self.metrics)
        else:
            self.best_model = self.auto_model.export_model()
        if self.tuner in [GREEDY_TUNER, RANDOM_TUNER, BAYESIAN_TUNER]:
            self.best_model.fit(X, y, verbose=1, epochs=final_model_epochs, batch_size=batch_size, callbacks=callbacks)
        self.best_model.save(self.best_model_file_path)
        return self

    def predict_scores(self, X, verbose=0, **kwargs):
        X, _ = self.reshape_inputs(X, None)
        predictions = self.best_model.predict(x=X, verbose=verbose, batch_size=X.shape[0], **kwargs)
        return predictions

    def evaluate(self, X, y, verbose=0, **kwargs):
        X, y = self.reshape_inputs(X, y)
        best_model_metrics = self.best_model.evaluate(X, y, verbose=verbose, batch_size=X.shape[0], **kwargs)
        return best_model_metrics

    def summary(self, **kwargs):
        self.best_model.summary(**kwargs)

    def search_space_summary(self):
        """Print search space summary.

        The methods print a summary of the hyperparameters in the search
        space, which can be called before calling the `search` method.
        """
        self.logger.info("Search space summary")
        hp = self.auto_model.tuner.oracle.get_space()
        self.logger.info("Default search space size: %d" % len(hp.space))
        for p in hp.space:
            config = p.get_config()
            name = config.pop("name")
            self.logger.info("%s (%s)" % (name, p.__class__.__name__))
            self.logger.info(config)

    def _construct_model_(self):
        raise NotImplemented
